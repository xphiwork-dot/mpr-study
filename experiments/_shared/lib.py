"""Shared helpers for the Colab T4 experiments (runs on the VM and locally).

Stdlib only at import time; numpy / bm25s / torch / sentence-transformers are imported lazily.
Upstream prompts are loaded from the pinned clone (never copied into this repo).
"""
import os as _os
_os.environ.setdefault("JAX_PLATFORMS", "cpu")  # bm25s may import jax, which would preallocate most of the GPU
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import gzip
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request


# Locations. On the VM everything lives under /content/t4; locally under the repo.
ROOT = os.environ.get("T4_ROOT", "/content/t4")
DATA_DIR = os.environ.get("MPR_QA_DIR", os.path.join(ROOT, "data", "mpr_dataset", "question_answer"))
UPSTREAM_PROMPTS = os.environ.get("MPR_PROMPTS", os.path.join(ROOT, "upstream", "MPR", "baselines", "prompts.py"))
CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "results")
USERS = list(range(12))
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = "qwen"  # --served-model-name of the vLLM server


def load_user(u):
    with open(os.path.join(DATA_DIR, f"question_answer_{u}.json"), encoding="utf-8") as f:
        return json.load(f)


def all_questions(data):
    """[(hop:int, rc_id, qa)] with a question and an answer (upstream has one null question)."""
    rows = []
    for h in sorted(data["qa"], key=int):
        for rc in sorted(data["qa"][h]):
            qa = data["qa"][h][rc]
            if qa.get("question") and qa.get("answer"):
                rows.append((int(h), rc, qa))
    return rows


def load_prompts():
    spec = importlib.util.spec_from_file_location("mpr_prompts", UPSTREAM_PROMPTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- jsonl with resume ----------

def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line cut by a dying session; that item is simply redone
    return rows


class JsonlWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, obj):
        with self.lock:
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())


def qkey(row):
    return f"{row['uid']}:{row['rc_id']}"


# ---------- vLLM (OpenAI-compatible) ----------

def chat(prompt, max_tokens, retries=5, temperature=0):
    """One user turn, greedy decoding. Returns (text, prompt_tokens, completion_tokens, seconds)."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    last = None
    for attempt in range(retries):
        t = time.time()
        try:
            req = urllib.request.Request(VLLM_URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            u = d["usage"]
            return d["choices"][0]["message"]["content"].strip(), u["prompt_tokens"], u["completion_tokens"], time.time() - t
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"vLLM request failed: {last}")


# ---------- retrieval ----------

class UserIndex:
    """BM25 + dense e5 index for one user's memory. Dense embeddings come from the retrieve step cache."""

    def __init__(self, uid, data, need_dense=True, encoder=None):
        import numpy as np
        self.uid = uid
        self.ids = list(data["message"])
        self.texts = [data["message"][i] for i in self.ids]
        self.messages = data["message"]
        import bm25s
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(self.texts, stopwords="en", show_progress=False), show_progress=False)
        self.emb = None
        self.encoder = encoder
        if need_dense:
            self.emb = np.load(os.path.join(CACHE, f"emb_user{uid}.npy")).astype("float32")

    def add_dense(self, encoder):
        """Attach the cached embeddings and the encoder to an index first built for BM25 only."""
        import numpy as np
        self.encoder = encoder
        self.emb = np.load(os.path.join(CACHE, f"emb_user{self.uid}.npy")).astype("float32")

    def sparse(self, query, k):
        import bm25s
        res, _ = self.bm25.retrieve(bm25s.tokenize(query, stopwords="en", show_progress=False), k=k, show_progress=False)
        return [self.ids[j] for j in res[0]]

    def dense(self, query, k):
        import numpy as np
        q = self.encoder.encode([query], normalize_embeddings=True)[0].astype("float32")
        top = np.argsort(-(self.emb @ q))[:k]
        return [self.ids[j] for j in top]


def get_encoder(model_name="intfloat/e5-base-v2", device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = SentenceTransformer(model_name, device=dev)
    if dev == "cuda":
        enc.half()
    return enc


# ---------- slices ----------

def load_slice(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_gz_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def now():
    return time.time()
