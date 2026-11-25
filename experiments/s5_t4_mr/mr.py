#!/usr/bin/env python3
"""Faithful MR / StructuredReason runner for the private Kaggle T4 experiment.

This is an ADAPTED re-implementation of upstream
``reasonings/StructuredReason.py`` at nuster1128/MPR commit
``a24953c8f342c9f3b2064e562ead49d477d49138``.  It keeps the upstream call
flow, prompt parameter names, path formatting, buffer update and retrieval keys;
differences are deliberate and documented in README.md / report.md:

* branch-generation calls use seeded sampling (T=0.7, top_p=1.0), selection,
  the initial thought and the final answer stay greedy.  The upstream local
  server decoding settings are not shipped; its hosted-client path uses T=0.9.
* selection parsing uses ASCII digits and ``int`` rather than ``eval``; we strip
  surrounding whitespace and fall back to path 0 for anything else, including
  out-of-range indices (upstream uses ``str.isdigit()`` + ``eval`` and the same
  fallback for non-digits/out-of-range).
* empty responses are retried a bounded number of times.  An empty selection
  falls back to path 0 like upstream's ``'None'`` result; an empty thought or
  answer fails the question.
* every call is capped (200 tokens for thoughts, 64 for the answer) and has a
  per-question deadline / global deadline.
* no gold answer is ever passed to the runner; it is attached only after the
  answer call returns.

The module deliberately imports only ``lib`` (stdlib) at import time; numpy,
torch and sentence-transformers are imported lazily in the embedding helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_lib_dir() -> str:
    for d in (HERE, os.path.join(HERE, "..", "_shared"), os.path.join(HERE, "code")):
        if os.path.isfile(os.path.join(d, "lib.py")):
            return os.path.abspath(d)
    raise RuntimeError("lib.py not found next to mr.py or in ../_shared")


LIB_DIR = _find_lib_dir()
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)
import lib  # noqa: E402  (stdlib only at import time)

UPSTREAM_COMMIT = "a24953c8f342c9f3b2064e562ead49d477d49138"
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
E5_MODEL = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
MEMORIES = {"sparse": ("SparseRAG", "bm25"), "dense": ("DenseRAG", "e5")}
DEFAULT_VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("MR_SERVED_MODEL", "qwen")
SELECTION_RE = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class MRConfig:
    """Frozen method settings.  Do not tune on the evaluation slice."""

    max_step: int = 5
    branch_size: int = 2
    topk: int = 20
    max_thought_tokens: int = 200
    max_answer_tokens: int = 64
    branch_temperature: float = 0.7
    branch_top_p: float = 1.0
    greedy_temperature: float = 0.0
    experiment_seed: int = 0
    empty_retries: int = 2

    @property
    def rounds(self) -> int:
        # Upstream: range(1, max_step - 1), i.e. max_step - 2 middle rounds.
        return max(0, self.max_step - 2)

    @property
    def expected_calls(self) -> int:
        # 1 start + rounds * (1 selection + branch_size candidates) + 1 final selection + 1 answer.
        return 1 + self.rounds * (1 + self.branch_size) + 1 + 1


class MRFailure(RuntimeError):
    """A single (question, memory) could not be completed."""


# ---------------------------------------------------------------------------
# deterministic helpers


def stable_seed(
    experiment_seed: int, uid: int, rc_id: str, round_idx: int, branch_idx: int
) -> int:
    """Stable per-(question, round, branch) seed, independent of thread order."""
    raw = f"mpr-mr|{experiment_seed}|{uid}|{rc_id}|{round_idx}|{branch_idx}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def parse_selection(text: str, n_paths: int) -> tuple[int, str]:
    """Return (index, status) with the upstream fallback-to-0 semantics.

    Deliberate parsing differences from upstream are limited to stripping
    surrounding whitespace and accepting ASCII digits only; ``eval`` is never
    used.  Status values: ok, fallback_non_digit, fallback_out_of_range.
    """
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if SELECTION_RE.fullmatch(stripped):
        idx = int(stripped)
        if 0 <= idx < n_paths:
            return idx, "ok"
        return 0, "fallback_out_of_range"
    return 0, "fallback_non_digit"


def format_thoughts(path) -> str:
    return "\n".join("(Step %d) %s" % (i + 1, t) for i, t in enumerate(path))


def format_paths(path_buffer) -> str:
    return "\n".join(
        "Path %d: %s"
        % (pid, " ".join("(Step %d) %s" % (sid + 1, t) for sid, t in enumerate(path)))
        for pid, path in enumerate(path_buffer)
    )


def _norm_dup(text: str) -> str:
    return " ".join(str(text).lower().split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hashes(prompts) -> dict:
    """Hashes/identifiers of the upstream templates this run uses (no text stored)."""
    out = {}
    for name in (
        "Structured_Start_RAG_Prompt_Template",
        "Structured_Middle_RAG_Prompt_Template",
        "Structured_Final_RAG_Prompt_Template",
        "Structured_Selection_RAG_Prompt_Template",
        "Naive_RAG_Prompt_Template",
    ):
        text = getattr(prompts, name)
        out[name] = {
            "sha256": sha256_text(text),
            "chars": len(text),
            "lines": text.count("\n") + 1,
        }
    return out


# ---------------------------------------------------------------------------
# vLLM (OpenAI-compatible) client


def vllm_chat(
    prompt: str,
    max_tokens: int,
    *,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
    url: str | None = None,
    model: str | None = None,
    timeout: int = 900,
    retries: int = 5,
) -> dict:
    """One user turn against the local vLLM server.

    Returns text, token counts, finish reason, seconds and the actual sampling
    parameters.  Transport errors, 429 and 5xx are retried with backoff; a 4xx
    request error is raised immediately (fail fast, do not hide a bad request).
    """
    url = (url or DEFAULT_VLLM_URL).rstrip("/")
    model = model or DEFAULT_MODEL
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        body["top_p"] = float(top_p)
    if seed is not None:
        body["seed"] = int(seed)
    payload = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                url + "/v1/chat/completions",
                payload,
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            choice = d["choices"][0]
            usage = d.get("usage") or {}
            text = choice.get("message", {}).get("content")
            return {
                "text": "" if text is None else str(text),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "finish_reason": choice.get("finish_reason"),
                "seconds": time.time() - t0,
                "temperature": float(temperature),
                "top_p": None if top_p is None else float(top_p),
                "seed": None if seed is None else int(seed),
            }
        except urllib.error.HTTPError as e:  # bad request must not be retried forever
            detail = e.read()[:800].decode("utf-8", "replace")
            if 400 <= e.code < 500 and e.code not in (408, 409, 429):
                raise MRFailure(f"vLLM HTTP {e.code}: {detail}") from e
            last = f"HTTP {e.code}: {detail}"
        except Exception as e:  # noqa: BLE001 - transport errors are retried
            last = repr(e)
        time.sleep(min(20.0, 2.0 * (attempt + 1)))
    raise MRFailure(f"vLLM request failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------------
# StructuredReason call flow


class MRRunner:
    """One question through upstream's StructuredReason call flow."""

    def __init__(self, config: MRConfig, prompts, chat_fn, retrieve_fn, memory: str):
        if memory not in MEMORIES:
            raise ValueError(memory)
        self.cfg = config
        self.P = prompts
        self.chat_fn = chat_fn
        self.retrieve_fn = retrieve_fn
        self.memory = memory

    def _generate(
        self,
        kind: str,
        params: dict,
        max_tokens: int,
        *,
        sampling: bool,
        uid: int,
        rc_id: str,
        round_idx: int | None,
        branch_idx: int | None,
        retrieval_query: str | None,
        retrieved_ids,
        trace: dict,
    ) -> str:
        name = f"{kind}_RAG_Prompt_Template"
        prompt = getattr(self.P, name).format(**params)
        if sampling:
            seed = stable_seed(
                self.cfg.experiment_seed, uid, rc_id, int(round_idx), int(branch_idx)
            )
            temperature, top_p = self.cfg.branch_temperature, self.cfg.branch_top_p
        else:
            seed, temperature, top_p = None, self.cfg.greedy_temperature, None

        empties = 0
        t0 = time.time()
        while True:
            out = self.chat_fn(
                prompt, max_tokens, temperature=temperature, top_p=top_p, seed=seed
            )
            text = out.get("text", "")
            if text.strip():
                break
            empties += 1
            if empties > self.cfg.empty_retries:
                if kind == "Structured_Selection":
                    # Upstream fast_run returns 'None' after its retries, which
                    # path_selection treats as a non-digit fallback to path 0.
                    text = ""
                    break
                raise MRFailure(f"empty {kind} response after {empties} attempts")
        elapsed = time.time() - t0
        event = {
            "kind": kind,
            "template": name,
            "prompt_sha256": sha256_text(prompt),
            "prompt_chars": len(prompt),
            "retrieval_query": retrieval_query,
            "retrieved_ids": retrieved_ids,
            "response": text,
            "prompt_tokens": int(out.get("prompt_tokens") or 0),
            "completion_tokens": int(out.get("completion_tokens") or 0),
            "finish_reason": out.get("finish_reason"),
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
            "empty_retries": empties,
            "seconds": elapsed,
            "round": round_idx,
            "branch": branch_idx,
        }
        trace["events"].append(event)
        trace["stats"]["calls"] += 1
        trace["stats"]["prompt_tokens"] += event["prompt_tokens"]
        trace["stats"]["completion_tokens"] += event["completion_tokens"]
        trace["stats"]["seconds"] += elapsed
        if event["completion_tokens"] >= max_tokens:
            trace["stats"]["truncations"][kind] = (
                trace["stats"]["truncations"].get(kind, 0) + 1
            )
        return text

    def _select(self, path_buffer, question, uid, rc_id, round_idx, trace):
        paths = format_paths(path_buffer)
        raw = self._generate(
            "Structured_Selection",
            {"question": question, "reasoning_paths": paths},
            self.cfg.max_answer_tokens,
            sampling=False,
            uid=uid,
            rc_id=rc_id,
            round_idx=round_idx,
            branch_idx=None,
            retrieval_query=None,
            retrieved_ids=None,
            trace=trace,
        )
        idx, status = parse_selection(raw, len(path_buffer))
        trace["selections"].append(
            {
                "round": round_idx,
                "buffer_size": len(path_buffer),
                "buffer_depths": [len(p) for p in path_buffer],
                "raw": raw,
                "selected": idx,
                "status": status,
            }
        )
        if status != "ok":
            trace["stats"]["selection_fallbacks"] += 1
        return idx

    def solve(self, uid: int, rc_id: str, question: str) -> dict:
        cfg = self.cfg
        trace = {
            "events": [],
            "selections": [],
            "candidate_pairs": [],
            "path_buffer_sizes": [],
            "path_buffer_depths": [],
            "selected_path_ids": [],
            "selected_path_depths": [],
            "stats": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "seconds": 0.0,
                "selection_fallbacks": 0,
                "truncations": {},
            },
            "settings": asdict(cfg),
            "memory": self.memory,
            "upstream": {
                "repo": "nuster1128/MPR",
                "commit": UPSTREAM_COMMIT,
                "class": "reasonings.StructuredReason",
            },
        }

        # Start thought: retrieval key = question.
        ids, refs = self.retrieve_fn(uid, question)
        first_ids = list(ids)
        start = self._generate(
            "Structured_Start",
            {"question": question, "references": refs},
            cfg.max_thought_tokens,
            sampling=False,
            uid=uid,
            rc_id=rc_id,
            round_idx=None,
            branch_idx=None,
            retrieval_query=question,
            retrieved_ids=ids,
            trace=trace,
        )
        path_buffer = [[start]]

        # Middle rounds: select from the *entire* current buffer, then append
        # branch_size children of the selected path.  Old paths are retained.
        for current_step in range(1, cfg.max_step - 1):
            pid = self._select(path_buffer, question, uid, rc_id, current_step, trace)
            trace["selected_path_ids"].append(pid)
            trace["selected_path_depths"].append(len(path_buffer[pid]))
            current_path = path_buffer[pid]
            previous_thoughts = format_thoughts(current_path)
            new_path_list = []
            new_thoughts_by_branch = []
            for branch_id in range(cfg.branch_size):
                ids, refs = self.retrieve_fn(uid, current_path[-1])
                thought = self._generate(
                    "Structured_Middle",
                    {
                        "question": question,
                        "references": refs,
                        "previous_thoughts": previous_thoughts,
                    },
                    cfg.max_thought_tokens,
                    sampling=True,
                    uid=uid,
                    rc_id=rc_id,
                    round_idx=current_step,
                    branch_idx=branch_id,
                    retrieval_query=current_path[-1],
                    retrieved_ids=ids,
                    trace=trace,
                )
                new_path_list.append(current_path + [thought])
                new_thoughts_by_branch.append(thought)
            if len(new_thoughts_by_branch) > 1:
                duplicate = _norm_dup(new_thoughts_by_branch[0]) == _norm_dup(
                    new_thoughts_by_branch[1]
                )
                trace["candidate_pairs"].append(
                    {
                        "round": current_step,
                        "duplicate": duplicate,
                        "branch0_sha256": sha256_text(new_thoughts_by_branch[0]),
                        "branch1_sha256": sha256_text(new_thoughts_by_branch[1]),
                    }
                )
            # Exactly upstream line 45: path_buffer[:pid] + path_buffer[pid:] + new_path_list.
            path_buffer = path_buffer[:pid] + path_buffer[pid:] + new_path_list
            trace["path_buffer_sizes"].append(len(path_buffer))
            trace["path_buffer_depths"].append([len(p) for p in path_buffer])

        # Final selection (inference only), then retrieval on the selected path's
        # last thought and the final answer.
        pid = self._select(path_buffer, question, uid, rc_id, cfg.max_step - 1, trace)
        trace["selected_path_ids"].append(pid)
        trace["selected_path_depths"].append(len(path_buffer[pid]))
        final_path = path_buffer[pid]
        ids, refs = self.retrieve_fn(uid, final_path[-1])
        answer = self._generate(
            "Structured_Final",
            {
                "question": question,
                "references": refs,
                "previous_thoughts": format_thoughts(final_path),
            },
            cfg.max_answer_tokens,
            sampling=False,
            uid=uid,
            rc_id=rc_id,
            round_idx=cfg.max_step - 1,
            branch_idx=None,
            retrieval_query=final_path[-1],
            retrieved_ids=ids,
            trace=trace,
        )

        trace["first_retrieved"] = first_ids
        trace["final_path"] = final_path
        trace["final_path_depth"] = len(final_path)
        trace["final_retrieved"] = list(ids)
        pairs = trace["candidate_pairs"]
        trace["duplicate_candidate_pairs"] = sum(1 for p in pairs if p["duplicate"])
        trace["candidate_pairs_total"] = len(pairs)
        trace["expected_calls"] = cfg.expected_calls
        if trace["stats"]["calls"] != cfg.expected_calls:
            raise MRFailure(
                f"internal call-count mismatch: {trace['stats']['calls']} != {cfg.expected_calls}"
            )
        return {"response": answer, "trace": trace}


# ---------------------------------------------------------------------------
# retrieval / memory


def get_encoder_mr(device: str | None = None):
    """e5 encoder at the pinned HF revision; no query:/passage: prefixes."""
    import torch  # local import: no CUDA context in the launcher
    from sentence_transformers import SentenceTransformer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = SentenceTransformer(E5_MODEL, revision=E5_REVISION, device=dev)
    if str(dev).startswith("cuda"):
        enc.half()
    return enc


class MemoryStore:
    """Per-user BM25/e5 index over the released statement memory."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        topk: int,
        dense: bool,
        device: str | None = None,
    ):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.topk = topk
        self.dense = dense
        self.device = device
        self.data = {}
        self.index = {}
        self.lock = threading.Lock()
        self.enc = None
        self.enc_lock = threading.Lock()
        lib.DATA_DIR = data_dir
        lib.CACHE = cache_dir

    def user(self, uid: int):
        with self.lock:
            if uid not in self.data:
                self.data[uid] = lib.load_user(uid)
            if uid not in self.index:
                if self.dense and self.enc is None:
                    self.enc = get_encoder_mr(self.device)
                self.index[uid] = lib.UserIndex(
                    uid, self.data[uid], need_dense=self.dense, encoder=self.enc
                )
            return self.data[uid], self.index[uid]

    def refs(self, uid: int, query: str):
        _, ix = self.user(uid)
        if self.dense:
            with self.enc_lock:
                ids = ix.dense(query, self.topk)
        else:
            ids = ix.sparse(query, self.topk)
        return ids, "\n".join(ix.messages[i] for i in ids)


def prepare_embeddings(
    data_dir: str, cache_dir: str, users, device: str | None, batch_size: int = 256
) -> dict:
    """Build the ordered e5 embedding cache, matching retrieve.py's convention."""
    import numpy as np

    os.makedirs(cache_dir, exist_ok=True)
    lib.DATA_DIR = data_dir
    lib.CACHE = cache_dir
    enc = get_encoder_mr(device)
    manifest = {
        "model": E5_MODEL,
        "revision": E5_REVISION,
        "normalize_embeddings": True,
        "prefixes": None,
        "dtype": "float16",
        "batch_size": batch_size,
        "device": str(device or ("cuda" if hasattr(enc, "device") else "cpu")),
        "users": {},
    }
    for uid in users:
        d = lib.load_user(uid)
        ids = list(d["message"])
        path = os.path.join(cache_dir, f"emb_user{uid}.npy")
        if os.path.exists(path):
            emb = np.load(path)
            if emb.shape[0] != len(ids):
                raise MRFailure(
                    f"embedding cache {path} has {emb.shape[0]} rows for {len(ids)} statements"
                )
        else:
            texts = [d["message"][i] for i in ids]
            t0 = time.time()
            emb = enc.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ).astype("float16")
            tmp = path + ".tmp.npy"
            np.save(tmp, emb)
            os.replace(tmp, path)
            print(
                f"[prepare] user {uid}: {len(ids)} statements in {time.time() - t0:.1f}s",
                flush=True,
            )
        with open(path, "rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        manifest["users"][str(uid)] = {
            "rows": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "sha256": h.hexdigest(),
            "path": f"emb_user{uid}.npy",
        }
    out = os.path.join(cache_dir, "embeddings_manifest.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out)
    return manifest


# ---------------------------------------------------------------------------
# data / rows


def row_key(row) -> str:
    return f"{row['uid']}:{row['rc_id']}"


def load_valid_rows(path: str) -> dict:
    """Deduplicated successful rows keyed by uid:rc_id; cut tails are ignored."""
    rows = {}
    for row in lib.read_jsonl(path):
        if not isinstance(row, dict) or "uid" not in row or "rc_id" not in row:
            continue
        if not str(row.get("response", "")).strip() or int(row.get("calls") or 0) <= 0:
            continue
        rows[row_key(row)] = row
    return rows


def smoke_items(items):
    """Three fixed questions, one from each hop group (2-4, 5-7, 8-10)."""
    out = []
    used_users = set()
    for target_hop in (2, 6, 10):
        candidates = sorted(
            (i for i in items if int(i["hop"]) == target_hop),
            key=lambda x: (x["uid"], x["rc_id"]),
        )
        if not candidates:
            raise MRFailure(f"smoke hop {target_hop} missing from slice")
        pick = next(
            (i for i in candidates if i["uid"] not in used_users), candidates[0]
        )
        used_users.add(pick["uid"])
        out.append(pick)
    return out


def make_row(memory_label: str, item, qa, result: dict) -> dict:
    trace = result["trace"]
    stats = trace["stats"]
    return {
        "uid": item["uid"],
        "rc_id": item["rc_id"],
        "k_hop": int(item["hop"]),
        "memory": memory_label,
        "response": result["response"],
        "ground_truth": qa["answer"],
        "first_retrieved": trace["first_retrieved"],
        "calls": stats["calls"],
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "secs": round(float(stats["seconds"]), 3),
        "thoughts": trace["final_path"],
        "trace": trace,
    }


def _append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_json_atomic(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _parse_users(text: str) -> list[int]:
    if text == "all":
        return list(lib.USERS)
    return [int(x) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# commands


def cmd_prepare(args):
    users = _parse_users(args.users)
    manifest = prepare_embeddings(
        args.data_dir, args.cache_dir, users, args.device, batch_size=args.batch_size
    )
    _write_json_atomic(
        os.path.join(args.output_dir, "embeddings_manifest.json"), manifest
    )
    print(
        json.dumps({"prepared_users": users, "cache_dir": args.cache_dir}, indent=2),
        flush=True,
    )
    return 0


def cmd_run(args):
    cfg = MRConfig(
        max_step=args.max_step,
        branch_size=args.branch_size,
        topk=args.topk,
        experiment_seed=args.seed,
    )
    if cfg.expected_calls != args.expected_calls:
        raise MRFailure(
            f"expected-calls setting {args.expected_calls} != derived {cfg.expected_calls}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    items = lib.load_slice(args.slice)["items"]
    if args.stage == "smoke":
        items = smoke_items(items)
    if args.limit:
        items = items[: args.limit]
    prompts = lib.load_prompts()
    memories = [m.strip() for m in args.memories.split(",") if m.strip()]
    for m in memories:
        if m not in MEMORIES:
            raise MRFailure(f"unknown memory {m}")
    deadline = float(args.deadline or 0) or None
    global_failures = 0
    deadline_hit = False
    t_all = time.time()
    for mem in memories:
        label = MEMORIES[mem][0]
        name = f"StructuredReason_{label}_k{cfg.topk}_s{cfg.max_step}"
        path = os.path.join(args.output_dir, name + ".jsonl")
        done = load_valid_rows(path)
        todo = [i for i in items if f"{i['uid']}:{i['rc_id']}" not in done]
        print(
            f"[{name}] {len(done)} done, {len(todo)} to do, stage={args.stage}",
            flush=True,
        )
        if not todo:
            continue
        store = MemoryStore(
            args.data_dir,
            args.cache_dir,
            cfg.topk,
            dense=(mem == "dense"),
            device=args.device,
        )
        chat_fn = lambda p, mt, temperature, top_p, seed: vllm_chat(  # noqa: E731
            p,
            mt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            url=args.vllm_url,
            model=args.model,
            timeout=args.request_timeout,
        )
        runner = MRRunner(cfg, prompts, chat_fn, store.refs, memory=mem)
        timing_path = os.path.join(args.output_dir, "timing.jsonl")
        fail_path = os.path.join(args.output_dir, "failures.jsonl")
        done_count = calls = pt = ct = 0
        failures = 0
        t_mem = time.time()
        pending = [(item, 1) for item in todo]
        while pending:
            if deadline and time.time() > deadline:
                deadline_hit = True
                print(
                    f"[{name}] deadline reached, stopping with {len(pending)} unstarted",
                    flush=True,
                )
                break
            batch = pending[: args.workers]
            pending = pending[args.workers :]
            with ThreadPoolExecutor(max_workers=min(args.workers, len(batch))) as ex:
                futs = {}
                for item, attempt in batch:
                    data, _ = store.user(
                        item["uid"]
                    )  # build index outside worker burst
                    qa = data["qa"][str(item["hop"])][item["rc_id"]]
                    futs[
                        ex.submit(
                            runner.solve, item["uid"], item["rc_id"], qa["question"]
                        )
                    ] = (item, qa, attempt)
                for fut in as_completed(futs):
                    item, qa, attempt = futs[fut]
                    try:
                        result = fut.result()
                        row = make_row(label, item, qa, result)
                        _append_jsonl(path, row)
                        done_count += 1
                        calls += row["calls"]
                        pt += row["prompt_tokens"]
                        ct += row["completion_tokens"]
                    except Exception as e:  # noqa: BLE001 - record and retry once
                        failures += 1
                        _append_jsonl(
                            fail_path,
                            {
                                "uid": item["uid"],
                                "rc_id": item["rc_id"],
                                "k_hop": item["hop"],
                                "memory": label,
                                "attempt": attempt,
                                "stage": args.stage,
                                "error": repr(e),
                                "traceback": traceback.format_exc(),
                                "time": time.time(),
                            },
                        )
                        print(
                            f"  failure {item['uid']}:{item['rc_id']} attempt {attempt}: {e}",
                            flush=True,
                        )
                        if attempt < args.max_attempts:
                            pending.append((item, attempt + 1))
                        else:
                            global_failures += 1
            print(
                f"  [{name}] {done_count}/{len(todo)} done, {failures} failures",
                flush=True,
            )
        wall = time.time() - t_mem
        _append_jsonl(
            timing_path,
            {
                "stage": args.stage,
                "file": name,
                "questions_done": done_count,
                "questions_target": len(todo),
                "calls": calls,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "wall_s": round(wall, 1),
                "workers": args.workers,
                "failures": failures,
                "deadline_hit": deadline_hit,
                "time": time.time(),
                "settings": asdict(cfg),
            },
        )
        print(
            f"[{name}] stage {args.stage} done in {wall:.0f}s, {done_count} rows, "
            f"{calls} calls, {failures} failures",
            flush=True,
        )
    print(
        f"[run] total {time.time() - t_all:.0f}s, failures={global_failures}, "
        f"deadline_hit={deadline_hit}",
        flush=True,
    )
    coverage = coverage_report(args.output_dir, items, args.data_dir, memories)
    complete = all(v["complete"] for v in coverage["memories"].values())
    return 0 if complete else 3


def cmd_sanity(args):
    """One-call NaiveReason deployment check on the fixed smoke questions."""
    items = smoke_items(lib.load_slice(args.slice)["items"])
    prompts = lib.load_prompts()
    cfg = MRConfig(experiment_seed=args.seed, topk=args.topk)
    os.makedirs(args.output_dir, exist_ok=True)
    for mem in args.memories.split(","):
        mem = mem.strip()
        label = MEMORIES[mem][0]
        path = os.path.join(args.output_dir, f"NaiveReason_{label}_k{cfg.topk}.jsonl")
        done = load_valid_rows(path)
        todo = [i for i in items if f"{i['uid']}:{i['rc_id']}" not in done]
        if not todo:
            continue
        store = MemoryStore(
            args.data_dir,
            args.cache_dir,
            cfg.topk,
            dense=(mem == "dense"),
            device=args.device,
        )
        for item in todo:
            data, _ = store.user(item["uid"])
            qa = data["qa"][str(item["hop"])][item["rc_id"]]
            ids, refs = store.refs(item["uid"], qa["question"])
            prompt = getattr(prompts, "Naive_RAG_Prompt_Template").format(
                question=qa["question"], references=refs
            )
            out = vllm_chat(
                prompt,
                cfg.max_answer_tokens,
                temperature=0.0,
                seed=None,
                url=args.vllm_url,
                model=args.model,
                timeout=args.request_timeout,
            )
            row = {
                "uid": item["uid"],
                "rc_id": item["rc_id"],
                "k_hop": int(item["hop"]),
                "memory": label,
                "response": out["text"],
                "ground_truth": qa["answer"],
                "first_retrieved": ids,
                "calls": 1,
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "secs": round(out["seconds"], 3),
                "prompt_sha256": sha256_text(prompt),
                "stage": "sanity",
            }
            _append_jsonl(path, row)
    return 0


def _coverage_one(path, items, data_dir, label):
    lib.DATA_DIR = data_dir
    rows = load_valid_rows(path)
    expected = {f"{i['uid']}:{i['rc_id']}" for i in items}
    got = set(rows)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    wrong_calls = []
    wrong_gt = []
    wrong_hop = []
    cache = {}
    for k, r in rows.items():
        if int(r.get("calls") or 0) != MRConfig().expected_calls:
            wrong_calls.append(k)
        uid = int(r["uid"])
        if uid not in cache:
            cache[uid] = lib.load_user(uid)
        try:
            qa = cache[uid]["qa"][str(r["k_hop"])][r["rc_id"]]
        except KeyError:
            wrong_gt.append(k)
            continue
        if str(r.get("ground_truth")) != str(qa.get("answer")):
            wrong_gt.append(k)
        item = next((i for i in items if f"{i['uid']}:{i['rc_id']}" == k), None)
        if item is None or int(r["k_hop"]) != int(item["hop"]):
            wrong_hop.append(k)
    return {
        "memory": label,
        "rows": len(rows),
        "expected": len(expected),
        "missing": missing[:20],
        "missing_count": len(missing),
        "extra": extra[:20],
        "extra_count": len(extra),
        "wrong_call_count": wrong_calls[:20],
        "wrong_call_count_n": len(wrong_calls),
        "wrong_ground_truth": wrong_gt[:20],
        "wrong_ground_truth_n": len(wrong_gt),
        "wrong_hop_n": len(wrong_hop),
        "complete": not missing
        and not extra
        and not wrong_calls
        and not wrong_gt
        and not wrong_hop
        and len(rows) == len(expected),
    }


def coverage_report(output_dir: str, items, data_dir: str, memories) -> dict:
    out = {"expected": len(items), "memories": {}}
    for mem in memories:
        label = MEMORIES[mem][0]
        path = os.path.join(output_dir, f"StructuredReason_{label}_k20_s5.jsonl")
        out["memories"][mem] = _coverage_one(path, items, data_dir, label)
    return out


def smoke_gate_report(output_dir: str, items, data_dir: str) -> dict:
    """Pre-registered smoke gates; must pass before the full 306x2 run."""
    cov = coverage_report(output_dir, items, data_dir, ["sparse", "dense"])
    reasons = []
    stats = {
        "rows": 0,
        "calls": 0,
        "duplicate_pairs": 0,
        "candidate_pairs": 0,
        "selection_fallbacks": 0,
        "selections": 0,
        "empty_responses": 0,
        "retrieval_calls": 0,
        "retrieval_empty": 0,
    }
    for mem in ("sparse", "dense"):
        label = MEMORIES[mem][0]
        path = os.path.join(output_dir, f"StructuredReason_{label}_k20_s5.jsonl")
        rows = load_valid_rows(path)
        for r in rows.values():
            tr = r.get("trace") or {}
            st = tr.get("stats") or {}
            stats["rows"] += 1
            stats["calls"] += int(r.get("calls") or 0)
            stats["duplicate_pairs"] += int(tr.get("duplicate_candidate_pairs") or 0)
            stats["candidate_pairs"] += int(tr.get("candidate_pairs_total") or 0)
            stats["selection_fallbacks"] += int(st.get("selection_fallbacks") or 0)
            stats["selections"] += len(tr.get("selections") or [])
            for ev in tr.get("events") or []:
                if ev.get("retrieval_query") is not None:
                    stats["retrieval_calls"] += 1
                    if not ev.get("retrieved_ids"):
                        stats["retrieval_empty"] += 1
                if (
                    not str(ev.get("response", "")).strip()
                    and ev.get("kind") != "Structured_Selection"
                ):
                    stats["empty_responses"] += 1
        if not cov["memories"][mem]["complete"]:
            reasons.append(f"{mem}: incomplete rows/uniqueness/ground truth")
        if any(not str(r.get("response", "")).strip() for r in rows.values()):
            reasons.append(f"{mem}: empty final answer")
    if stats["rows"] != 6:
        reasons.append(f"smoke rows {stats['rows']} != 6")
    if stats["calls"] != 6 * MRConfig().expected_calls:
        reasons.append(
            f"smoke calls {stats['calls']} != {6 * MRConfig().expected_calls}"
        )
    if stats["empty_responses"]:
        reasons.append(f"{stats['empty_responses']} empty non-selection responses")
    if stats["retrieval_empty"]:
        reasons.append(f"{stats['retrieval_empty']} retrieval calls returned no ids")
    if stats["selections"] and stats["selection_fallbacks"] == stats["selections"]:
        reasons.append(
            "all selections fell back; selector parsing/deployment is broken"
        )
    if (
        stats["candidate_pairs"]
        and stats["duplicate_pairs"] == stats["candidate_pairs"]
    ):
        reasons.append(
            "all branch candidate pairs are exact duplicates; sampling is not diverse"
        )
    return {"pass": not reasons, "reasons": reasons, "coverage": cov, "stats": stats}


def sanity_report(sanity_dir: str, baseline_dir: str, items) -> dict:
    """Compare our one-call NR outputs with the committed s1 baseline.

    Pre-registered gate: no empty answers, ours' EM on these 6 cases is no more
    than 0.34 below the baseline's, and answer lengths are not wildly different.
    Exact-response agreement is reported but not gated (vLLM versions can change
    tokenisation/decoding while the deployment remains healthy).
    """
    import score  # flat next to lib.py/mr.py on Kaggle; ../_shared locally

    expected = {f"{i['uid']}:{i['rc_id']}" for i in items}
    report = {"cases": [], "memories": {}}
    reasons = []
    for mem in ("sparse", "dense"):
        label = MEMORIES[mem][0]
        ours_path = os.path.join(sanity_dir, f"NaiveReason_{label}_k20.jsonl")
        base_candidates = [
            os.path.join(baseline_dir, f"s1_NaiveReason_{label}_k20.jsonl"),
            os.path.join(baseline_dir, "s1", f"NaiveReason_{label}_k20.jsonl"),
        ]
        base_path = next((p for p in base_candidates if os.path.exists(p)), None)
        ours = load_valid_rows(ours_path)
        if base_path is None:
            reasons.append(f"{mem}: baseline NaiveReason_{label}_k20.jsonl missing")
            continue
        base = {
            row_key(r): r for r in lib.read_jsonl(base_path) if row_key(r) in expected
        }
        missing_base = sorted(expected - set(base))
        if missing_base:
            reasons.append(f"{mem}: baseline missing {len(missing_base)} smoke keys")
        n = em_ours = em_base = exact = 0
        lens_ours = []
        lens_base = []
        empty = 0
        for k in sorted(expected & set(ours) & set(base)):
            ro, rb = ours[k], base[k]
            if not str(ro.get("response", "")).strip():
                empty += 1
            eo = bool(score.em(ro["response"], ro["ground_truth"]))
            eb = bool(score.em(rb["response"], rb["ground_truth"]))
            ex = score.normalize(ro["response"]) == score.normalize(rb["response"])
            n += 1
            em_ours += eo
            em_base += eb
            exact += ex
            lens_ours.append(len(str(ro["response"])))
            lens_base.append(len(str(rb["response"])))
            report["cases"].append(
                {
                    "key": k,
                    "memory": label,
                    "ours": ro["response"],
                    "baseline": rb["response"],
                    "em_ours": eo,
                    "em_baseline": eb,
                    "exact_normalized_match": ex,
                }
            )
        mem_report = {
            "n": n,
            "em_ours": em_ours,
            "em_baseline": em_base,
            "empty": empty,
            "exact_normalized_match": exact,
            "mean_answer_chars_ours": sum(lens_ours) / len(lens_ours)
            if lens_ours
            else 0,
            "mean_answer_chars_baseline": sum(lens_base) / len(lens_base)
            if lens_base
            else 0,
        }
        report["memories"][mem] = mem_report
        if n != len(expected):
            reasons.append(f"{mem}: sanity rows {n} != {len(expected)}")
        if empty:
            reasons.append(f"{mem}: {empty} empty sanity answers")
        if n and (em_ours / n) < (em_base / n) - 0.34:
            reasons.append(
                f"{mem}: EM drift {em_ours}/{n} vs baseline {em_base}/{n} (>0.34)"
            )
        if mem_report["mean_answer_chars_baseline"] and (
            mem_report["mean_answer_chars_ours"]
            > 5 * max(1.0, mem_report["mean_answer_chars_baseline"])
        ):
            reasons.append(f"{mem}: answer length drift")
    report["pass"] = not reasons
    report["reasons"] = reasons
    return report


# ---------------------------------------------------------------------------
# CLI


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-embeddings")
    p.add_argument("--data-dir", default=lib.DATA_DIR)
    p.add_argument("--cache-dir", default=lib.CACHE)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--users", default="all")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.set_defaults(func=cmd_prepare)

    g = sub.add_parser("run")
    g.add_argument("--stage", choices=["smoke", "full"], required=True)
    g.add_argument("--slice", required=True)
    g.add_argument("--data-dir", default=lib.DATA_DIR)
    g.add_argument("--cache-dir", default=lib.CACHE)
    g.add_argument("--output-dir", required=True)
    g.add_argument("--memories", default="sparse,dense")
    g.add_argument("--topk", type=int, default=20)
    g.add_argument("--max-step", type=int, default=5)
    g.add_argument("--branch-size", type=int, default=2)
    g.add_argument("--expected-calls", type=int, default=12)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=48)
    g.add_argument("--max-attempts", type=int, default=2)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--deadline", type=float, default=0)
    g.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--device", default=None)
    g.add_argument("--request-timeout", type=int, default=900)
    g.set_defaults(func=cmd_run)

    s = sub.add_parser("sanity")
    s.add_argument("--slice", required=True)
    s.add_argument("--data-dir", default=lib.DATA_DIR)
    s.add_argument("--cache-dir", default=lib.CACHE)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--memories", default="sparse,dense")
    s.add_argument("--topk", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--device", default=None)
    s.add_argument("--request-timeout", type=int, default=900)
    s.set_defaults(func=cmd_sanity)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
