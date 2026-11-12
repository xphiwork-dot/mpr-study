#!/usr/bin/env python
"""Kaggle script kernel for S6 -- DecompositionReason (DR) on Qwen2.5-7B-Instruct-AWQ, one T4.

One session, one GPU job (design and gates: experiments/s6_kaggle_dr/README.md):
  stage/preflight -> dependencies -> pinned upstream clone -> e5 embedding cache -> vLLM server
  -> gates (divide prompt, DR smoke on 6 questions across hop groups, NaiveReason deployment sanity
     against archived S1 rows) -> full 306x2 evaluation (resumable, deadline-aware)
  -> validation + EM/contains -> status.json/DONE + evidence files.

Inputs are read-only from /kaggle/input/mpr-dr-inputs (private dataset: the released question_answer
JSONs, the s306 slice, dr.py/lib.py/score.py/settings.json, the archived S1 rows restricted to the slice,
and manifest.json with every sha256). Everything the job must publish is written under /kaggle/working;
the DR runner flushes and fsyncs every result row, so a cut session still leaves usable partial rows.

A watchdog exits cleanly ~10 minutes before the session timeout so Kaggle commits the output.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

START = time.time()
SESSION_TIMEOUT_S = int(os.environ.get("MPR_SESSION_TIMEOUT_S", "14400"))
WATCHDOG_EXIT_BEFORE_END_S = 600
STOP_SUBMIT_BEFORE_END_S = 1500
INPUT_ROOT = os.environ.get("MPR_INPUT_ROOT", "/kaggle/input/mpr-dr-inputs")
WORK = os.environ.get("MPR_WORK", "/kaggle/working")
ROOT = os.environ.get("MPR_ROOT", "/tmp/mpr-dr")      # scratch, not published
SCRIPTS = os.path.join(ROOT, "scripts")
CACHE = os.path.join(ROOT, "cache")
QA_DIR = os.path.join(ROOT, "data", "mpr_dataset", "question_answer")
SLICE = os.path.join(ROOT, "results", "slices", "s306.json")
UPSTREAM_DIR = os.path.join(ROOT, "upstream", "MPR")
OUT = os.path.join(WORK, "results", "kaggle", "dr")
RUNS = os.path.join(OUT, "runs")
STATUS_PATH = os.path.join(WORK, "status.json")
HARD_STOP = START + SESSION_TIMEOUT_S - WATCHDOG_EXIT_BEFORE_END_S
RUN_DEADLINE = START + SESSION_TIMEOUT_S - STOP_SUBMIT_BEFORE_END_S
UPSTREAM_COMMIT = "a24953c8f342c9f3b2064e562ead49d477d49138"
UPSTREAM_URL = "https://github.com/nuster1128/MPR"
HEALTH = "http://127.0.0.1:8000/health"
MODELS_URL = "http://127.0.0.1:8000/v1/models"
KERNEL_REF = "phinguynxun/mpr-dr-reasoning"
DATASET_REF = "phinguynxun/mpr-dr-inputs"

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "0",          # one T4 like the historical Colab run; a second T4 is unused
    "T4_ROOT": ROOT,
    "MPR_QA_DIR": QA_DIR,
    "MPR_SHARED_DIR": SCRIPTS,
    "MPR_PROMPTS": os.path.join(UPSTREAM_DIR, "baselines", "prompts.py"),
    "VLLM_URL": "http://127.0.0.1:8000",
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
})

_lock = threading.Lock()
_state = {"status": "starting", "stage": "start", "started": START, "session_timeout_s": SESSION_TIMEOUT_S}
_dr = None
_settings = None


# ---------------------------------------------------------------- small utilities
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass


def log(msg):
    print(f"[{time.time() - START:8.1f}s] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def set_status(status=None, stage=None, **extra):
    with _lock:
        if status:
            _state["status"] = status
        if stage:
            _state["stage"] = stage
        _state.update(extra)
        _state["updated"] = time.time()
        _state["elapsed_s"] = round(time.time() - START, 1)
        write_json(STATUS_PATH, _state)


def run_cmd(cmd, log_path, timeout=None, check=True):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
        f.flush()
        t = time.time()
        p = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        f.write(f"\n$ rc={p.returncode} wall={time.time() - t:.1f}s\n")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {cmd} (see {log_path})")
    return p


def find_input(name):
    direct = os.path.join(INPUT_ROOT, name)
    if os.path.isfile(direct):
        return direct
    for root, _dirs, files in os.walk(INPUT_ROOT):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")


def watchdog():
    while time.time() < HARD_STOP:
        time.sleep(5)
    try:
        set_status(status="complete" if _state.get("complete_612") else "partial",
                   stage="watchdog_exit",
                   watchdog=f"hard stop at +{SESSION_TIMEOUT_S - WATCHDOG_EXIT_BEFORE_END_S}s")
        log("watchdog: session deadline close, exiting so Kaggle commits the output")
    finally:
        os._exit(0)


# ---------------------------------------------------------------- stages
def stage_inputs():
    set_status(stage="stage_inputs")
    for d in (SCRIPTS, CACHE, RUNS, os.path.dirname(SLICE), QA_DIR):
        os.makedirs(d, exist_ok=True)
    manifest = json.load(open(find_input("manifest.json"), encoding="utf-8"))
    verified = {}

    def copy_verified(name, src_name, dst):
        src = find_input(src_name)
        shutil.copyfile(src, dst)
        got = sha256_file(dst)
        ok = got == manifest["files"][src_name]["sha256"]
        verified[src_name] = {"sha256": got, "matches_manifest": ok}
        if not ok:
            raise RuntimeError(f"input hash mismatch for {src_name}")
        return dst

    for name in ("dr.py", "lib.py", "score.py", "settings.json"):
        copy_verified(name, name, os.path.join(SCRIPTS, name))
    copy_verified("s306.json", "s306.json", SLICE)
    for u in range(12):
        name = f"question_answer_{u}.json"
        copy_verified(name, name, os.path.join(QA_DIR, name))
    for name in ("baseline_nr_SparseRAG_k20.jsonl", "baseline_nr_DenseRAG_k20.jsonl"):
        copy_verified(name, name, os.path.join(ROOT, name))
    return manifest, verified


def preflight():
    set_status(stage="preflight")
    info = {"python": sys.version, "executable": sys.executable, "cpu_count": os.cpu_count(),
            "cwd": os.getcwd(), "root_scratch": ROOT, "work": WORK}
    try:
        info["nvidia_smi_L"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                              text=True).stdout.strip().splitlines()
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        info["gpus"] = [line.split(", ") for line in q.splitlines()]
    except Exception as e:  # noqa: BLE001
        info["gpu_error"] = repr(e)
    try:
        info["mem_total_kb"] = int(open("/proc/meminfo").readline().split()[1])
    except Exception:  # noqa: BLE001
        pass
    info["allocated_gpus"] = 1
    info["allocated_gpu_note"] = "CUDA_VISIBLE_DEVICES=0; a single vLLM replica; a second T4, if present, is idle"
    write_json(os.path.join(WORK, "preflight.json"), info)
    log(f"preflight gpus={info.get('gpus')} cpu={info.get('cpu_count')}")
    return info


def pip_versions(pkgs):
    import importlib.metadata as md
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


def probe_versions(modules):
    """Import each module in a SEPARATE process and report its version.

    Never import torch/vllm in the kernel process: on the historical T4 runs that opened a CUDA context
    costing about 3.5 GB and vLLM then ran out of memory on its first request
    (experiments/_shared/README.md, Pitfalls).
    """
    code = (
        "import json\n"
        "import importlib\n"
        "out = {}\n"
        "for m in %r:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        out[m] = str(getattr(mod, '__version__', None))\n"
        "    except Exception as e:\n"
        "        out[m] = 'IMPORT ERROR: ' + repr(e)\n"
        "print('VERSIONS_JSON=' + json.dumps(out))\n"
    ) % (list(modules),)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith("VERSIONS_JSON="):
            return json.loads(line.split("=", 1)[1]), p.returncode
    return {m: "IMPORT PROBE FAILED" for m in modules}, p.returncode


def stage_deps():
    set_status(stage="deps")
    import importlib.util
    logfile = os.path.join(WORK, "setup_pip.log")
    tracked = ["vllm", "torch", "transformers", "sentence-transformers", "bm25s", "numpy", "scipy",
               "accelerate", "torchao", "huggingface-hub", "peft"]
    before = pip_versions(tracked)
    installs, attempts = [], []

    def have(module):
        importlib.invalidate_caches()
        return importlib.util.find_spec(module) is not None

    if (before.get("bm25s") != "0.2.9") or not have("bm25s"):
        t = time.time()
        p = run_cmd([sys.executable, "-m", "pip", "install", "-q", "bm25s==0.2.9"], logfile,
                    timeout=900, check=False)
        attempts.append({"target": "bm25s==0.2.9", "rc": p.returncode, "secs": round(time.time() - t, 1)})
        installs.append("bm25s==0.2.9")
        if p.returncode != 0:
            raise RuntimeError(f"bm25s==0.2.9 install failed (see {logfile})")
    if not have("sentence_transformers"):
        t = time.time()
        p = run_cmd([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], logfile,
                    timeout=1800, check=False)
        attempts.append({"target": "sentence-transformers (unpinned)", "rc": p.returncode,
                         "secs": round(time.time() - t, 1)})
        installs.append("sentence-transformers (unpinned, image compatible)")
        if p.returncode != 0:
            raise RuntimeError(f"sentence-transformers install failed (see {logfile})")
    # the T4 sessions uninstalled torchao: vLLM's torch.compile path hit it (experiments/_shared/README.md)
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], logfile, timeout=300, check=False)
    if not have("vllm"):
        # unpinned first: the Kaggle image's Python/torch generation decides the compatible vLLM.
        for spec in ("vllm", "vllm==0.10.2"):
            t = time.time()
            log(f"vllm missing: pip install {spec}")
            p = run_cmd([sys.executable, "-m", "pip", "install", "-q", spec], logfile,
                        timeout=1800, check=False)
            attempts.append({"target": spec, "rc": p.returncode, "secs": round(time.time() - t, 1)})
            if p.returncode == 0:
                installs.append(spec + " (install attempt, unpinned first)")
                break
        if not have("vllm"):
            raise RuntimeError(f"could not install vLLM (see {logfile})")

    importlib.invalidate_caches()
    versions, rc = probe_versions(["vllm", "torch", "transformers", "sentence_transformers", "bm25s",
                                   "numpy", "scipy"])
    deps = {"before": before, "installs": installs, "attempts": attempts, "versions": versions,
            "probe_rc": rc, "pip_versions": pip_versions(tracked)}
    write_json(os.path.join(WORK, "deps.json"), deps)
    for mod in ("vllm", "torch", "transformers", "sentence_transformers", "bm25s"):
        if str(versions.get(mod, "")).startswith("IMPORT"):
            raise RuntimeError(f"dependency import failed: {mod}: {versions[mod]} (see {logfile})")
    log(f"deps versions={versions}")
    return deps


def stage_upstream(manifest):
    set_status(stage="upstream")
    os.makedirs(os.path.dirname(UPSTREAM_DIR), exist_ok=True)
    gitlog = os.path.join(WORK, "setup_upstream.log")
    if not os.path.isdir(os.path.join(UPSTREAM_DIR, ".git")):
        run_cmd(f"git init -q {UPSTREAM_DIR}", gitlog, timeout=120)
        run_cmd(f"git -C {UPSTREAM_DIR} remote add origin {UPSTREAM_URL}", gitlog, timeout=120)
        run_cmd(f"git -C {UPSTREAM_DIR} fetch -q --depth 1 origin {UPSTREAM_COMMIT}", gitlog, timeout=900)
        run_cmd(f"git -C {UPSTREAM_DIR} checkout -q FETCH_HEAD", gitlog, timeout=120)
    head = subprocess.run(["git", "-C", UPSTREAM_DIR, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    prompts = os.path.join(UPSTREAM_DIR, "baselines", "prompts.py")
    got = sha256_file(prompts)
    expected = manifest["upstream"]["prompts_sha256"]
    info = {"commit": head, "expected_commit": UPSTREAM_COMMIT,
            "prompts_sha256": got, "expected_prompts_sha256": expected}
    write_json(os.path.join(WORK, "upstream.json"), info)
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {head} != {UPSTREAM_COMMIT}")
    if got != expected:
        raise RuntimeError("upstream prompts.py hash mismatch (template drift): refusing to run")
    log(f"upstream {head}: prompts.py sha256 verified")
    return info


def stage_embeddings():
    """e5-base-v2 statement embeddings for all 12 users, computed before vLLM takes the GPU."""
    set_status(stage="embeddings")
    import numpy as np
    import lib
    enc = lib.get_encoder()
    rows, t0 = {}, time.time()
    for u in range(12):
        path = os.path.join(CACHE, f"emb_user{u}.npy")
        if os.path.exists(path):
            rows[u] = {"skipped": True}
            continue
        t = time.time()
        d = lib.load_user(u)
        ids = list(d["message"])
        texts = [d["message"][i] for i in ids]
        emb = enc.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        np.save(path, emb.astype("float16"))
        rows[u] = {"statements": len(ids), "shape": list(emb.shape), "secs": round(time.time() - t, 1),
                   "sha256": sha256_file(path)}
        log(f"embeddings user {u}: {len(ids)} statements in {rows[u]['secs']}s")
    info = {"users": rows, "wall_s": round(time.time() - t0, 1), "model": "intfloat/e5-base-v2",
            "prefixes": False, "normalized": True, "stored_dtype": "float16"}
    del enc
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        info["torch_cleanup_error"] = repr(e)
    try:
        info["gpu_mem_after"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    write_json(os.path.join(WORK, "embeddings.json"), info)
    log(f"embeddings done in {info['wall_s']}s; gpu_mem_after={info.get('gpu_mem_after')}")
    return info


def stage_retrieval_gate():
    """Independent checks that the cached memory indexes are wired like the historical runs."""
    set_status(stage="retrieval_gate")
    import bm25s
    import numpy as np
    import lib
    enc = lib.get_encoder()
    out = {"checks": [], "ok": True}
    try:
        for u in (0, 3, 7):
            d = lib.load_user(u)
            ids = list(d["message"])
            texts = [d["message"][i] for i in ids]
            ix = lib.UserIndex(u, d, need_dense=True, encoder=enc)
            q = d["qa"]["2"][sorted(d["qa"]["2"])[0]]["question"]
            ours = ix.sparse(q, 20)
            ret = bm25s.BM25()
            ret.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
            res, _ = ret.retrieve(bm25s.tokenize(q, stopwords="en", show_progress=False), k=20,
                                  show_progress=False)
            ref = [ids[j] for j in res[0]]
            out["checks"].append({"check": f"bm25_user{u}", "ok": ours == ref})
            out["ok"] &= ours == ref
            ours_d = ix.dense(q, 20)
            qv = enc.encode([q], normalize_embeddings=True)[0].astype("float32")
            ref_d = [ids[j] for j in np.argsort(-(ix.emb @ qv))[:20]]
            out["checks"].append({"check": f"e5_user{u}", "ok": ours_d == ref_d})
            out["ok"] &= ours_d == ref_d
            probe = enc.encode([texts[0], texts[5], texts[9]], normalize_embeddings=True).astype("float32")
            cos = [float(probe[i] @ ix.emb[j]) for i, j in enumerate((0, 5, 9))]
            ok = all(c > 0.999 for c in cos)
            out["checks"].append({"check": f"e5_cache_user{u}", "cosines": cos, "ok": ok})
            out["ok"] &= ok
    finally:
        del enc
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    write_json(os.path.join(WORK, "retrieval_gate.json"), out)
    log(f"retrieval gate ok={out['ok']}")
    if not out["ok"]:
        raise RuntimeError("retrieval gate failed: cached and independent retrieval disagree")
    return out


def _gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader",
                              "-i", "0"], capture_output=True, text=True).stdout.strip()
        used, total = [x.strip().split()[0] for x in out.split(",")]
        return int(total) - int(used)
    except Exception:  # noqa: BLE001
        return None


def _serve_attempt(label, cmd, extra_env, logfile, wait_s):
    """Start vLLM, wait for health; return (ok, info). Kill the process group on failure."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.update(extra_env)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n===== attempt {label}: {cmd}\nenv={extra_env}\n")
        f.flush()
        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    t0 = time.time()
    reason = None
    while time.time() - t0 < wait_s:
        try:
            urllib.request.urlopen(HEALTH, timeout=3)
            models = json.load(urllib.request.urlopen(MODELS_URL, timeout=10))
            served = [x.get("id") for x in models.get("data", [])]
            if _settings["model"]["served_name"] not in served:
                raise RuntimeError(f"served names {served} do not include {_settings['model']['served_name']}")
            return True, {"label": label, "cmd": cmd, "env": extra_env,
                          "health_after_s": round(time.time() - t0, 1), "models": models.get("data")}
        except Exception as e:  # noqa: BLE001
            reason = repr(e)
            if proc.poll() is not None:
                break
            time.sleep(10)
    alive = proc.poll() is None
    if alive:
        try:
            os.killpg(os.getpgid(proc.pid), 15)
            time.sleep(5)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            pass
    try:
        tail = "".join(open(logfile, errors="replace").readlines()[-30:])
    except Exception:  # noqa: BLE001
        tail = ""
    return False, {"label": label, "cmd": cmd, "env": extra_env, "died": not alive,
                   "last_error": reason, "tail": tail}


def stage_serve():
    set_status(stage="serve")
    m = _settings["model"]
    try:
        urllib.request.urlopen(HEALTH, timeout=3)
        log("vLLM already healthy")
        return
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("vllm"):
        launch = f"vllm serve {m['id']}"
    else:  # pip puts console scripts next to the interpreter, but be robust to PATH differences
        launch = f"{sys.executable} -m vllm.entrypoints.openai.api_server --model {m['id']}"
    base = (f"{launch} --revision {m['revision']} --served-model-name {m['served_name']} "
            f"--dtype {m['dtype']} --quantization {m['quantization']} --max-model-len {m['max_model_len']} "
            f"--gpu-memory-utilization {m['gpu_memory_utilization']} --max-num-seqs {m['max_num_seqs']} "
            f"--port 8000 --host 127.0.0.1")
    attempts = [
        # historical Colab deployment: same flags, engine-selected backend
        ("historical", base + f" --generation-config {m['generation_config']}", {}, 900),
        # fallback inside this same session if the image's newer engine rejects a flag or the backend:
        # Triton attention, no flashinfer sampler, no generation-config flag
        ("fallback_triton", base, {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
                                   "VLLM_USE_FLASHINFER_SAMPLER": "0"}, 600),
    ]
    logfile = os.path.join(WORK, "vllm.log")
    out = {"gpu_free_mb_before": _gpu_free_mb(), "attempts": []}
    for label, cmd, env, wait_s in attempts:
        log(f"vLLM attempt {label} (wait up to {wait_s}s)")
        ok, info = _serve_attempt(label, cmd, env, logfile, wait_s)
        out["attempts"].append(info)
        if ok:
            out.update({k: v for k, v in info.items() if k != "attempts"})
            out["used"] = label
            write_json(os.path.join(WORK, "server.json"), out)
            log(f"vLLM healthy via {label} after {info['health_after_s']}s")
            return out
        log(f"vLLM attempt {label} failed: {info.get('last_error')} died={info.get('died')}")
    write_json(os.path.join(WORK, "server.json"), out)
    raise RuntimeError("vLLM did not become healthy with either configuration; see vllm.log")


def stage_divide_gate():
    """A real DR Divide call: prompt formatting, engine, and the 5-sub-question instruction."""
    set_status(stage="divide_gate")
    import dr
    import lib
    prompts = lib.load_prompts()
    q = "What is the attachment count of the Lunar Initiative Memo after subtracting 1?"
    p = prompts.Decomposition_Divide_RAG_Prompt_Template.format(question=q, max_sub_question=5)
    text, pt, ct, secs = lib.chat(p, 200)
    subs, meta = dr.parse_division(text, 5, q, _settings["parser"]["policy"])
    out = {"prompt_sha256": dr.sha256_text(p), "response": text, "prompt_tokens": pt,
           "completion_tokens": ct, "secs": secs, "sub_questions": subs, "parser": meta,
           "ok": bool(text.strip()) and 1 <= len(subs) <= 5}
    write_json(os.path.join(WORK, "divide_gate.json"), out)
    log(f"divide gate ok={out['ok']} sub_questions={len(subs)}")
    if not out["ok"]:
        raise RuntimeError("divide gate failed: empty response or unusable division")
    return out


def smoke_keys(items):
    """Deterministic 6 questions spanning the three hop groups (2 each at hops 2, 5, 9)."""
    reps = _settings["run"]["smoke"]["hop_groups"]
    n = int(_settings["run"]["smoke"]["keys_per_hop_group"])
    groups = []
    for group, hop in reps.items():
        got = [it for it in items if int(it["hop"]) == int(hop)][:n]
        if len(got) < n:
            raise RuntimeError(f"not enough questions at hop {hop} for the smoke stage")
        groups.append({"group": group, "hop": int(hop),
                       "items": [{"uid": it["uid"], "rc_id": it["rc_id"], "hop": it["hop"]} for it in got]})
    flat = [it for g in groups for it in g["items"]]
    assert len(flat) == 6, flat
    return groups, flat


def stage_smoke_dr(items):
    set_status(stage="smoke_dr")
    import dr
    import lib
    groups, flat = smoke_keys(items)
    keys = [lib.qkey(it) for it in flat]
    t0 = time.time()
    summary = dr.run(SLICE, RUNS, _settings, memories=("sparse", "dense"),
                     workers=int(_settings["run"]["workers"]), deadline=None,
                     status_file=STATUS_PATH, only_keys=keys)
    wall = round(time.time() - t0, 1)
    checks, ok = [], True
    report = {"groups": groups, "requested": keys, "wall_s": wall, "summary": summary}
    for name in ("SparseRAG", "DenseRAG"):
        path = os.path.join(RUNS, f"DecompositionReason_{name}_k20.jsonl")
        rows = [r for r in lib.read_jsonl(path) if lib.qkey(r) in set(keys)]
        rows, _dup = dr.unique_rows(rows)
        for r in rows:
            n = len(r.get("sub_questions") or [])
            c = {"key": lib.qkey(r), "memory": name, "calls": r.get("calls"), "sub_questions": n,
                 "empty_answers": sum(1 for a in r.get("sub_answers", []) if not str(a).strip()),
                 "max_prompt_tokens": max([t["prompt_tokens"] for t in r.get("call_trace", [])] or [0]),
                 "parser_fallback": (r.get("parser") or {}).get("fallback"),
                 "n_solve_ids": [len(x or []) for x in r.get("solve_retrieved", [])],
                 "merge_ids": len(r.get("merge_retrieved") or []),
                 "response": (r.get("response") or "")[:120],
                 "calls_ok": 3 <= (r.get("calls") or 0) <= 7,
                 "subs_ok": n == (r.get("calls") or -2) - 2,
                 "prompt_fits": c_max(r) < 8192}
            c["ok"] = (c["calls_ok"] and c["subs_ok"] and c["empty_answers"] == 0
                       and all(x == 20 for x in c["n_solve_ids"]) and c["merge_ids"] == 20
                       and bool(c["response"].strip()) and c["prompt_fits"])
            ok &= c["ok"]
            checks.append(c)
        report[f"{name}_rows"] = len(rows)
        if len(rows) < 6:
            ok = False
    report.update({"ok": ok, "checks": checks})
    write_json(os.path.join(WORK, "smoke_dr.json"), report)
    log(f"DR smoke rows sparse/dense={report['SparseRAG_rows']}/{report['DenseRAG_rows']} wall={wall}s ok={ok}")
    if not ok:
        raise RuntimeError("DR smoke gate failed: see smoke_dr.json")
    return report


def c_max(row):
    return max([t["prompt_tokens"] + t["max_tokens"] for t in row.get("call_trace", [])] or [0])


def stage_sanity_naive(items):
    """Re-run NaiveReason on the smoke keys and compare with the archived S1 rows.

    Pre-registered gate (settings.json): per memory, at least 3/6 exact response matches and at least
    5/6 questions with >=18/20 identical retrieved ids. Failure means unexplained deployment drift, so
    the full DR run must not start.
    """
    set_status(stage="sanity_naive")
    import dr
    _, keys = smoke_keys(items)
    gates = _settings["run"]["baseline_sanity"]["gates"]
    runner = dr.Runner(_settings)
    rows, report = [], {"memories": {}, "gates": gates, "ok": True}
    for mem, name in (("sparse", "SparseRAG"), ("dense", "DenseRAG")):
        base = {}
        path = os.path.join(ROOT, f"baseline_nr_{name}_k20.jsonl")
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                base[f"{r['uid']}:{r['rc_id']}"] = r
        stats = {"keys": len(keys), "exact_response": 0, "retrieval_overlap_ge_18": 0,
                 "missing_archived": 0, "details": []}
        for it in keys:
            row = runner.solve_naive(mem, it)
            rows.append(row)
            k = f"{it['uid']}:{it['rc_id']}"
            b = base.get(k)
            if b is None:
                stats["missing_archived"] += 1
                stats["details"].append({"key": k, "error": "missing archived row"})
                continue
            exact = str(row["response"]).strip() == str(b["response"]).strip()
            overlap = len(set(row["first_retrieved"] or []) & set(b["first_retrieved"] or []))
            stats["exact_response"] += int(exact)
            stats["retrieval_overlap_ge_18"] += int(overlap >= 18)
            stats["details"].append({"key": k, "exact_response": exact, "retrieval_overlap": overlap,
                                     "response": row["response"], "archived_response": b["response"],
                                     "prompt_tokens": row["prompt_tokens"]})
        stats["ok"] = (stats["missing_archived"] == 0
                       and stats["exact_response"] >= gates["min_exact_response_matches_of_6"]
                       and stats["retrieval_overlap_ge_18"]
                       >= gates["min_questions_with_retrieval_overlap_18_of_20"])
        report["memories"][name] = stats
        report["ok"] &= stats["ok"]
        log(f"naive sanity {name}: exact={stats['exact_response']}/6 "
            f"overlap>=18={stats['retrieval_overlap_ge_18']}/6 missing={stats['missing_archived']} "
            f"ok={stats['ok']}")
    with open(os.path.join(OUT, "sanity_naive.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(os.path.join(WORK, "sanity_naive.json"), report)
    return report


def stage_run(items):
    set_status(stage="runtime_estimate")
    import dr
    import lib
    smoke = json.load(open(os.path.join(WORK, "smoke_dr.json"), encoding="utf-8"))
    smoke_rows = smoke["SparseRAG_rows"] + smoke["DenseRAG_rows"]
    smoke_tokens, smoke_engine = 0, 0.0
    for name in ("SparseRAG", "DenseRAG"):
        rows, _ = dr.unique_rows(lib.read_jsonl(os.path.join(RUNS, f"DecompositionReason_{name}_k20.jsonl")))
        for r in rows:
            smoke_tokens += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
            smoke_engine += r.get("secs", 0.0)
    remaining = RUN_DEADLINE - time.time()
    full_tokens = smoke_tokens / max(smoke_rows, 1) * 612
    rate = smoke_tokens / max(smoke["wall_s"], 1e-6)
    est = full_tokens / max(rate, 1e-6) * 1.3   # 1.3 safety on a 12-question, fully-concurrent smoke
    estimate = {"smoke_rows": smoke_rows, "smoke_tokens": smoke_tokens,
                "smoke_engine_secs": round(smoke_engine, 1), "smoke_wall_s": smoke["wall_s"],
                "aggregate_tokens_per_s": round(rate, 1),
                "projected_full_tokens": round(full_tokens),
                "estimated_full_secs": round(est, 1), "remaining_secs": round(remaining, 1),
                "decision": "start_full_run" if est <= remaining * 0.9 else "do_not_start_full_run"}
    write_json(os.path.join(WORK, "runtime_estimate.json"), estimate)
    log(f"runtime estimate {est:.0f}s vs remaining {remaining:.0f}s -> {estimate['decision']}")
    if estimate["decision"] != "start_full_run":
        raise RuntimeError("runtime estimate does not fit the remaining session budget")
    set_status(stage="full_run")
    summary = dr.run(SLICE, RUNS, _settings, memories=("sparse", "dense"),
                     workers=int(_settings["run"]["workers"]), deadline=RUN_DEADLINE,
                     status_file=STATUS_PATH)
    write_json(os.path.join(WORK, "run_summary.json"), summary)
    return summary


def stage_validate_and_score(items):
    set_status(stage="validate")
    import dr
    import lib
    import score as score_mod
    slice_meta = lib.load_slice(SLICE)
    flags = {f"{i['uid']}:{i['rc_id']}": i for i in slice_meta["items"]}
    expected = {f"{i['uid']}:{i['rc_id']}" for i in items}
    users = {}

    def qa_for(r):
        if r["uid"] not in users:
            users[r["uid"]] = lib.load_user(r["uid"])
        return users[r["uid"]]["qa"][str(r["k_hop"])][r["rc_id"]]

    validation = {"slice": os.path.basename(SLICE), "expected_keys": len(expected), "memories": {},
                  "ok": True}
    scores = {"slice": os.path.basename(SLICE), "n_expected": len(expected), "memories": {},
              "leak_free_rule": "slice flag leak_any == false", "metric_source": "experiments/_shared/score.py"}
    for name in ("SparseRAG", "DenseRAG"):
        path = os.path.join(RUNS, f"DecompositionReason_{name}_k20.jsonl")
        rows, dup = dr.unique_rows(lib.read_jsonl(path))
        keys = {lib.qkey(r) for r in rows}
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        failures = lib.read_jsonl(os.path.join(RUNS, f"failures_{name}.jsonl"))
        invalid = []
        for r in rows:
            try:
                qa = qa_for(r)
            except KeyError as e:
                invalid.append({"key": lib.qkey(r), "failed": [f"no released-data row: {e!r}"]})
                continue
            checks = {
                "gt_matches_dataset": r["ground_truth"] == qa["answer"],
                "calls_3_to_7": 3 <= r.get("calls", 0) <= 7,
                "subs_match_calls": len(r.get("sub_questions") or []) == r.get("calls", -1) - 2,
                "solve_ids_20": all(len(x or []) == 20 for x in r.get("solve_retrieved", [])),
                "merge_ids_20": len(r.get("merge_retrieved") or []) == 20,
                "response_nonempty": bool(str(r.get("response") or "").strip()),
            }
            bad = [k for k, v in checks.items() if not v]
            if bad:
                invalid.append({"key": lib.qkey(r), "failed": bad})
        n = len(rows)
        mem_v = {"rows": n, "duplicates_ignored": dup, "missing": len(missing), "extra": len(extra),
                 "failures": len(failures), "invalid_rows": len(invalid),
                 "missing_examples": missing[:5], "extra_examples": extra[:5],
                 "invalid_examples": invalid[:5],
                 "parser_fallback_rows": sum(1 for r in rows if (r.get("parser") or {}).get("fallback")),
                 "rows_with_dropped_blank": sum(1 for r in rows if (r.get("parser") or {}).get("dropped_blank")),
                 "mean_sub_questions": round(sum(len(r.get("sub_questions") or []) for r in rows) / n, 3) if n else None,
                 "calls": sum(r.get("calls", 0) for r in rows),
                 "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
                 "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
                 "engine_secs": round(sum(r.get("secs", 0.0) for r in rows), 1),
                 "max_prompt_plus_completion_tokens": max([c_max(r) for r in rows] or [0])}
        mem_v["ok"] = (n == len(expected) and not missing and not extra and not dup and not failures and not invalid)
        validation["memories"][name] = mem_v
        validation["ok"] &= mem_v["ok"]

        def metrics(subset):
            e = [float(score_mod.em(r["response"], r["ground_truth"])) for r in subset]
            c = [float(score_mod.contains(r["response"], r["ground_truth"])) for r in subset]
            return {"n": len(subset), "em": round(sum(e) / len(e), 4) if e else None,
                    "contains": round(sum(c) / len(c), 4) if c else None}

        leakfree = [r for r in rows if lib.qkey(r) in flags and not flags[lib.qkey(r)]["leak_any"]]
        groups = {"2-4": [r for r in rows if 2 <= r["k_hop"] <= 4],
                  "5-7": [r for r in rows if 5 <= r["k_hop"] <= 7],
                  "8-10": [r for r in rows if 8 <= r["k_hop"] <= 10]}
        scores["memories"][name] = {"all": metrics(rows), "leak_free": metrics(leakfree),
                                    "hop_groups": {g: metrics(v) for g, v in groups.items()},
                                    "per_hop": {str(h): metrics([r for r in rows if r["k_hop"] == h])
                                                for h in range(2, 11)}}
    validation["complete_612"] = all(m["rows"] == 306 for m in validation["memories"].values())
    write_json(os.path.join(WORK, "validation.json"), validation)
    write_json(os.path.join(WORK, "score_dr.json"), scores)
    log(f"validation ok={validation['ok']} complete_612={validation['complete_612']} "
        f"rows={ {k: v['rows'] for k, v in validation['memories'].items()} }")
    return validation, scores


def stage_finish(manifest, verified, validation=None, error=None):
    deps = {}
    try:
        deps = json.load(open(os.path.join(WORK, "deps.json")))
    except Exception:  # noqa: BLE001
        pass
    manifest = manifest or {}
    out_manifest = {"experiment": (_settings or {}).get("experiment"),
                    "reason": (_settings or {}).get("reason"),
                    "code_commit": manifest.get("code_commit"),
                    "input_manifest": manifest,
                    "input_files_verified": verified,
                    "model": (_settings or {}).get("model"),
                    "settings": _settings,
                    "software_versions": deps.get("versions"),
                    "installs": deps.get("installs"),
                    "kaggle": {"kernel": KERNEL_REF, "dataset": DATASET_REF,
                               "session_timeout_s": SESSION_TIMEOUT_S},
                    "started": START, "finished": time.time(),
                    "wall_s": round(time.time() - START, 1)}
    write_json(os.path.join(WORK, "manifest_out.json"), out_manifest)
    status = "failed" if error else ("complete" if (validation and validation["complete_612"]) else "partial")
    set_status(status=status, stage="finished", error=error,
               rows=({k: v["rows"] for k, v in (validation or {}).get("memories", {}).items()} or None),
               complete_612=(validation or {}).get("complete_612"),
               wall_s=round(time.time() - START, 1))
    with open(os.path.join(WORK, "DONE"), "w", encoding="utf-8") as f:
        f.write(f"status={status} stage={_state.get('stage')} wall_s={round(time.time() - START, 1)} "
                f"error={error}\n")
    log(f"finish status={status} wall={round(time.time() - START, 1)}s error={error}")


def main():
    global _dr, _settings
    os.makedirs(WORK, exist_ok=True)
    logf = open(os.path.join(WORK, "run.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, logf)
    sys.stderr = Tee(sys.stderr, logf)
    set_status(status="running", stage="start")
    threading.Thread(target=watchdog, daemon=True).start()

    manifest, verified = None, {}
    try:
        manifest, verified = stage_inputs()
        sys.path.insert(0, SCRIPTS)
        import dr
        import lib
        _dr = dr
        _settings = dr.load_settings()
        set_status(settings_sha256=dr.config_hash(_settings), code_commit=manifest.get("code_commit"))
        preflight()
        stage_deps()
        stage_upstream(manifest)
        stage_embeddings()
        stage_retrieval_gate()
        stage_serve()
        stage_divide_gate()
        items = lib.load_slice(SLICE)["items"]
        if len(items) != 306:
            raise RuntimeError(f"slice has {len(items)} questions, expected 306")
        stage_smoke_dr(items)
        sanity = stage_sanity_naive(items)
        if not sanity["ok"]:
            raise RuntimeError("NaiveReason deployment sanity check failed (see sanity_naive.json); "
                               "refusing to run the full DR evaluation")
        stage_run(items)
        validation, _scores = stage_validate_and_score(items)
        stage_finish(manifest, verified, validation=validation)
        log("evaluation complete" if validation["complete_612"] else "partial evaluation preserved")
        return 0
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        log("FATAL: " + err)
        traceback.print_exc()
        validation = None
        try:
            if _dr is not None and _settings is not None:
                import lib
                validation, _ = stage_validate_and_score(lib.load_slice(SLICE)["items"])
        except Exception as e2:  # noqa: BLE001
            log(f"validation after failure also failed: {e2!r}")
        stage_finish(manifest, verified, validation=validation, error=err)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.exit(code)
