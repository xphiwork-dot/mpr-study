#!/usr/bin/env python3
"""Private Kaggle GPU entrypoint for the MR / StructuredReason experiment.

One job only.  Stages, in order:
  environment + private input verification -> pinned upstream clone ->
  pinned package install -> e5 embedding cache (all 12 users) -> vLLM server ->
  NR deployment sanity check on fixed questions -> MR smoke across all hop groups ->
  full 306 x {BM25, e5} evaluation -> coverage/manifest/package.

The job stops before the full run if a setup, compatibility, sanity, smoke or
throughput gate fails.  Every completed row is fsynced by mr.py, so a hard kill
still leaves partial evidence in /kaggle/working.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone

SESSION_TIMEOUT_S = 4 * 60 * 60   # approved single-job cap (CLI --timeout 14400; weekly allowance 30 h)
RESERVE_S = 30 * 60               # stop the full run in time to write status/manifest/tar
START = time.time()
START_UTC = datetime.now(timezone.utc).isoformat()
WORK = "/kaggle/working"
RUN = os.path.join(WORK, "mr_run")
OUT = os.path.join(WORK, "results", "kaggle", "mr")
SANITY = os.path.join(RUN, "sanity")
CACHE = os.path.join(RUN, "cache")
LOGS = os.path.join(RUN, "logs")
TAR = os.path.join(WORK, "mpr-mr-results.tar.gz")
VLLM_PORT = 8000
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
UPSTREAM_COMMIT = "a24953c8f342c9f3b2064e562ead49d477d49138"
VLLM_PIN_CUDA13 = "0.32.0"   # needs NVIDIA driver >= 580 (CUDA 13 wheels)
VLLM_PIN_CUDA12 = "0.11.2"   # fallback for 525 <= driver < 580 (CUDA 12 wheels)
VLLM_PIN = VLLM_PIN_CUDA13
BM25S_PIN = "0.2.9"
ST_PIN = "6.3.0"
WORKERS = 64        # matches vLLM --max-num-seqs so the server batch stays full

LOG_PATH = os.path.join(LOGS, "kernel.log")
_LOG_LOCK = threading.Lock()


def setup_dirs():
    for d in (RUN, OUT, SANITY, CACHE, LOGS, os.path.join(RUN, "upstream")):
        os.makedirs(d, exist_ok=True)


def log(*args):
    os.makedirs(LOGS, exist_ok=True)
    msg = " ".join(str(a) for a in args)
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    with _LOG_LOCK:
        print(line, flush=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_stream(cmd, *, env=None, timeout=None, check=True, tag=None):
    """Run a command, stream its output to the kernel log, return (rc, text)."""
    tag = tag or cmd[0]
    log(f"$ {tag}: {' '.join(str(c) for c in cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, bufsize=1)
    lines = []
    timer = None
    if timeout:
        def _kill():
            log(f"{tag}: TIMEOUT after {timeout}s, killing")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()
        timer = threading.Timer(timeout, _kill)
        timer.start()
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            log(f"  {line}")
        rc = p.wait()
    finally:
        if timer:
            timer.cancel()
    if check and rc != 0:
        raise RuntimeError(f"{tag} failed rc={rc}: {' '.join(cmd)}")
    return rc, "\n".join(lines)


INPUT_SLUG = "xunphi/mpr-mr-inputs"
INPUT_CANDIDATES = [
    "/kaggle/input/mpr-mr-inputs",              # legacy flat mount
    "/kaggle/input/datasets/" + INPUT_SLUG,     # current mount layout
]


def log_input_tree(root="/kaggle/input", limit=200):
    """Log the actual mount layout; Kaggle has used both layouts."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - root.count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        lines.append("  " * depth + os.path.basename(dirpath) + "/")
        for n in sorted(filenames)[:limit]:
            lines.append("  " * (depth + 1) + n)
    for line in lines[:limit]:
        log("input tree: " + line)


def find_input(wait_s=90):
    """Resolve the private dataset root, tolerating both Kaggle mount layouts.

    A CPU probe on the previous account showed the dataset mounted at
    ``/kaggle/input/datasets/<owner>/<slug>``; older runs assumed the flat
    ``/kaggle/input/<slug>`` layout, and the mount varies by run type.  Both
    candidates plus a bounded walk are kept.  The 90 s window is insurance
    against a slow mount at container start.
    """
    started = time.time()
    while True:
        for cand in [os.environ.get("MPR_INPUT_ROOT"), *INPUT_CANDIDATES]:
            if cand and os.path.isfile(os.path.join(cand, "s306.json")):
                root = os.path.abspath(cand)
                log(f"input dataset mounted at {root}")
                return root, os.path.join(root, "s306.json")
        hits = []
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if "s306.json" in filenames:
                hits.append(os.path.join(dirpath, "s306.json"))
            if dirpath.count(os.sep) - "/kaggle/input".count(os.sep) >= 4:
                dirnames[:] = []
        if hits:
            root = os.path.dirname(sorted(hits)[0])
            log(f"input dataset found by search at {root}")
            return root, os.path.join(root, "s306.json")
        if time.time() - started >= wait_s:
            log_input_tree()
            raise RuntimeError(
                "private input dataset does not contain s306.json "
                f"(searched {INPUT_CANDIDATES} and a bounded walk of /kaggle/input)"
            )
        time.sleep(5)


def verify_inputs(input_root, slice_path):
    manifest_path = os.path.join(input_root, "input_manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError("input_manifest.json missing from private dataset")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    for rel, meta in manifest.get("files", {}).items():
        path = os.path.join(input_root, rel)
        if not os.path.exists(path):
            raise RuntimeError(f"input file missing: {rel}")
        if sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"input hash mismatch: {rel}")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("input manifest upstream commit mismatch")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("input manifest model revision mismatch")
    log(f"input verified: {len(manifest.get('files', {}))} files, root={input_root}")
    return manifest


def clone_upstream():
    up = os.path.join(RUN, "upstream", "MPR")
    if os.path.exists(os.path.join(up, ".git")):
        have = subprocess.check_output(["git", "-C", up, "rev-parse", "HEAD"], text=True).strip()
        if have == UPSTREAM_COMMIT:
            log("upstream clone already at pinned commit")
            return up
        shutil.rmtree(up)
    run_stream(["git", "clone", "--no-checkout", "https://github.com/nuster1128/MPR", up],
               timeout=300, tag="git clone upstream")
    run_stream(["git", "-C", up, "checkout", UPSTREAM_COMMIT], timeout=120, tag="git checkout upstream")
    have = subprocess.check_output(["git", "-C", up, "rev-parse", "HEAD"], text=True).strip()
    if have != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream checkout mismatch {have}")
    log(f"upstream pinned at {have}")
    return up


def ensure_packages(vllm_pin=None):
    vllm_pin = vllm_pin or VLLM_PIN

    def installed_version(mod, dist):
        if importlib.util.find_spec(mod) is None:
            return None
        return package_version(dist)

    need = []
    if installed_version("vllm", "vllm") != vllm_pin:
        need.append(f"vllm=={vllm_pin}")
    if installed_version("bm25s", "bm25s") != BM25S_PIN:
        need.append(f"bm25s=={BM25S_PIN}")
    if installed_version("sentence_transformers", "sentence-transformers") != ST_PIN:
        need.append(f"sentence-transformers=={ST_PIN}")
    if not need:
        log("packages already present at pinned versions")
        return
    log(f"installing pinned packages: {need}")
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *need]
    rc, _ = run_stream(cmd, timeout=2400, check=False, tag="pip install")
    if rc != 0:
        log("pip install failed; trying uv pip install --system")
        run_stream(["uv", "pip", "install", "--system", "--no-cache-dir", *need],
                   timeout=2400, check=True, tag="uv pip install")
    importlib.invalidate_caches()
    for mod in ("vllm", "bm25s", "sentence_transformers"):
        if importlib.util.find_spec(mod) is None:
            raise RuntimeError(f"{mod} still not importable after install")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_driver_major(text):
    """Major NVIDIA driver version from ``nvidia-smi --query-gpu=...,driver_version``."""
    for row in (text or "").splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4:
            continue
        m = re.match(r"(\d+)\.", parts[-1])
        if m:
            return int(m.group(1))
    return None


def choose_vllm_pin(driver_major):
    """Pick the vLLM/Torch wheel family that can run on this driver."""
    if driver_major is None:
        return VLLM_PIN_CUDA13
    if driver_major >= 580:
        return VLLM_PIN_CUDA13
    if driver_major >= 525:
        return VLLM_PIN_CUDA12
    raise RuntimeError(
        f"NVIDIA driver {driver_major}.x is too old for the CUDA 12/13 wheels "
        f"(need >= 525); refusing to spend the session"
    )


def fallback_vllm_pin(primary):
    """Alternate wheel family to try if the primary install cannot serve.

    CUDA 13 wheels need driver >= 580; the CUDA 12 pin still runs on newer
    drivers, so it is a safe one-shot fallback there.  The reverse is not true.
    """
    if primary == VLLM_PIN_CUDA13:
        return VLLM_PIN_CUDA12
    return None


def gpu_info():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=30
        ).strip()
    except Exception as e:  # noqa: BLE001
        text = f"nvidia-smi failed: {e!r}"
    rows = [r for r in text.splitlines() if r.strip()]
    return {"nvidia_smi": text, "count": len(rows), "rows": rows}


def wait_health(proc, timeout_s=1500):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early rc={proc.returncode}; see {LOGS}/vllm.log")
        try:
            with urllib.request.urlopen(VLLM_URL + "/health", timeout=5) as r:
                if r.status == 200:
                    with urllib.request.urlopen(VLLM_URL + "/v1/models", timeout=10) as r2:
                        models = json.load(r2)
                    log(f"vLLM healthy after {time.time() - t0:.0f}s, models={models}")
                    return True
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(5)
    raise RuntimeError(f"vLLM not healthy after {timeout_s}s: {last}")


def start_vllm(env, gpu):
    log_path = os.path.join(LOGS, "vllm.log")
    venv = os.environ.copy()
    venv.update(env)
    venv["CUDA_VISIBLE_DEVICES"] = str(gpu)
    exe = shutil.which("vllm")
    if exe:
        cmd = [exe, "serve"]
    else:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    cmd += [
        MODEL,
        "--revision", MODEL_REVISION,
        "--tokenizer-revision", MODEL_REVISION,
        "--served-model-name", "qwen",
        "--dtype", "float16",
        "--quantization", "awq",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.80",
        "--max-num-seqs", "64",
        "--generation-config", "vllm",
        "--tensor-parallel-size", "1",
        "--disable-log-requests",
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
    ]
    log(f"starting vLLM on CUDA_VISIBLE_DEVICES={gpu}: {' '.join(cmd)}")
    f = open(log_path, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True,
                         env=venv, start_new_session=True)
    p._log_file = f  # type: ignore[attr-defined]
    try:
        wait_health(p)
    except Exception:
        stop_vllm(p)
        raise
    return p


def stop_vllm(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:  # noqa: BLE001
        log(f"stop_vllm: {e!r}")
    try:
        proc._log_file.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    log(f"vLLM stopped, rc={proc.poll()}")


def mr_env(input_root, up, env, *, cuda: str):
    e = dict(env)
    e.update({
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "T4_ROOT": RUN,
        "MPR_QA_DIR": input_root,
        "MPR_PROMPTS": os.path.join(up, "baselines", "prompts.py"),
        "VLLM_URL": VLLM_URL,
        "MR_SERVED_MODEL": "qwen",
        "HF_HOME": os.path.join(RUN, "hf"),
        "CUDA_VISIBLE_DEVICES": cuda,
    })
    return e


def run_mr(code_dir, args, e, timeout):
    cmd = [sys.executable, os.path.join(code_dir, "mr.py"), *args]
    return run_stream(cmd, env=e, timeout=timeout, check=False, tag="mr " + " ".join(args[:2]))


def gather_manifest(input_root, input_manifest, slice_path, up, code_dir, env_info, gates,
                    vllm_pin=None):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import mr
    import lib

    files = {}
    for dirpath, _, names in os.walk(OUT):
        for n in names:
            p = os.path.join(dirpath, n)
            rel = os.path.relpath(p, OUT)
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    embedding_manifest = os.path.join(CACHE, "embeddings_manifest.json")
    emb = json.load(open(embedding_manifest, encoding="utf-8")) if os.path.exists(embedding_manifest) else None
    prompts = mr.prompt_hashes(lib.load_prompts())
    slice_meta = json.load(open(slice_path, encoding="utf-8"))["meta"]
    return {
        "experiment": "s5_kaggle_mr StructuredReason (MR)",
        "branch": "experiments/mr-kaggle",
        "upstream": {"repo": "https://github.com/nuster1128/MPR", "commit": UPSTREAM_COMMIT,
                     "clone_head": subprocess.check_output(["git", "-C", up, "rev-parse", "HEAD"], text=True).strip()},
        "model": {"repo": MODEL, "revision": MODEL_REVISION, "served_name": "qwen",
                  "dtype": "float16", "quantization": "awq", "max_model_len": 8192,
                  "gpu_memory_utilization": 0.80, "max_num_seqs": 64,
                  "tensor_parallel_size": 1, "vllm": vllm_pin or VLLM_PIN},
        "embedding": emb,
        "slice": {"path": os.path.basename(slice_path), "sha256": sha256_file(slice_path),
                  "meta": slice_meta},
        "inputs": input_manifest,
        "prompts": prompts,
        "settings": {
            **mr.asdict(mr.MRConfig()),
            "workers": WORKERS, "topk": 20, "memories": ["SparseRAG", "DenseRAG"],
            "branch_decoding": "seeded sampling T=0.7, top_p=1.0",
            "greedy_decoding": "start, selections, final answer; T=0",
            "expected_calls_per_question": mr.MRConfig().expected_calls,
            "expected_rows": 612, "expected_rows_per_retriever": 306,
            "session_timeout_s": SESSION_TIMEOUT_S, "reserve_s": RESERVE_S,
        },
        "environment": env_info,
        "gates": gates,
        "artifacts": files,
        "started_utc": START_UTC,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - START, 1),
        "status": "complete" if gates.get("full_complete") else "partial_or_blocked",
    }


def check_cuda():
    """Verify the installed torch can actually use the GPU before spending time."""
    code = (
        "import json, torch; print(json.dumps({'torch': torch.__version__,"
        "'cuda': torch.version.cuda, 'available': torch.cuda.is_available(),"
        "'devices': torch.cuda.device_count(),"
        "'caps': [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]}))"
    )
    rc, text = run_stream([sys.executable, "-c", code], timeout=300, check=False,
                          tag="cuda check")
    info = {}
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
                break
            except Exception:  # noqa: BLE001
                continue
    log("cuda check: " + json.dumps(info))
    if rc != 0 or not info.get("available"):
        raise RuntimeError(f"torch cannot use CUDA after the pinned install: {info or text[-800:]}")
    return info


def main():
    setup_dirs()
    proc = None
    status = {"status": "running", "started_utc": START_UTC,
              "session_timeout_s": SESSION_TIMEOUT_S, "reserve_s": RESERVE_S}
    try:
        log(f"Kaggle MR kernel start; python={sys.version.split()[0]}")
        log("selected env: " + json.dumps({k: os.environ.get(k) for k in (
            "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_KERNEL_SLUG",
            "KAGGLE_KERNEL_VERSION", "KAGGLE_USERNAME")}, sort_keys=True))
        input_root, slice_path = find_input()
        input_manifest = verify_inputs(input_root, slice_path)
        code_dir = input_root
        if not os.path.exists(os.path.join(code_dir, "mr.py")):
            raise RuntimeError("dataset mr.py missing")
        gpu = gpu_info()
        log("GPU: " + gpu["nvidia_smi"])
        if gpu["count"] < 1:
            raise RuntimeError("no GPU allocated")
        vllm_gpu = "0"
        main_gpu = "1" if gpu["count"] >= 2 else ""
        driver_major = parse_driver_major(gpu.get("nvidia_smi", ""))
        vllm_pin = choose_vllm_pin(driver_major)
        log(f"allocation: vLLM GPU={vllm_gpu}, retrieval/query encoder GPU={main_gpu or 'CPU'}, "
            f"driver major={driver_major}, vllm pin={vllm_pin}")

        up = clone_upstream()
        ensure_packages(vllm_pin)
        cuda_info = check_cuda()
        # Make the pinned prompt file visible to the in-process gate helpers too.
        os.environ["MPR_PROMPTS"] = os.path.join(up, "baselines", "prompts.py")
        os.environ["MPR_QA_DIR"] = input_root
        os.environ["T4_ROOT"] = RUN
        env_info = {
            "python": sys.version,
            "packages": {n: package_version(n) for n in (
                "vllm", "torch", "transformers", "sentence-transformers", "bm25s",
                "numpy", "tokenizers", "huggingface-hub", "accelerate", "safetensors")},
            "gpu": gpu,
            "driver_major": driver_major,
            "vllm_pin": vllm_pin,
            "cuda": cuda_info,
            "allocation": {"vllm_gpu": vllm_gpu, "main_gpu": main_gpu or "cpu"},
        }
        write_json(os.path.join(RUN, "environment.json"), env_info)

        base_env = os.environ.copy()
        base_env.update({
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "HF_HOME": os.path.join(RUN, "hf"),
        })
        prep_env = mr_env(input_root, up, base_env, cuda=vllm_gpu)
        rc, _ = run_mr(code_dir, [
            "prepare-embeddings", "--data-dir", input_root,
            "--cache-dir", CACHE, "--output-dir", OUT, "--users", "all",
            "--device", "cuda" if gpu["count"] else "cpu", "--batch-size", "256",
        ], prep_env, timeout=2400)
        if rc != 0:
            raise RuntimeError("embedding preparation failed")
        log("embedding cache ready")

        main_env = mr_env(input_root, up, base_env, cuda=main_gpu)
        try:
            proc = start_vllm(base_env, vllm_gpu)
        except Exception as e:  # noqa: BLE001 - one bounded retry with the other wheel family
            alt = fallback_vllm_pin(vllm_pin)
            if alt is None:
                raise
            log(f"vLLM failed with pin {vllm_pin} ({e!r}); retrying with {alt}")
            ensure_packages(alt)
            vllm_pin = alt
            cuda_info = check_cuda()
            env_info["vllm_pin"] = vllm_pin
            env_info["cuda"] = cuda_info
            write_json(os.path.join(RUN, "environment.json"), env_info)
            proc = start_vllm(base_env, vllm_gpu)
        log("vLLM ready")

        sanity_rc, _ = run_mr(code_dir, [
            "sanity", "--slice", slice_path,
            "--data-dir", input_root,
            "--cache-dir", CACHE, "--output-dir", SANITY,
            "--topk", "20", "--device", "cuda" if main_gpu else "cpu",
        ], main_env, timeout=1800)
        sys.path.insert(0, code_dir)
        import lib
        import mr
        smoke = mr.smoke_items(lib.load_slice(slice_path)["items"])
        sanity = mr.sanity_report(SANITY, input_root, smoke)
        sanity["run_rc"] = sanity_rc
        write_json(os.path.join(OUT, "sanity_report.json"), sanity)
        log("sanity: " + json.dumps(sanity, indent=2)[:4000])
        if not sanity["pass"]:
            raise RuntimeError("NR deployment sanity gate failed: " + "; ".join(sanity["reasons"]))

        rc, _ = run_mr(code_dir, [
            "run", "--stage", "smoke", "--slice", slice_path,
            "--data-dir", input_root,
            "--cache-dir", CACHE, "--output-dir", OUT,
            "--memories", "sparse,dense", "--topk", "20", "--max-step", "5",
            "--branch-size", "2", "--expected-calls", "12", "--seed", "0",
            "--workers", "6", "--device", "cuda" if main_gpu else "cpu",
        ], main_env, timeout=1800)
        smoke_gate = mr.smoke_gate_report(OUT, smoke, input_root)
        smoke_gate["run_rc"] = rc
        write_json(os.path.join(OUT, "smoke_report.json"), smoke_gate)
        log("smoke gate: " + json.dumps(smoke_gate, indent=2)[:4000])
        if not smoke_gate["pass"]:
            raise RuntimeError("MR smoke gate failed: " + "; ".join(smoke_gate["reasons"]))

        deadline = START + SESSION_TIMEOUT_S - RESERVE_S
        log(f"full evaluation deadline epoch={deadline:.0f} "
            f"({deadline - time.time():.0f}s left)")
        full_rc, _ = run_mr(code_dir, [
            "run", "--stage", "full", "--slice", slice_path,
            "--data-dir", input_root,
            "--cache-dir", CACHE, "--output-dir", OUT,
            "--memories", "sparse,dense", "--topk", "20", "--max-step", "5",
            "--branch-size", "2", "--expected-calls", "12", "--seed", "0",
            "--workers", str(WORKERS), "--deadline", str(deadline),
            "--device", "cuda" if main_gpu else "cpu",
        ], main_env, timeout=max(600, int(deadline - time.time()) + 300))
        items = lib.load_slice(slice_path)["items"]
        coverage = mr.coverage_report(OUT, items, input_root,
                                      ["sparse", "dense"])
        full_complete = full_rc == 0 and all(v["complete"] for v in coverage["memories"].values())
        gates = {
            "sanity": sanity, "smoke": smoke_gate, "coverage": coverage,
            "full_run_rc": full_rc, "full_complete": full_complete,
        }
        write_json(os.path.join(OUT, "coverage_report.json"), coverage)
        status.update({
            "status": "complete" if full_complete else "partial_or_blocked",
            "gates": gates,
            "elapsed_s": round(time.time() - START, 1),
        })
        manifest = gather_manifest(input_root, input_manifest, slice_path, up, code_dir,
                                  env_info, gates, vllm_pin)
        write_json(os.path.join(OUT, "manifest.json"), manifest)
        write_json(os.path.join(OUT, "status.json"), status)
        log("status: " + json.dumps(status, indent=2)[:4000])
        return 0 if full_complete else 3

    except Exception as e:  # noqa: BLE001
        log("FATAL: " + repr(e))
        log(traceback.format_exc())
        status.update({"status": "blocked", "error": repr(e),
                       "traceback": traceback.format_exc(),
                       "elapsed_s": round(time.time() - START, 1)})
        write_json(os.path.join(OUT, "status.json"), status)
        return 1
    finally:
        stop_vllm(proc)
        try:
            # Best effort package even on failure.  This is what the CLI pulls.
            with tarfile.open(TAR, "w:gz") as tf:
                tf.add(OUT, arcname="results/kaggle/mr")
                if os.path.isdir(LOGS):
                    tf.add(LOGS, arcname="logs")
                emb = os.path.join(CACHE, "embeddings_manifest.json")
                if os.path.exists(emb):
                    tf.add(emb, arcname="cache/embeddings_manifest.json")
            for name in ("status.json", "manifest.json", "smoke_report.json",
                         "sanity_report.json", "coverage_report.json"):
                src = os.path.join(OUT, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(WORK, "mpr-mr-" + name))
            with open(os.path.join(WORK, "mpr-mr-package.txt"), "w", encoding="utf-8") as f:
                f.write(f"tar={TAR}\nbytes={os.path.getsize(TAR)}\n")
            log(f"packaged {TAR} ({os.path.getsize(TAR)} bytes)")
        except Exception as e:  # noqa: BLE001
            log(f"PACKAGE FAILED: {e!r}")


if __name__ == "__main__":
    sys.exit(main())
