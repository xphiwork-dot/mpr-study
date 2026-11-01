"""Write the four Colab notebooks (sessions 1, 2, 3a, 3b) from one template, so they stay consistent.

Each notebook runs ON the Colab VM (no `colab` CLI needed): clone this repo, install vLLM, fetch the upstream clone
and the dataset, start `vllm serve` for Qwen2.5-7B-Instruct-AWQ, run that session's jobs, summarise, and save the raw
results to /content (and optionally Google Drive). The notebooks are written without outputs and contain no credentials:
the repo is private, so the clone cell reads an optional Colab secret named GITHUB_TOKEN at run time.

Usage: python experiments/_shared/make_notebooks.py        (rewrites the four .ipynb files; stdlib only)
Provenance: written by us, no upstream reference. The commands inside the notebooks mirror experiments/_shared/t4.sh
(pinned model revision, vLLM flags, flat scripts/t4 layout on the VM) and the sessionN.sh drivers.
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_REV = "b25037543e9394b818fdfca67ab2a00ecc7dd641"  # Qwen/Qwen2.5-7B-Instruct-AWQ, same pin as t4.sh
UPSTREAM_COMMIT = "a24953c"
DATASET_ID = "1a0Ya9buk1I0hsGoaRX8bW0QJjsKuN4FR"  # same id as setup.sh

# jobs: (job name, script in scripts/t4/, arguments, needs the vLLM server)
SESSIONS = {
    "s1": {
        "dir": "s1_t4_one_call_grid", "file": "s1_t4_one_call_grid.ipynb",
        "title": "Session 1: throughput, 12-user retrieval error bars, one-call grid on 900 questions",
        "readme": "experiments/s1_t4_one_call_grid/README.md", "status": None,
        "jobs": [
            ("thr", "throughput.py", "--slice results/slices/s900.json --seconds 90"),
            ("grid1", "grid.py", "--slice results/slices/s900.json --reason naive --memories ignoramus,oracle,sparse,dense --topk 20 --tag s1"),
        ],
        "summaries": [
            "summarize.py throughput", "summarize.py retrieval",
            "summarize.py grid --tag s1 --slice results/t4/slices/s900.json",
        ],
        "tags": ["s1", "retrieval", "throughput"], "main_summary": "summary_s1.md",
    },
    "s2": {
        "dir": "s2_t4_chain_of_thought", "file": "s2_t4_chain_of_thought.ipynb",
        "title": "Session 2: five-step chain-of-thought (SequentialReason) grid on 306 questions",
        "readme": "experiments/s2_t4_chain_of_thought/README.md", "status": None,
        "jobs": [
            (f"grid2_{m}", "grid.py", f"--slice results/slices/s306.json --reason seq --max-step 5 --memories {m} --topk 20 --tag s2 --workers 48")
            for m in ("ignoramus", "oracle", "sparse", "dense")],
        "summaries": [
            "summarize.py grid --tag s2 --slice results/t4/slices/s306.json",
            "summarize.py compare --tag s2 --nr-tag s1 --slice results/t4/slices/s306.json",
        ],
        "tags": ["s2"], "main_summary": "summary_s2_vs_s1.md",
    },
    "s3a": {
        "dir": "s3a_t4_topk_sweep", "file": "s3a_t4_topk_sweep.ipynb",
        "title": "Session 3a: top-k sweep (k = 10 / 30 / 50) with one-call NaiveReason on 900 questions",
        "readme": "experiments/s3a_t4_topk_sweep/README.md", "status": None,
        "jobs": [
            (f"grid3_k{k}", "grid.py", f"--slice results/slices/s900.json --reason naive --memories sparse,dense --topk {k} --tag s3_topk")
            for k in (10, 30, 50)],
        "summaries": [
            "summarize.py grid --tag s3_topk --slice results/t4/slices/s900.json",
            "summarize.py topk",
        ],
        "tags": ["s3_topk"], "main_summary": "summary_s3_topk_sweep.md",
    },
    "s3b": {
        "dir": "s3b_t4_step_sweep_not_run", "file": "s3b_t4_step_sweep.ipynb",
        "title": "Session 3b: reasoning-step sweep (2 / 3 / 8 steps), NOT RUN so far",
        "readme": "experiments/s3b_t4_step_sweep_not_run/README.md",
        "status": "This session has never been run: no free T4 could be obtained on 2026-11-01 (16 `colab new --gpu T4` attempts returned 503). "
                  "The notebook is a ready-to-run version of `session3b.sh`, committed unexecuted.",
        "jobs": [
            (f"grid3_s{s}", "grid.py", f"--slice results/slices/s306.json --reason seq --max-step {s} --memories sparse,dense --topk 20 --tag s3_steps --workers 48")
            for s in (2, 3, 8)],
        "summaries": [
            "summarize.py grid --tag s3_steps --slice results/t4/slices/s306.json",
            "summarize.py steps",
        ],
        "tags": ["s3_steps"], "main_summary": "summary_s3_steps_sweep.md", "limit": True,
    },
}


def lines(text):
    return text.strip("\n").splitlines(True)


def md(cid, text):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines(text)}


def code(cid, text):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": lines(text)}


CLONE = '''
import base64, os, shutil, subprocess

REPO = "/content/mpr-study"
if os.path.isdir(os.path.join(REPO, "experiments")):
    print("repo already at", REPO)          # e.g. uploaded by hand: nothing to clone
else:
    cmd = ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO]
    try:
        from google.colab import userdata  # optional Colab secret GITHUB_TOKEN (read-only token), never stored in this notebook
        token = userdata.get("GITHUB_TOKEN")
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd[1:1] = ["-c", f"http.extraheader=Authorization: Basic {basic}"]
    except Exception:
        pass                                 # public repo or no secret: plain clone
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-300:], (r.stderr[-300:] if r.returncode else ""))
    assert r.returncode == 0, "clone failed: set the GITHUB_TOKEN secret (private repo) or upload the repo folder to /content/mpr-study"
print(sorted(os.listdir(REPO)))
'''

INSTALL = '''
# vLLM brings its own torch. Do NOT import vllm or torch in this kernel: that opens a CUDA context (~3.5 GB) and
# vLLM then runs out of memory on its first request. Check for the package with find_spec only.
!pip -q install uv gdown
!uv pip install --system -q vllm bm25s==0.2.9 sentence-transformers
!uv pip uninstall --system -q torchao || true      # Colab's torchao is refused by peft; not needed here
import importlib.util
print("vllm installed:", importlib.util.find_spec("vllm") is not None)
'''

STAGE = '''
import glob, zipfile

T4 = "/content/t4"                     # same layout as the colab-CLI driver (experiments/_shared/t4.sh)
for d in ("scripts/t4", "logs", "results"):
    os.makedirs(f"{T4}/{d}", exist_ok=True)
# flat scripts/t4/<name>.py on the VM, as t4.sh does
for f in (glob.glob(f"{REPO}/experiments/_shared/*.py") + glob.glob(f"{REPO}/experiments/s*/*.py")
          + [f"{REPO}/experiments/00_probes_retrieval_leak/leak_check.py"]):
    shutil.copy(f, f"{T4}/scripts/t4/")
shutil.copytree(f"{REPO}/results/t4/slices", f"{T4}/results/slices", dirs_exist_ok=True)

# upstream code: only its prompt templates are read (baselines/prompts.py); never copied into the repo
if not os.path.isdir(f"{T4}/upstream/MPR/.git"):
    os.makedirs(f"{T4}/upstream", exist_ok=True)
    !git clone -q https://github.com/nuster1128/MPR {T4}/upstream/MPR
!git -C {T4}/upstream/MPR checkout -q @@COMMIT@@

# dataset (about 9 MB zip, the public Google Drive file named in the authors' README)
if not os.path.isdir(f"{T4}/data/mpr_dataset/question_answer"):
    os.makedirs(f"{T4}/data", exist_ok=True)
    !gdown @@DATASET@@ -O /content/t4_dataset.zip
    zipfile.ZipFile("/content/t4_dataset.zip").extractall(f"{T4}/data")
print(sorted(os.listdir(f"{T4}/data")), sorted(os.listdir(f"{T4}/results")))
'''

RESUME = '''
# Optional: Google Drive keeps the raw results between Colab sessions, so a fresh VM resumes where the last one stopped.
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    prev = f"{DRIVE_DIR}/mpr_{SESSION}_results.tgz"
    if os.path.exists(prev):
        !tar xzf {prev} -C {T4}
        print("restored", prev)
'''

HELPERS = '''
import time, urllib.request

MODEL_REV = "@@REV@@"   # Qwen/Qwen2.5-7B-Instruct-AWQ, pinned as in t4.sh


def serve():
    """Start vllm serve in the background and wait until /health answers (cached weights: about 2 minutes)."""
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        print("server already healthy"); return
    except Exception:
        pass
    cmd = ("vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --revision " + MODEL_REV + " --served-model-name qwen --dtype float16 "
           "--quantization awq --max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 64 "
           "--generation-config vllm --port 8000 --host 127.0.0.1")
    subprocess.Popen(cmd + f" > {T4}/logs/vllm.log 2>&1", shell=True, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
            print(f"vLLM healthy after {time.time() - t0:.0f} s"); return
        except Exception:
            time.sleep(15)
    raise RuntimeError("vLLM did not become healthy in 15 minutes: see /content/t4/logs/vllm.log")


def unserve():
    subprocess.run("pkill -f 'vllm serve'; pkill -f EngineCore; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True)


def save():
    """Pack raw results and logs into /content (and Drive when SAVE_TO_DRIVE); called after every job."""
    out = f"/content/mpr_{SESSION}_results.tgz"
    subprocess.run(f"cd {T4} && tar czf {out} results logs/*.log logs/*.exit 2>/dev/null", shell=True)
    if SAVE_TO_DRIVE:
        shutil.copy(out, DRIVE_DIR)
    print("saved", out)


def job(name, script, args=""):
    """Run scripts/t4/<script> on this VM in the background (survives a closed tab), poll its log, then save results."""
    exitf = f"{T4}/logs/{name}.exit"
    if os.path.exists(exitf):
        os.remove(exitf)
    cmd = (f"cd {T4} && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false T4_ROOT={T4} "
           f"python scripts/t4/{script} {args} > logs/{name}.log 2>&1; echo $? > logs/{name}.exit")
    subprocess.Popen(cmd, shell=True, start_new_session=True)
    print("launched", name, script, args)
    while not os.path.exists(exitf):
        time.sleep(60)
        print("  ", "".join(open(f"{T4}/logs/{name}.log").readlines()[-1:])[-200:].strip())
    rc = int(open(exitf).read().strip())
    print(f"job {name} finished rc={rc}")
    save()
    assert rc == 0, f"job {name} failed: see {T4}/logs/{name}.log"
'''


def build(key, s):
    status = s["status"] or ("Committed unexecuted as a notebook (`not executed as a notebook`): the same steps were run for this session "
                             "through the `colab` CLI driver; the folder README says what was run and where the results are.")
    cells = [
        md("intro", f"""
# {s['title']}

Explicit vs Implicit Memory (MPR, KDD '26) study repo. Runs on a **free-tier Colab T4** (Runtime > Change runtime type > T4 GPU).

{status}

What it does: clone the repo, install vLLM, fetch the upstream code (nuster1128/MPR at commit `{UPSTREAM_COMMIT}`, only to read its
prompt templates) and the dataset, serve `Qwen/Qwen2.5-7B-Instruct-AWQ` with vLLM, run this session's jobs, write the markdown
summaries, and save the raw results to `/content` (and to Google Drive if you set `SAVE_TO_DRIVE`). All jobs are resumable: a
finished question is skipped when you rerun a cell after the session was cut.

Folder README (experiment, provenance, expected numbers): `{s['readme']}`.
"""),
        code("config", f"""
# ---- settings (no credentials here) ----
REPO_URL = "https://github.com/xphiwork-dot/mpr-study.git"   # private repo: see the clone cell
REPO_REF = "main"
SAVE_TO_DRIVE = False        # True: mount Google Drive, keep a copy of the raw results there and resume from it
DRIVE_DIR = "/content/drive/MyDrive/mpr_study_results"
SESSION = "{key}"
""" + ("LIMIT = 0                   # >0: only the first N questions per run (shorter session, as LIMIT=150 in session3b.sh)\n" if s.get("limit") else "")
             + "\n!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n"),
        code("clone", CLONE),
        code("install", INSTALL),
        code("stage", STAGE.replace("@@COMMIT@@", UPSTREAM_COMMIT).replace("@@DATASET@@", DATASET_ID)),
        code("resume", RESUME),
        code("helpers", HELPERS.replace("@@REV@@", MODEL_REV)),
        md("retrieve-md", """
## 1. Retrieval pass (no LLM)

`retrieve.py` embeds every user's statements with e5-base-v2 (the cache used by the dense-memory runs) and writes per-question
recall for all 12 users. It needs the GPU **without** the vLLM server, so it runs first; a user that is already done is skipped.
"""),
        code("retrieve", 'job("retrieve", "retrieve.py")'),
        md("serve-md", "## 2. Start vLLM\n\nQwen2.5-7B-Instruct AWQ (4-bit), fp16 compute, `--max-model-len 8192`; the runner uses greedy decoding."),
        code("serve", "serve()"),
        md("run-md", "## 3. Run this session's jobs\n\nEach cell is one call of the same script and arguments as in the session shell driver. Rerun a cell after a cut session: finished questions are skipped."),
    ]
    for i, (name, script, args) in enumerate(s["jobs"], 1):
        if s.get("limit"):
            cells.append(code(f"job{i}", f'LIMIT_ARG = f"--limit {{LIMIT}}" if LIMIT else ""\njob("{name}", "{script}", "{args} " + LIMIT_ARG)'))
        else:
            cells.append(code(f"job{i}", f'job("{name}", "{script}", "{args}")'))
    cells.append(code("stop", "unserve()   # free the GPU"))
    copy_tags = "\n".join(
        f'if os.path.exists(f"{{T4}}/results/{t}"):\n    shutil.copytree(f"{{T4}}/results/{t}", f"{{REPO}}/results/t4/{t}", dirs_exist_ok=True)' for t in s["tags"])
    sums = "\n".join(f"!cd {{REPO}} && python3 experiments/_shared/{x}" for x in s["summaries"])
    cells += [
        md("sum-md", """
## 4. Summaries

Copy the raw results into the cloned repo's `results/t4/` and run the summariser there. It reads `results/t4/<tag>/*.jsonl`; the clone
already holds the earlier sessions' committed results, which the comparisons need. Output: `results/t4/summary_*.md` in the clone.
"""),
        code("summarise", f"""
os.makedirs(f"{{REPO}}/results/t4", exist_ok=True)
{copy_tags}
{sums}
print([f for f in sorted(os.listdir(f"{{REPO}}/results/t4")) if f.startswith("summary_")])
"""),
        md("save-md", """
## 5. Save

The raw results and logs are in `/content/mpr_<session>_results.tgz` (and on Drive if `SAVE_TO_DRIVE`). To commit them to the repo,
download the archive (uncomment the last line) and unpack it under `results/t4/` locally. Release the GPU when done
(Runtime > Disconnect and delete runtime).
"""),
        code("save", f"""
save()
summary = f"{{REPO}}/results/t4/{s['main_summary']}"
if os.path.exists(summary):
    shutil.copy(summary, "/content/")
# from google.colab import files; files.download(f"/content/mpr_{{SESSION}}_results.tgz")   # uncomment to download
"""),
    ]
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main():
    for key, s in SESSIONS.items():
        path = os.path.join(ROOT, s["dir"], s["file"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(build(key, s), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
