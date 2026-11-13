"""Build and (optionally) upload the private Kaggle resources for S6 (DR).

    python3 experiments/s6_kaggle_dr/package.py build          # stage dirs + manifest, no upload
    python3 experiments/s6_kaggle_dr/package.py dataset        # create/version phinguynxun/mpr-dr-inputs
    python3 experiments/s6_kaggle_dr/package.py kernel         # push phinguynxun/mpr-dr-reasoning (4h T4)
    python3 experiments/s6_kaggle_dr/package.py status|output  # kernel status / download outputs

Only an explicit allowlist is uploaded (no .git, no credentials, no upstream sources, no other agents'
state). `*.zip`, logs and build output stay out. The kernel metadata points at one private dataset and
enables internet (needed for the pinned upstream clone and the Qwen/e5 model downloads).

The dataset holds:
  question_answer_<u>.json (12)  released MPR data, hashed
  s306.json                      the frozen 306-question slice (34 per hop 2-10, all 12 users)
  dr.py lib.py score.py          experiment code + reused shared helpers (read-only copies)
  settings.json                  frozen settings (also loaded by the kernel)
  baseline_nr_*.jsonl            archived S1 NaiveReason rows restricted to the slice (sanity gate only)
  manifest.json                  sha256 of every file above + code commit, upstream/model revisions

The kernel dir holds only kernel.py + kernel-metadata.json; `kaggle kernels push` uploads the code file,
and the helpers come from the dataset.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPO, "build", "kaggle")
DATASET_DIR = os.path.join(BUILD, "dr_dataset")
KERNEL_DIR = os.path.join(BUILD, "dr_kernel")
DATASET_REF = "phinguynxun/mpr-dr-inputs"
KERNEL_REF = "phinguynxun/mpr-dr-reasoning"
UPSTREAM_COMMIT = "a24953c8f342c9f3b2064e562ead49d477d49138"
KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/.local/bin/kaggle"))
# Shared account: 6 GPU-hours per refresh and two serialized workers, so this job gets 2.5 h (MR takes 3 h).
SESSION_TIMEOUT_S = int(os.environ.get("MPR_SESSION_TIMEOUT_S", "9000"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, check=True):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed rc={p.returncode}: {cmd}")
    return p


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def require_clean_tree():
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit("tracked files are dirty; commit before packaging so the manifest matches the code:\n" + dirty)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build():
    require_clean_tree()
    commit = git("rev-parse", "HEAD")
    subject = git("log", "-1", "--pretty=%s")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    upstream = os.path.join(REPO, "upstream", "MPR")
    head = subprocess.run(["git", "-C", upstream, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream clone is at {head}, expected {UPSTREAM_COMMIT}")
    prompts = os.path.join(upstream, "baselines", "prompts.py")
    prompts_sha = sha256_file(prompts)
    settings = json.load(open(os.path.join(HERE, "settings.json"), encoding="utf-8"))

    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(DATASET_DIR)
    os.makedirs(KERNEL_DIR)

    files = {}

    def add(src, name):
        dst = os.path.join(DATASET_DIR, name)
        shutil.copyfile(src, dst)
        files[name] = {"sha256": sha256_file(dst), "bytes": os.path.getsize(dst), "source": os.path.relpath(src, REPO)}
        return dst

    qa_src = os.path.join(REPO, "data", "mpr_dataset", "question_answer")
    if not os.path.isdir(qa_src):
        raise SystemExit(f"missing released dataset at {qa_src} (gitignored; see experiments/_shared/README.md)")
    for u in range(12):
        src = os.path.join(qa_src, f"question_answer_{u}.json")
        if not os.path.isfile(src):
            raise SystemExit(f"missing {src}")
        add(src, f"question_answer_{u}.json")
    slice_src = os.path.join(REPO, "results", "t4", "slices", "s306.json")
    add(slice_src, "s306.json")
    slice_keys = {(i["uid"], i["rc_id"]) for i in json.load(open(slice_src, encoding="utf-8"))["items"]}
    assert len(slice_keys) == 306, len(slice_keys)
    for name, src in (("dr.py", os.path.join(HERE, "dr.py")),
                      ("lib.py", os.path.join(REPO, "experiments", "_shared", "lib.py")),
                      ("score.py", os.path.join(REPO, "experiments", "_shared", "score.py")),
                      ("settings.json", os.path.join(HERE, "settings.json"))):
        add(src, name)
    for mem in ("SparseRAG", "DenseRAG"):
        src = os.path.join(REPO, "results", "t4", "s1", f"NaiveReason_{mem}_k20.jsonl")
        rows = [r for r in read_jsonl(src) if (r["uid"], r["rc_id"]) in slice_keys]
        if len(rows) != 306:
            raise SystemExit(f"{src}: {len(rows)} rows on the slice, expected 306")
        name = f"baseline_nr_{mem}_k20.jsonl"
        dst = os.path.join(DATASET_DIR, name)
        with open(dst, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[name] = {"sha256": sha256_file(dst), "bytes": os.path.getsize(dst),
                       "source": os.path.relpath(src, REPO), "restricted_to_slice_keys": len(slice_keys)}

    manifest = {
        "experiment": "s6_kaggle_dr",
        "reason": settings["reason"],
        "code_commit": commit,
        "code_commit_subject": subject,
        "code_branch": branch,
        "upstream": {"repo": settings["upstream"]["repo"], "commit": UPSTREAM_COMMIT,
                     "prompts_sha256": prompts_sha},
        "model": settings["model"],
        "kaggle": {"kernel": KERNEL_REF, "dataset": DATASET_REF},
        "files": files,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_host": os.uname().nodename,
        "python": sys.version,
        "privacy": "all Kaggle resources are private; no credentials, .git, upstream sources or other agents' state",
    }
    with open(os.path.join(DATASET_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")

    meta = {
        "id": DATASET_REF,
        "title": "MPR DR private inputs",
        "subtitle": "MPR released data, s306 slice, DR code and manifest",
        "description": (
            "Private inputs for the DR (DecompositionReason) Kaggle session of the MPR study. "
            "Contains the released per-user question_answer JSON files, the frozen 306-question slice, "
            "the experiment code and a sha256 manifest. Not for redistribution."),
        "licenses": [{"name": "other"}],
        "keywords": [],
    }
    with open(os.path.join(DATASET_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")

    shutil.copyfile(os.path.join(HERE, "kernel.py"), os.path.join(KERNEL_DIR, "kernel.py"))
    kernel_meta = {
        "id": KERNEL_REF,
        "title": "MPR DR reasoning",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET_REF],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(KERNEL_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=1)
        f.write("\n")
    print(f"built {DATASET_DIR} ({sum(v['bytes'] for v in files.values()) / 1e6:.1f} MB) and {KERNEL_DIR}")
    print(f"commit {commit[:12]} {subject}")
    print(f"upstream prompts sha256 {prompts_sha}")
    return manifest


def dataset(action):
    if action == "create":
        run([KAGGLE, "datasets", "create", "-p", DATASET_DIR, "--keep-tabular"])
    elif action == "version":
        notes = f"DR inputs for {git('rev-parse', '--short', 'HEAD')}"
        run([KAGGLE, "datasets", "version", "-p", DATASET_DIR, "-m", notes, "--keep-tabular"])
    else:
        raise SystemExit(action)


def kernel():
    run([KAGGLE, "kernels", "push", "-p", KERNEL_DIR, "--timeout", str(SESSION_TIMEOUT_S),
         "--accelerator", "NvidiaTeslaT4"])


def status():
    run([KAGGLE, "kernels", "status", KERNEL_REF], check=False)
    print()
    run([KAGGLE, "quota"], check=False)


def output():
    out = os.path.join(REPO, "results", "kaggle", "dr", "raw")
    os.makedirs(out, exist_ok=True)
    run([KAGGLE, "kernels", "output", KERNEL_REF, "-p", out, "-o"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "dataset", "kernel", "status", "output", "dataset-version"])
    args = ap.parse_args()
    if args.action == "build":
        build()
    elif args.action == "dataset":
        dataset("create")
    elif args.action == "dataset-version":
        dataset("version")
    elif args.action == "kernel":
        kernel()
    elif args.action == "status":
        status()
    elif args.action == "output":
        output()


if __name__ == "__main__":
    main()
