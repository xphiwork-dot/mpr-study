#!/usr/bin/env python3
"""Build the explicit-allowlist private Kaggle input dataset for MR.

The staged folder contains only:
  question_answer_{0..11}.json                      (released dataset memory/QA)
  s306.json                                         (frozen evaluation slice)
  s1_*.jsonl, s2_*.jsonl                            (committed NR/SR rows, sanity + local analysis)
  mr.py, lib.py, score.py, kernel.py                (runner, reused helpers, runnable entrypoint)
  input_manifest.json                               (hashes/revisions)
  dataset-metadata.json                             (private dataset metadata)

Everything is flat: the Kaggle CLI default dir-mode=skip would otherwise drop
subdirectories, so the staging layout deliberately has no directories.

No upstream source/templates, Git metadata, credentials, caches, weights or
other agents' files are copied.  Upstream is cloned at runtime by kernel.py.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(ROOT)
UPSTREAM_COMMIT = "a24953c8f342c9f3b2064e562ead49d477d49138"
MODEL_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
E5_MODEL = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()


def find_dataset_dir(explicit):
    candidates = [explicit, os.path.join(ROOT, "data", "mpr_dataset"),
                  os.path.join(PARENT, "mpr-study", "data", "mpr_dataset")]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "question_answer")):
            return os.path.abspath(c)
    raise SystemExit("released dataset not found; pass --dataset-dir (question_answer/ inside)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/mpr-mr-kaggle/dataset")
    ap.add_argument("--dataset-dir", default=os.environ.get("MPR_DATASET_DIR"))
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--dataset-id", default="phinguynxun/mpr-mr-inputs")
    ap.add_argument("--title", default="MPR MR inputs")
    args = ap.parse_args()

    data_dir = find_dataset_dir(args.dataset_dir)
    out = os.path.abspath(args.out)
    if args.clean and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    files = {}

    def add(src, rel):
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = {"bytes": os.path.getsize(dst), "sha256": sha256(dst)}

    for u in range(12):
        add(os.path.join(data_dir, "question_answer", f"question_answer_{u}.json"),
            f"question_answer_{u}.json")
    add(os.path.join(ROOT, "results", "t4", "slices", "s306.json"), "s306.json")
    for tag in ("s1", "s2"):
        for name in sorted(os.listdir(os.path.join(ROOT, "results", "t4", tag))):
            if name.endswith(".jsonl") and name != "timing.jsonl":
                add(os.path.join(ROOT, "results", "t4", tag, name), f"{tag}_{name}")
    add(os.path.join(HERE, "mr.py"), "mr.py")
    add(os.path.join(HERE, "kernel.py"), "kernel.py")
    add(os.path.join(ROOT, "experiments", "_shared", "lib.py"), "lib.py")
    add(os.path.join(ROOT, "experiments", "_shared", "score.py"), "score.py")

    dirty = bool(git("status", "--porcelain"))
    manifest = {
        "experiment": "s5_kaggle_mr",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": dirty,
        "upstream_repo": "https://github.com/nuster1128/MPR",
        "upstream_commit": UPSTREAM_COMMIT,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "model_revision": MODEL_REVISION,
        "embedding_model": E5_MODEL,
        "embedding_revision": E5_REVISION,
        "slice": "s306.json",
        "files": dict(sorted(files.items())),
    }
    with open(os.path.join(out, "input_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    meta = {
        "title": args.title,
        "id": args.dataset_id,
        "licenses": [{"name": "other"}],
        "subtitle": "MPR MR evaluation slice, released QA memory, runner code and baselines",
        "description": (
            "Private allowlisted inputs for the MR (StructuredReason) Kaggle experiment. "
            "Contains the released MPR question_answer files, the frozen s306 slice, "
            "committed NR/SR baseline rows, and the experiment-local runner code, "
            "including kernel.py so a UI-started notebook can exec the exact committed "
            "entrypoint. Upstream MPR source and prompt templates are not included; the "
            "kernel clones the pinned upstream commit at runtime."
        ),
    }
    with open(os.path.join(out, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    total = sum(v["bytes"] for v in files.values())
    print(json.dumps({"staged": out, "files": len(files), "bytes": total,
                      "git_commit": manifest["git_commit"], "git_dirty": dirty}, indent=2))
    print(f"next: kaggle datasets create -p {out}   (private by default)")


if __name__ == "__main__":
    main()
