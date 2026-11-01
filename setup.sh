#!/usr/bin/env bash
# One-shot setup: venv (uv), pinned upstream clone + our patch, dataset, question subset.
# Re-runnable. Nothing from upstream is stored in git; upstream/ and data/ are gitignored.
#
# Env knobs:
#   TORCH_INDEX   torch wheel index (default: CPU wheels, small). For a GPU use e.g.
#                 TORCH_INDEX=https://download.pytorch.org/whl/cu118
#   SKIP_DATASET  set to 1 to skip the dataset download
set -euo pipefail
cd "$(dirname "$0")"

UPSTREAM_URL=https://github.com/nuster1128/MPR
UPSTREAM_COMMIT=a24953c
DATASET_ID=1a0Ya9buk1I0hsGoaRX8bW0QJjsKuN4FR
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v git >/dev/null || { echo "git not found" >&2; exit 1; }

# Behind a corporate proxy, point Python at the system CA bundle.
if [ -z "${REQUESTS_CA_BUNDLE:-}" ] && [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

echo "== 1/5 venv (.venv, python 3.10)"
[ -d .venv ] || uv venv -p 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"
uv pip install -r requirements.txt
uv pip install torch --index-url "$TORCH_INDEX"

echo "== 2/5 upstream MPR at $UPSTREAM_COMMIT (fetched, not vendored; it has no license)"
if [ ! -d upstream/MPR/.git ]; then
  mkdir -p upstream
  git clone -q "$UPSTREAM_URL" upstream/MPR
fi
git -C upstream/MPR checkout -q "$UPSTREAM_COMMIT"

echo "== 3/5 apply patches/farm.patch"
if git -C upstream/MPR apply --check ../../patches/farm.patch 2>/dev/null; then
  git -C upstream/MPR apply ../../patches/farm.patch
  echo "applied"
elif git -C upstream/MPR apply --reverse --check ../../patches/farm.patch 2>/dev/null; then
  echo "already applied"
else
  echo "patch does not apply cleanly; reset with: git -C upstream/MPR checkout -- ." >&2
  exit 1
fi
mkdir -p upstream/MPR/results

echo "== 4/5 dataset (9.3 MB, gitignored)"
if [ "${SKIP_DATASET:-0}" = 1 ]; then
  echo "skipped"
elif [ -f data/unz/mpr_dataset/question_answer/question_answer_0.json ]; then
  echo "already present"
else
  mkdir -p data
  uvx gdown "$DATASET_ID" -O data/mpr_dataset.zip
  mkdir -p data/unz
  "$VIRTUAL_ENV/bin/python" -I -c "import zipfile,sys; zipfile.ZipFile('data/mpr_dataset.zip').extractall('data/unz')"
fi

echo "== 5/5 question subset for the optional farm run"
if [ "${SKIP_DATASET:-0}" != 1 ]; then
  "$VIRTUAL_ENV/bin/python" scripts/make_subset.py
fi

echo
echo "done. next: .venv/bin/python scripts/retrieval_recall.py ; .venv/bin/python scripts/leak_check.py"
