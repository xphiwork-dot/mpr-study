#!/usr/bin/env bash
# Colab T4 fallback driver for the MR experiment.  Holds the shared GPU lock for
# the whole job: session creation, upload, run, output download and stop.
#
# The same private dataset staging used for Kaggle is unpacked on the VM and
# exposed at the path kernel.py resolves (/kaggle/input/datasets/<owner>/<slug>),
# so the exact committed entrypoint runs unchanged; outputs land in
# /kaggle/working and are downloaded here.
#
# Usage: experiments/s5_kaggle_mr/colab_mr.sh [output-dir]
#   MR_COLAB_SESSION  session name (default mpr-mr)
#   MR_COLAB_KEEP=1   leave the Colab session running on exit (debugging)
#   GPU_LOCK_WAIT_S   seconds to wait for the shared lock (default 21600)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LOCK="${KAGGLE_GPU_LOCK:-/home/xphi/.local/state/mpr-study/kaggle-gpu.lock}"
S="${MR_COLAB_SESSION:-mpr-mr}"
OUT="${1:-$ROOT/results/kaggle/mr/_colab_out}"
STAGE="${MR_COLAB_STAGE:-/tmp/mpr-mr-colab}"
WAIT_S="${GPU_LOCK_WAIT_S:-21600}"
MAX_POLLS="${MR_COLAB_MAX_POLLS:-270}"   # ~4.5 h at 60 s

DS="$STAGE/dataset"
DS_TGZ="$STAGE/mr_inputs.tgz"
RESUME_TGZ="$STAGE/mr_resume.tgz"

mkdir -p "$STAGE" "$OUT" "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -w "$WAIT_S" 9; then
  echo "[lock] timed out after ${WAIT_S}s waiting for $LOCK" >&2
  exit 1
fi
echo "[lock] acquired $(date -u +%FT%TZ)"

cleanup() {
  if [ "${MR_COLAB_KEEP:-0}" != "1" ]; then
    timeout 120 colab stop -s "$S" >/dev/null 2>&1 || true
  fi
  echo "[lock] releasing $(date -u +%FT%TZ)"
}
trap cleanup EXIT

# ---------------------------------------------------------------- staging
python3 "$HERE/pack_mr.py" --out "$DS" --clean >/dev/null
tar czf "$DS_TGZ" -C "$DS" .
rm -f "$RESUME_TGZ"
if compgen -G "$OUT/*.jsonl" >/dev/null; then
  ( cd "$OUT" && tar czf "$RESUME_TGZ" -- *.jsonl )
  echo "[stage] resume rows: $(ls "$OUT"/*.jsonl | wc -l) files"
fi

# ---------------------------------------------------------------- session
if ! colab sessions 2>&1 | grep -q "$S"; then
  n=0
  until timeout 300 colab new -s "$S" --gpu T4; do
    n=$((n + 1))
    echo "[colab] new failed (attempt $n)" >&2
    [ "$n" -ge 3 ] && { echo "[colab] no T4 session" >&2; exit 2; }
    sleep $((60 * n))
  done
fi
echo "[colab] session $S ready"

colab upload -s "$S" "$DS_TGZ" /content/mr_inputs.tgz >/dev/null
if [ -f "$RESUME_TGZ" ]; then
  colab upload -s "$S" "$RESUME_TGZ" /content/mr_resume.tgz >/dev/null
fi

# ---------------------------------------------------------------- prepare VM
prep=$(timeout 240 colab exec -s "$S" <<'PY'
import subprocess
script = r'''
set -e
mkdir -p /content/mr_inputs
rm -rf /content/mr_inputs/*
tar xzf /content/mr_inputs.tgz -C /content/mr_inputs
# kernel.py checks MPR_INPUT_ROOT first; the symlink is only belt-and-braces in
# case Colab refuses to create /kaggle.
mkdir -p /kaggle/input/datasets/xunphi /kaggle/working/results/kaggle/mr 2>/dev/null || true
ln -sfn /content/mr_inputs /kaggle/input/datasets/xunphi/mpr-mr-inputs 2>/dev/null || true
mkdir -p /kaggle/working/results/kaggle/mr
if [ -f /content/mr_resume.tgz ]; then tar xzf /content/mr_resume.tgz -C /kaggle/working/results/kaggle/mr; fi
rm -f /content/mr_kernel.exit
nvidia-smi -L
'''
r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=200)
print((r.stdout or '') + (r.stderr or ''))
raise SystemExit(r.returncode)
PY
) || { echo "[colab] VM preparation failed: $prep" >&2; exit 3; }
echo "[colab] $(echo "$prep" | grep -m1 'GPU' || echo 'GPU check output missing')"
if ! echo "$prep" | grep -q 'GPU [0-9]'; then
  echo "[colab] no GPU on the session; refusing to start the experiment" >&2
  exit 4
fi

# ---------------------------------------------------------------- run
timeout 180 colab exec -s "$S" <<'PY' >/dev/null
import subprocess
cmd = ("cd /kaggle/working && "
       "MPR_INPUT_ROOT=/content/mr_inputs "
       "python -u /kaggle/input/datasets/xunphi/mpr-mr-inputs/kernel.py "
       "> /content/mr_kernel.log 2>&1; echo $? > /content/mr_kernel.exit")
subprocess.Popen(['bash', '-c', cmd], start_new_session=True)
print('launched')
PY
echo "[run] kernel launched $(date -u +%FT%TZ)"

poll() {
  timeout 150 colab exec -s "$S" <<'PY'
import glob, os
e, log = '/content/mr_kernel.exit', '/content/mr_kernel.log'
print('EXIT=' + open(e).read().strip() if os.path.exists(e) else 'EXIT=running')
counts = {}
for p in glob.glob('/kaggle/working/results/kaggle/mr/*.jsonl'):
    name = os.path.basename(p).rsplit('.jsonl', 1)[0]
    try:
        counts[name] = sum(1 for line in open(p, encoding='utf-8') if line.strip())
    except OSError:
        counts[name] = -1
sparse = counts.get('StructuredReason_SparseRAG_k20_s5', 0)
dense = counts.get('StructuredReason_DenseRAG_k20_s5', 0)
print(f'ROWS sparse={sparse}/306 dense={dense}/306')
if os.path.exists(log):
    print(''.join(open(log).readlines()[-3:])[-600:])
else:
    print('nolog')
PY
}

fails=0
for i in $(seq 1 "$MAX_POLLS"); do
  sleep 60
  if ! out=$(poll); then
    fails=$((fails + 1))
    echo "[poll] failed ($fails)" >&2
    [ "$fails" -ge 5 ] && { echo "[poll] session lost" >&2; break; }
    continue
  fi
  fails=0
  echo "$out" | tail -4
  echo "$out" | grep -q 'EXIT=running' || break
done

# ---------------------------------------------------------------- download
pull() {
  timeout 240 colab exec -s "$S" <<'PY' >/dev/null 2>&1 || true
import subprocess
script = r'''
set -e
rm -rf /content/mr_out && mkdir -p /content/mr_out
if [ -d /kaggle/working/results/kaggle/mr ]; then
  cp -a /kaggle/working/results/kaggle/mr/. /content/mr_out/
fi
# top-level copies written by kernel.py (status/manifest/gates/package note)
cp /kaggle/working/mpr-mr-* /content/mr_out/ 2>/dev/null || true
# kernel log (gates, vLLM setup, timings), embeddings manifest, packaged tarball
if [ -d /kaggle/working/mr_run/logs ]; then
  mkdir -p /content/mr_out/logs && cp -a /kaggle/working/mr_run/logs/. /content/mr_out/logs/
fi
cp /kaggle/working/mr_run/cache/embeddings_manifest.json /content/mr_out/ 2>/dev/null || true
cp /kaggle/working/mpr-mr-results.tar.gz /content/mr_out/ 2>/dev/null || true
tar czf /content/mr_results.tgz -C /content/mr_out .
'''
r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=180)
print((r.stdout or '') + (r.stderr or ''))
raise SystemExit(r.returncode)
PY
  colab download -s "$S" /content/mr_results.tgz "$OUT/mr_results.tgz" >/dev/null 2>&1 || return 1
  tar xzf "$OUT/mr_results.tgz" -C "$OUT"
}
pull || echo "[pull] no results downloaded" >&2
echo "[pull] jsonl files: $(find "$OUT" -maxdepth 1 -name '*.jsonl' | wc -l), status: $(test -f "$OUT/status.json" && echo yes || echo no)"
echo "[done] $(date -u +%FT%TZ) outputs in $OUT"
