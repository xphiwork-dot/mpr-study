#!/usr/bin/env bash
# Local driver for the ONE S6 DR Kaggle GPU job (see experiments/s6_kaggle_dr/README.md).
#
#   experiments/s6_kaggle_dr/kaggle_run.sh
#
# What it does, in order:
#   1. acquire the shared host advisory lock (~/.local/state/mpr-study/kaggle-gpu.lock) so the peer's MR
#      job never runs concurrently with ours (serialization default from the delegation);
#   2. show live GPU quota, push the private kernel (T4, session timeout 9000 s = 2.5 h);
#   3. poll `kaggle kernels status` until the job is terminal (a push is not completion);
#   4. download every output file to results/kaggle/dr/raw and print the status/DONE lines;
#   5. release the lock on exit (also on failure).
#
# There is no retry and no replacement job: a failed run is preserved and reported, and any new GPU push
# needs fresh user approval. Set KAGGLE=/path/to/kaggle to override the CLI location.
set -uo pipefail
cd "$(dirname "$0")/../.."
K="${KAGGLE:-$HOME/.local/bin/kaggle}"
REF=phinguynxun/mpr-dr-reasoning
LOCK="${MPR_KAGGLE_LOCK:-$HOME/.local/state/mpr-study/kaggle-gpu.lock}"
OUT=results/kaggle/dr/raw
LOG="${MPR_KAGGLE_RUN_LOG:-$HOME/.local/state/mpr-study/dr-kaggle-run.log}"
mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG")" "$OUT"

exec 9>"$LOCK"
echo "== $(date -u +%FT%TZ) waiting for the GPU lock ($LOCK)" | tee -a "$LOG"
if ! flock -w 14400 9; then
  echo "== $(date -u +%FT%TZ) lock not acquired within 4 h; not pushing" | tee -a "$LOG"
  exit 1
fi
echo "== $(date -u +%FT%TZ) GPU lock acquired" | tee -a "$LOG"
trap 'echo "== $(date -u +%FT%TZ) releasing the GPU lock" | tee -a "$LOG"' EXIT

echo "== live quota" | tee -a "$LOG"
"$K" quota 2>&1 | tee -a "$LOG"

echo "== pushing $REF (T4, timeout 9000 s, expected wall 1.2-1.8 h)" | tee -a "$LOG"
"$K" kernels push -p build/kaggle/dr_kernel --timeout 9000 --accelerator NvidiaTeslaT4 2>&1 | tee -a "$LOG"
push_rc=${PIPESTATUS[0]}
if [ "$push_rc" -ne 0 ]; then
  echo "== push failed rc=$push_rc; nothing launched" | tee -a "$LOG"
  exit "$push_rc"
fi

status=""
for _ in $(seq 1 260); do
  sleep 60
  status="$("$K" kernels status "$REF" 2>&1 | tail -1)"
  echo "$(date -u +%FT%TZ) $status" | tee -a "$LOG"
  case "$status" in
    *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*) break;;
  esac
done
echo "== terminal status: $status" | tee -a "$LOG"

for attempt in 1 2 3; do
  if "$K" kernels output "$REF" -p "$OUT" -o 2>&1 | tee -a "$LOG"; then
    break
  fi
  echo "== output download attempt $attempt failed; retrying in 60 s" | tee -a "$LOG"
  sleep 60
done
echo "== downloaded files" | tee -a "$LOG"
find "$OUT" -maxdepth 2 -type f -printf '%10s  %p\n' 2>/dev/null | sort -k2 | tee -a "$LOG"
echo "== status.json" | tee -a "$LOG"
cat "$OUT/status.json" 2>/dev/null | tee -a "$LOG" || echo "(no status.json downloaded)" | tee -a "$LOG"
echo "== DONE" | tee -a "$LOG"
cat "$OUT/DONE" 2>/dev/null | tee -a "$LOG" || echo "(no DONE marker downloaded)" | tee -a "$LOG"
