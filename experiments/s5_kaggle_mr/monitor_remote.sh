#!/usr/bin/env bash
# Monitor an already-started Kaggle run (for example a version started from the
# Kaggle UI with GPU T4 x2) while holding the shared GPU lock through terminal
# status and output download.  Does not push anything.
#
# Usage: experiments/s5_kaggle_mr/monitor_remote.sh [kernel-slug] [output-dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
LOCK="${KAGGLE_GPU_LOCK:-/home/xphi/.local/state/mpr-study/kaggle-gpu.lock}"
SLUG="${1:-phinguynxun/mpr-mr-reasoning}"
OUT="${2:-$ROOT/results/kaggle/mr/_kaggle_out}"
WAIT_S="${GPU_LOCK_WAIT_S:-21600}"

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -w "$WAIT_S" 9; then
  echo "[lock] timed out after ${WAIT_S}s waiting for $LOCK" >&2
  exit 1
fi
echo "[lock] acquired $(date -u +%FT%TZ)"
trap 'echo "[lock] releasing $(date -u +%FT%TZ)"' EXIT

while :; do
  status="$("$KAGGLE" kernels status "$SLUG" 2>&1 | tail -n 1)"
  echo "[status] $(date -u +%FT%TZ) $status"
  case "$status" in
    *KernelWorkerStatus.COMPLETE*|*KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL*|\
    *KernelWorkerStatus.FAILED*) break ;;
  esac
  sleep 60
done

mkdir -p "$OUT"
"$KAGGLE" kernels output "$SLUG" -p "$OUT"
echo "[done] $(date -u +%FT%TZ) outputs in $OUT"
