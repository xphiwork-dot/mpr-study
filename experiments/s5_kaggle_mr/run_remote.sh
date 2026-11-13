#!/usr/bin/env bash
# Push the one private MR kernel while holding the shared GPU lock through
# terminal status and output download.  The lock must outlive `kaggle kernels
# push` (which returns as soon as the version is queued), otherwise the peer
# can start its job while this one is still running.
#
# Usage: experiments/s5_kaggle_mr/run_remote.sh [output-dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
LOCK="${KAGGLE_GPU_LOCK:-/home/xphi/.local/state/mpr-study/kaggle-gpu.lock}"
SLUG="phinguynxun/mpr-mr-reasoning"
OUT="${1:-$ROOT/results/kaggle/mr/_kaggle_out}"
WAIT_S="${GPU_LOCK_WAIT_S:-21600}"

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -w "$WAIT_S" 9; then
  echo "[lock] timed out after ${WAIT_S}s waiting for $LOCK" >&2
  exit 1
fi
echo "[lock] acquired $(date -u +%FT%TZ)"
trap 'echo "[lock] releasing $(date -u +%FT%TZ)"' EXIT

"$KAGGLE" kernels push -p "$HERE" --accelerator NvidiaTeslaT4 --timeout 14400

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
