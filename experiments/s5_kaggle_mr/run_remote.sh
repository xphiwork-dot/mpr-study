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
SLUG="xunphi/mpr-mr-reasoning"
DATASET="xunphi/mpr-mr-inputs"
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

# A just-uploaded dataset is not mountable until Kaggle finishes processing it.
# Poll its status first (up to ~5 min) so the kernel does not fail on a missing
# /kaggle/input mount.
for i in $(seq 1 30); do
  ds_status="$("$KAGGLE" datasets status "$DATASET" 2>&1 | tail -n 1)"
  echo "[dataset] $(date -u +%FT%TZ) $ds_status"
  case "$ds_status" in
    *ready*) break ;;
    *error*|*failed*) echo "[dataset] $DATASET is not ready: $ds_status" >&2; exit 1 ;;
  esac
  [ "$i" -eq 30 ] && { echo "[dataset] timed out waiting for $DATASET" >&2; exit 1; }
  sleep 10
done

# Verified recipe on this account: `competition_sources` + machine_shape
# NvidiaH100 in kernel-metadata.json routes to an immediately-allocated 2x
# Tesla T4 session (driver 580, CUDA 13, internet on) whereas plain
# NvidiaTeslaT4 stayed QUEUED.  vLLM itself still runs one replica on GPU 0.
"$KAGGLE" kernels push -p "$HERE" --accelerator NvidiaH100 --timeout 14400

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
