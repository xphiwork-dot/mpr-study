#!/usr/bin/env bash
# Session 3: top-k sweep and reasoning-step sweep. The estimate (results/t4/estimates.md, about 143 min in total)
# is above one free-tier session, so the two halves run separately:
#   session3.sh topk    NaiveReason, BM25 and e5, k in 10 / 30 / 50, on the 900-question slice (k = 20 is session 1); about 1 h
#   session3.sh steps   SequentialReason with 2 / 3 / 8 steps, BM25 and e5, k = 20, on the 306-question slice
#                       (5 steps is session 2; 2 steps = start + final only); about 1.5 h, shorten with LIMIT=150
# Compare against tags s1 / s2 with: python3 scripts/t4/summarize.py grid --tag s3_topk (and s3_steps).
set -uo pipefail
cd "$(dirname "$0")/../.."
T=scripts/t4/t4.sh
PART="${1:-topk}"
LIM=()
[ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
$T up || exit 1
$T job retrieve retrieve.py || exit 1     # skips users whose embeddings are already cached
$T serve || exit 1
if [ "$PART" = topk ]; then
  for k in 10 30 50; do
    $T job "grid3_k$k" grid.py --slice results/slices/s900.json --reason naive --memories sparse,dense --topk $k --tag s3_topk || exit 1
  done
  python3 scripts/t4/summarize.py grid --tag s3_topk --slice results/t4/slices/s900.json >/dev/null
else
  for s in 2 3 8; do
    $T job "grid3_s$s" grid.py --slice results/slices/s306.json --reason seq --max-step $s --memories sparse,dense --topk 20 --tag s3_steps --workers 48 "${LIM[@]}" || exit 1
  done
  python3 scripts/t4/summarize.py grid --tag s3_steps --slice results/t4/slices/s306.json >/dev/null
fi
echo "session 3 ($PART) done; stop the VM with: $T down"
