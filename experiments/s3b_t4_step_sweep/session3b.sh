#!/usr/bin/env bash
# Session 3b (about 1.5 h): reasoning-step sweep. NOT RUN: no free T4 could be obtained (16 attempts returned 503 on
# 2026-11-01, see results/t4/timing_log.md). The script exists and the step-summary mode was smoke-tested on committed
# runs only.
#   SequentialReason with 2 / 3 / 8 steps, BM25 and e5, k = 20, on the 306-question slice
#   (5 steps is session 2, tag s2; 2 steps = start + final only). Shorten with LIMIT=150 (first N questions).
# Compare against tags s1 / s2 with: python3 experiments/_shared/summarize.py steps
# Use T4_SESSION=mpr-t4-s3b to keep this session separate from earlier ones (the name used for the failed attempts).
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/_shared/t4.sh
LIM=()
[ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
$T up || exit 1
$T job retrieve retrieve.py || exit 1     # skips users whose embeddings are already cached
$T serve || exit 1
for s in 2 3 8; do
  $T job "grid3_s$s" grid.py --slice results/slices/s306.json --reason seq --max-step $s --memories sparse,dense --topk 20 --tag s3_steps --workers 48 "${LIM[@]}" || exit 1
done
python3 experiments/_shared/summarize.py grid --tag s3_steps --slice results/t4/slices/s306.json >/dev/null
python3 experiments/_shared/summarize.py steps >/dev/null
echo "session 3b (steps) done; stop the VM with: $T down"
