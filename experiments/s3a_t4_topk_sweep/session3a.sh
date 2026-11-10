#!/usr/bin/env bash
# Session 3a (about 1 h): top-k sweep. NaiveReason, BM25 and e5, k in 10 / 30 / 50, on the 900-question slice
# (k = 20 is session 1, tag s1). The reasoning-step half of the old session 3 is session3b.sh.
# The estimate (results/t4/estimates.md, about 143 min for 3a + 3b) is above one free-tier session, hence two scripts.
# Compare against tag s1 with: python3 experiments/_shared/summarize.py topk   (and grid --tag s3_topk).
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/_shared/t4.sh
$T up || exit 1
$T job retrieve retrieve.py || exit 1     # skips users whose embeddings are already cached
$T serve || exit 1
for k in 10 30 50; do
  $T job "grid3_k$k" grid.py --slice results/slices/s900.json --reason naive --memories sparse,dense --topk $k --tag s3_topk || exit 1
done
python3 experiments/_shared/summarize.py grid --tag s3_topk --slice results/t4/slices/s900.json >/dev/null
python3 experiments/_shared/summarize.py topk >/dev/null
echo "session 3a (top-k) done; stop the VM with: $T down"
