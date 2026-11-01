#!/usr/bin/env bash
# Session 2 (about 1.5 h): chain-of-thought grid (SequentialReason, 5 steps) on the 306-question nested slice,
# four memory types. Needs session 1's retrieval cache only for the e5 memory; `up` restores it from the
# last pulled results/t4/vm_results.tgz, and retrieve.py skips users already done.
# Order is cheapest memory first so a cut session loses the least; rerun to continue.
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/_shared/t4.sh
$T up || exit 1
$T job retrieve retrieve.py || exit 1   # no-op when the embeddings are restored; needed before vLLM starts
$T serve || exit 1
for mem in ignoramus oracle sparse dense; do
  $T job "grid2_$mem" grid.py --slice results/slices/s306.json --reason seq --max-step 5 --memories $mem --topk 20 --tag s2 --workers 48 || exit 1
done
python3 experiments/_shared/summarize.py grid --tag s2 --slice results/t4/slices/s306.json >/dev/null
echo "session 2 done; stop the VM with: $T down"
