#!/usr/bin/env bash
# Session 1 (about 1.5 h): setup, vLLM, throughput test, one-call grid (no memory / Oracle / BM25 / e5) on 900
# questions, retrieval-only error bars over all 12 users. Rerun it after a cut session: finished work is skipped.
# After it, run locally:  python scripts/t4/summarize.py throughput|retrieval|grid --tag s1
set -uo pipefail
cd "$(dirname "$0")/../.."
T=scripts/t4/t4.sh
$T up || exit 1
# retrieval first: it needs the GPU without the vLLM server holding most of it
$T unserve
$T job retrieve retrieve.py || exit 1
$T serve || exit 1
$T job thr throughput.py --slice results/slices/s900.json --seconds 90 || exit 1
$T job grid1 grid.py --slice results/slices/s900.json --reason naive --memories ignoramus,oracle,sparse,dense --topk 20 --tag s1 || exit 1
python3 scripts/t4/summarize.py throughput >/dev/null
python3 scripts/t4/summarize.py retrieval >/dev/null
python3 scripts/t4/summarize.py grid --tag s1 --slice results/t4/slices/s900.json >/dev/null
echo "session 1 done; stop the VM with: $T down"
