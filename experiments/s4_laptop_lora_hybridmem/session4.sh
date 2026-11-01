#!/usr/bin/env bash
# Session 4 (about 1.5 h): small-model LoRA and HybridMem.
#   0. smoke test (2 epochs, 40 facts, 5 clusters): checks the code path in a few minutes
#   1. self-ask pairs for the 600-fact slice, written by the 7B vLLM server (stage ask)
#   2. server stopped; single LoRA, then HybridMem with 5 clusters, same facts and pairs, learning rate 2e-4
#      ("strong") and the upstream 1e-5 (stage train)
# Raw outputs land in results/t4/sft/<tag>/c<clusters>_<preset>/ (eval.jsonl, summary.json).
set -uo pipefail
cd "$(dirname "$0")/../.."
T=experiments/_shared/t4.sh
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
$T up || exit 1
$T serve || exit 1
$T job sft_smoke_ask sft.py --model "$MODEL" --clusters 5 --smoke --stage ask --tag sft_smoke || exit 1
$T job sft600_ask sft.py --model "$MODEL" --facts 600 --questions 60 --stage ask --tag sft600 || exit 1
$T unserve
$T job sft_smoke sft.py --model "$MODEL" --clusters 5 --smoke --stage train --tag sft_smoke || exit 1
for preset in strong upstream; do
  for c in 1 5; do
    $T job "sft_c${c}_$preset" sft.py --model "$MODEL" --facts 600 --questions 60 --clusters $c --lr-preset $preset --stage train --tag sft600 || exit 1
  done
done
echo "session 4 done; stop the VM with: $T down"
