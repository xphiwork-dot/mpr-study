# Running on a Colab T4

Experiments for a free-tier Colab T4 (16 GB, fp16 only, 12 GB RAM, 2 vCPU, sessions can be cut at any time).
Everything is driven from your machine with the `colab` CLI (google-colab-cli, already authorised; the scripts never
prompt). Results and timings are in `results/t4/`; what was actually run is in `docs/reproduction-status.md`.

## One-time

```bash
uv tool install google-colab-cli        # then authorise once, see `colab skill`
colab sessions                          # must print without a login prompt
mkdir -p data && uvx gdown 1a0Ya9buk1I0hsGoaRX8bW0QJjsKuN4FR -O data/mpr_dataset.zip   # the driver uploads this zip
```

## Sessions

Each session script is re-runnable; finished work is skipped, so run it again after a cut session.

| Script | Content |
|---|---|
| `scripts/t4/session1.sh` | VM + vLLM setup, throughput test, retrieval-only error bars (12 users), one-call grid on 900 questions (no memory, Oracle, BM25, e5) |
| `scripts/t4/session2.sh` | chain-of-thought (SequentialReason, 5 steps) grid on the 306-question slice, four memory types |
| `scripts/t4/session3.sh topk` / `steps` | top-k sweep (10/30/50); reasoning-step sweep (2/3/8); two halves because the estimate (`results/t4/estimates.md`) exceeds one session |
| `scripts/t4/session4.sh` | small-model LoRA and HybridMem (5 clusters), no vLLM |

Measured durations and which sessions have been run: `results/t4/timing_log.md` and `docs/reproduction-status.md`.

```bash
scripts/t4/session1.sh                  # needs: colab CLI, data/mpr_dataset.zip, python3
scripts/t4/t4.sh down                   # ALWAYS: pulls results, stops the VM (idle VMs burn quota)
colab sessions                          # must list nothing of yours
python3 scripts/t4/summarize.py grid --tag s1 --slice results/t4/slices/s900.json
```

## The exact `colab` commands the driver issues (`scripts/t4/t4.sh`)

```bash
colab new -s mpr-t4 --gpu T4                          # up (4 attempts, growing sleep, then it reports and stops)
colab upload -s mpr-t4 /tmp/t4_code.tgz /content/t4_code.tgz     # scripts + slice files
colab upload -s mpr-t4 data/mpr_dataset.zip /content/t4_dataset.zip
colab upload -s mpr-t4 results/t4/vm_results.tgz /content/t4_prev.tgz   # earlier results, so a fresh VM resumes
colab exec  -s mpr-t4                                 # (python on stdin) unpack, clone upstream at a24953c, install vLLM
colab exec  -s mpr-t4                                 # start `vllm serve` / start a job in the background / poll it
colab download -s mpr-t4 /content/vm_results.tgz results/t4/vm_results.tgz   # after every job
colab stop -s mpr-t4                                  # down
```

Jobs run as background processes on the VM (`/content/t4/logs/<name>.log`, `<name>.exit`), so a dropped local
connection does not kill them; `t4.sh job` polls once a minute.

## What runs where

- Model: `Qwen/Qwen2.5-7B-Instruct-AWQ` (revision pinned in `t4.sh`), `vllm serve --dtype float16 --quantization awq
  --max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 64 --generation-config vllm`, greedy decoding.
  vLLM picks the Triton attention backend and the Marlin AWQ kernel on the T4 by itself.
- `retrieve.py`: BM25 + e5-base-v2 top-50 for all 32,400 questions, per-question recall@10/20/30/50, embedding cache.
- `grid.py`: NaiveReason / SequentialReason against the server, upstream prompt templates (loaded from the pinned
  clone), memories Ignoramus / Oracle / SparseRAG (BM25) / DenseRAG (e5). One jsonl line per finished question.
- `throughput.py`: cold-cache request throughput at several concurrencies, with real grid prompts.
- `sft.py`: our own small LoRA / HybridMem trainer (fp16, dynamic padding, distinct adapter names).
- `summarize.py`: markdown tables with bootstrap intervals, leak-free rescoring, paired differences.
- Slices: `make_slice.py`, seed 0, stratified by hop (100 per hop = 900 questions over all 12 users; the 34-per-hop
  306-question slice is a subset). The slice files are committed in `results/t4/slices/`.

## Resumability

- Every result line is flushed and fsynced when a question finishes; a rerun reads the file and skips finished
  questions (`uid:rc_id` key). A line cut in half by a dying VM is ignored and redone.
- `retrieve.py` skips users that have a `.done` marker and an embedding cache.
- `sft.py` appends self-ask pairs as they are generated and saves the adapter after every epoch (AdamW moments are
  not saved, so a resumed epoch restarts them).
- The driver pulls `results/` and `logs/` after every job and `down` pulls once more; `up` pushes the last pull back
  to a fresh VM. If the T4 is unavailable, `up` retries four times and exits non-zero.

## Pitfalls hit while building this (all fixed in the scripts)

- Importing `vllm` or `torch` in the Colab kernel opens a CUDA context that takes ~3.5 GB; vLLM then ran out of memory
  on its first request. The driver checks for vLLM with `importlib.util.find_spec` only.
- `bm25s` can pull in `jax`, which preallocates most of the GPU; jobs run with `JAX_PLATFORMS=cpu`.
- A throughput test that repeats the same prompts is answered from vLLM's prefix cache (98% hit rate observed) and
  overstates throughput by an order of magnitude; the test adds a unique request line to every prompt.
