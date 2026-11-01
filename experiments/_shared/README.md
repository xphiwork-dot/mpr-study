# Running on a Colab T4 (shared driver and helpers)

This folder holds the pieces every T4 session uses: `t4.sh` (the `colab` CLI driver), `lib.py`, `grid.py`, `summarize.py`,
`score.py`, `common.py`, `estimate.py`. Each session has its own folder with a README, its driver script and a Colab notebook:
`experiments/s1_t4_one_call_grid/`, `s2_t4_chain_of_thought/`, `s3a_t4_topk_sweep/`, `s3b_t4_step_sweep_not_run/`.

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
| `experiments/s1_t4_one_call_grid/session1.sh` | VM + vLLM setup, throughput test, retrieval-only error bars (12 users), one-call grid on 900 questions (no memory, Oracle, BM25, e5) |
| `experiments/s2_t4_chain_of_thought/session2.sh` | chain-of-thought (SequentialReason, 5 steps) grid on the 306-question slice, four memory types |
| `experiments/s3a_t4_topk_sweep/session3a.sh` | top-k sweep (10/30/50), NaiveReason, BM25 and e5, 900 questions; run |
| `experiments/s3b_t4_step_sweep_not_run/session3b.sh` | reasoning-step sweep (2/3/8); NOT run (no T4 available); the estimate (`results/t4/estimates.md`) exceeds one session together with 3a, hence two scripts |
| `experiments/s4_laptop_lora_hybridmem/session4.sh` | small-model LoRA and HybridMem (5 clusters) on a T4, no vLLM during training; only the smoke test ran on a T4, the real run was done on the laptop |

Measured durations and which sessions have been run: `results/t4/timing_log.md` and `docs/reproduction-status.md`.

```bash
experiments/s1_t4_one_call_grid/session1.sh   # needs: colab CLI, data/mpr_dataset.zip, python3
experiments/_shared/t4.sh down              # ALWAYS: pulls results, stops the VM (idle VMs burn quota)
colab sessions                          # must list nothing of yours
python3 experiments/_shared/summarize.py grid --tag s1 --slice results/t4/slices/s900.json
```

## The exact `colab` commands the driver issues (`experiments/_shared/t4.sh`)

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

In the repo the scripts live in `experiments/_shared/` and the per-session folders and import each other through a relative
`../_shared` path; `up` copies them flat into `/content/t4/scripts/t4/` on the VM (so `t4.sh job NAME grid.py ...` takes a bare file name),
and the slice files to `/content/t4/results/slices/`.

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

## (d) Provenance of the shared files

Labels as in [`../README.md`](../README.md); "upstream" is `nuster1128/MPR` at commit `a24953c`. These files are used by several experiments (S1, S2, S3a, S3b; `lib.py`, `score.py`
and `common.py` also by the probes and S4).

| File / function | Label | Reference |
|---|---|---|
| `lib.py: load_user`, `all_questions` | ORIGINAL TO THIS REPO | no upstream reference. They read the released dataset layout (`message`, `qa`) that upstream `run_qa.py:36-37, 45` also reads, and skip the one null question of the release. |
| `lib.py: load_prompts` | ORIGINAL TO THIS REPO | no upstream reference for the loader. It imports `upstream/MPR/baselines/prompts.py` from the gitignored pinned clone at run time; the template text is upstream's and is never stored in this repo. |
| `lib.py: read_jsonl`, `JsonlWriter`, `qkey` | ADAPTED | upstream `utils.py:58-69` (`save_jsonl_add`, `load_jsonl`: one JSON line appended per record). Ours adds a lock, flush and fsync after each line, skips a half-written last line, and a `uid:rc_id` key for resuming. |
| `lib.py: chat` | ADAPTED | upstream `utils.py:110-119` (`LLM.fast_run`) sends one user turn; upstream calls an OpenAI-style client with `temperature=0.9` (`utils.py:92-96`) and retries on an empty result. Ours posts the same single-user-turn message to a local vLLM server with greedy decoding, returns token counts and retries on transport errors. |
| `lib.py: UserIndex` (`sparse`, `dense`) | ADAPTED | upstream `baselines/SparseRAG.py:14-26` (bm25s index and top-k retrieve, English stopwords) and `baselines/DenseRAG.py:22-39` (normalised e5 embeddings, score by `matmul`, sort, top-k). Ours caches the embeddings from `retrieve.py`, scores with numpy, and returns ids. |
| `lib.py: get_encoder` | ADAPTED | upstream `DenseRAG.py:14` (`SentenceTransformer(encoder_path)` on cuda); ours converts it to fp16 on the GPU. |
| `lib.py: load_slice`, `write_gz_csv`, `now` | ORIGINAL TO THIS REPO | no upstream reference. |
| `grid.py: Runner.solve` (naive and seq branches) | ADAPTED | upstream `reasonings/NaiveReason.py:7-8` and `reasonings/SequentialReason.py:7-29`; see the S2 README (d). Mock-checked against the upstream source: identical prompts and retrieval keys for NR and for SR with 2, 3, 5 and 8 steps. |
| `grid.py: Runner.refs`, `Runner.call` | ADAPTED | memories from upstream `Ignoramus.py:14-22`, `Oracle.py:15-17` (join the gold statements as given, ids can repeat), `SparseRAG.py:21-26`, `DenseRAG.py:32-39`; template choice `<call_type>_RAG` / `_Ignoramus` from `BaseMethod.py:46` and `Ignoramus.py:18` (we use `getattr` instead of `eval`). |
| `grid.py: main`, `Runner.user` (resume, thread pool, per-file timing, result line) | ORIGINAL TO THIS REPO | no upstream reference for the driver; the result fields `k_hop`, `rc_id`, `response`, `ground_truth` follow `run_qa.py:53-59`, with extra fields for tokens, retrieved ids and thoughts. |
| `score.py` | ORIGINAL TO THIS REPO | no upstream reference (upstream has no scorer, `docs/upstream-notes.md` item 2). The normalisation follows the SQuAD convention that paper Section 3.4.1 cites for EM ([20]); not diffed against the SQuAD script. |
| `common.py` | ORIGINAL TO THIS REPO | no upstream reference (dataset loader for the probes). |
| `summarize.py` (all modes), `estimate.py` | ORIGINAL TO THIS REPO | no upstream reference (no analysis code upstream). Bootstrap and paired-bootstrap intervals are ours; EM and contains come from `score.py`. |
| `t4.sh` | ORIGINAL TO THIS REPO | no upstream reference. It pins upstream `a24953c` and the model revision and wraps the `colab` CLI. |
| `make_notebooks.py` | ORIGINAL TO THIS REPO | no upstream reference; mirrors `t4.sh` and the session drivers. |

Not verified in the restructuring task: the end-to-end `t4.sh` flow and the notebooks (one `colab new --gpu T4` attempt on 2026-11-01 returned HTTP 503 again, no session was created and `colab sessions` showed none; the staging step that copies the scripts flat was exercised only by building the same layout in a temp directory and importing every script from it).

