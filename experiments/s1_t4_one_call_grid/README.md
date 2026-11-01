# S1 - Colab T4 session 1: throughput, retrieval error bars, one-call grid

Session 1 of the Colab T4 runs: serve Qwen2.5-7B-Instruct (4-bit AWQ) with vLLM, measure throughput, measure retrieval recall with error
bars over all 12 users, run the one-call NaiveReason grid (no memory / Oracle / BM25 / e5) on 900 questions, and rescore without leaking questions.
See [`../README.md`](../README.md) for the provenance labels and [`../_shared/README.md`](../_shared/README.md) for the shared driver and helpers.

## (a) The experiment in the paper

- Section 4.2 and **Figure 3** ("Overall performances of explicit memory"): accuracy of Ignoramus, Oracle, SparseRAG, DenseRAG (and TreeRAG,
  GraphRAG) under NR, SR, MR, DR, per hop count. We reproduce the **NR** column for four of the six memory types. Claims tested: Oracle is the upper bound,
  retrieval beats no memory, accuracy falls with the hop count, DenseRAG is better on short hops and SparseRAG on long hops (all Section 4.2).
- Section 3.4.2: k = 20, BM25s for sparse, e5-base-v2 for dense retrieval (copied). Section 3.4.1: accuracy is EM after normalisation (our scorer is not theirs).
- Section 4.6 and Figure 7(a) report average task completion time per hop count. Our throughput test (calls per second on a T4 under vLLM) is **not** that experiment;
  it is an addition for planning the later sessions.
- The 12-user retrieval error bars and the leak-free rescoring are our additions (no counterpart in the paper); the retrieval probe extends [`00`](../00_probes_retrieval_leak/).

Differences from the paper: Qwen2.5-7B-Instruct in 4-bit AWQ on one T4 instead of Qwen2.5-7B on 8 A800 GPUs; greedy decoding; answers capped at 64 tokens;
900 questions over all 12 users instead of the paper's 2,700-question sub-dataset per user; our scorer. Absolute numbers are not comparable with the paper's.

## (b) What we ran

- Hardware: Colab free-tier T4 (16 GB, fp16 only), session `mpr-t4`, created 2026-11-01. Model `Qwen/Qwen2.5-7B-Instruct-AWQ` (revision pinned in `t4.sh`), vLLM 0.32.0
  (`docs/reproduction-status.md` section 6), `--max-model-len 8192`, greedy.
- Slice: [`results/t4/slices/s900.json`](../../results/t4/slices/s900.json), 100 questions per hop for hops 2-10 over all 12 users, seed 0 (900 questions);
  479 of them restate a number from their gold statements (338 with the 3+ digit rule), per [`summary_s1.md`](../../results/t4/summary_s1.md).
- One call per question, memories Ignoramus / Oracle / SparseRAG (BM25) / DenseRAG (e5), k = 20.
- Key numbers (copied from [`results/t4/summary_s1.md`](../../results/t4/summary_s1.md); EM with 95% bootstrap interval, n = 900):
  no memory 0.041 [0.029, 0.054], Oracle 0.257 [0.229, 0.286], BM25 0.114 [0.094, 0.136], e5 0.109 [0.089, 0.129].
  Paired differences on the same questions: Oracle minus BM25 +0.142 [+0.113, +0.170]; e5 minus BM25 -0.006 [-0.027, +0.016]; BM25 minus no memory +0.073 [+0.053, +0.094].
  EM by hop group 2-4 / 5-7 / 8-10 (n = 300 each): Oracle 0.383 / 0.223 / 0.163, BM25 0.200 / 0.063 / 0.080, e5 0.177 / 0.080 / 0.070.
  Leak-free subset (421 questions): no memory 0.014, Oracle 0.290, BM25 0.128, e5 0.121 (versus no memory 0.065 on the 479 leaking questions).
- Retrieval over 12 users (32,399 questions), from [`summary_retrieval.md`](../../results/t4/summary_retrieval.md): recall@20 BM25 0.488, e5 0.498; hop 2 0.877 / 0.997,
  hop 10 0.340 / 0.284; all-gold@20 at hop 10 0.016 / 0.005. e5 leads at hops 2-4, BM25 from hop 5 on.
- Throughput, from [`summary_throughput.md`](../../results/t4/summary_throughput.md): about 2.05-2.10 calls/s at 8, 32 and 64 concurrent requests (prompt-token bound, about 1.1-1.2k prompt tokens/s,
  about 560 prompt tokens per call); one request decodes at 38 tokens/s; five-step SR at 64 concurrent questions 19.2 questions/min.
- Generation time: 1,288 s for the four memories (no memory 165.5 s, Oracle 208.0 s, BM25 435.1 s, e5 479.5 s), 900 calls each; retrieval job 210 s ([`timing_log.md`](../../results/t4/timing_log.md)).
- What went wrong on the way (all in `timing_log.md`): vLLM ran out of GPU memory because the Colab kernel had imported vllm, bm25s pulled in jax which preallocated the GPU,
  a first throughput run was inflated by vLLM's prefix cache and discarded, and the first e5 attempt failed on all questions (a bug in `grid.py`, fixed, rerun).

## (c) High-level code walkthrough

`session1.sh` (from your machine, via the `colab` CLI):

1. `t4.sh up` creates the T4 session, uploads the scripts, the dataset zip and any earlier results, installs vLLM.
2. `t4.sh job retrieve retrieve.py`: with the GPU free, embed every user's statements with e5 and write per-question recall for BM25 and e5 (k = 10/20/30/50) -> `results/t4/retrieval/recall.csv.gz`.
3. `t4.sh serve`: start `vllm serve` for the AWQ model.
4. `t4.sh job thr throughput.py`: cold-cache throughput at 8 / 32 / 64 concurrent requests with real grid prompts (a unique `[req N]` line defeats vLLM's prefix cache).
5. `t4.sh job grid1 grid.py ...`: one-call NaiveReason for the four memories on the 900-question slice; one JSON line per finished question, resumable.
6. `summarize.py throughput | retrieval | grid --tag s1` write the markdown summaries; `t4.sh down` pulls results and stops the VM.

| File | What it does |
|---|---|
| `session1.sh` | the driver above |
| `s1_t4_one_call_grid.ipynb` | the same steps run on the Colab VM itself (no `colab` CLI) |
| `retrieve.py` | retrieval-only pass over 12 users: BM25 and e5 top-50, recall@k and all-gold@k per question, embedding cache; resumable per user |
| `throughput.py` | throughput phases (warm-up, 3 concurrency levels, single-request decode, SR at 64) |
| `make_slice.py` | builds the stratified, nested slices `s900.json` and `s306.json` (seed 0) with the leak flags |
| `../_shared/grid.py` | the question runner (shared with s2, s3a, s3b); `../_shared/lib.py`, `summarize.py`, `score.py`, `t4.sh` |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `retrieve.py` (BM25 and e5 calls) | ADAPTED | upstream `baselines/SparseRAG.py:17-24` (bm25s tokenize / index / retrieve with English stopwords) and `baselines/DenseRAG.py:22-24` (`encode(..., normalize_embeddings=True)`, no prefixes). Ours: top-50 for all questions at once, e5 in fp16, GPU top-k, cached embeddings. |
| `retrieve.py` (recall@k / all-gold@k, per-user resume, merge to csv) | ORIGINAL TO THIS REPO | no upstream reference (upstream has no retrieval metric); extends our probe [`00`](../00_probes_retrieval_leak/). |
| `throughput.py` | ORIGINAL TO THIS REPO | no upstream reference. It drives `grid.Runner.solve` with a nonce prefix; the paper's efficiency analysis (Section 4.6, Figure 7) measures task completion time, a different quantity. |
| `make_slice.py` | ORIGINAL TO THIS REPO | no upstream reference. The paper samples a stratified sub-dataset uniformly over 2-10 hops (Section 3.4.1); the nested slice, the seeded shuffle and the leak flags (via `leak_check.numbers`) are ours. |
| `session1.sh` | ORIGINAL TO THIS REPO | no upstream reference (driver script). |
| `s1_t4_one_call_grid.ipynb` | ORIGINAL TO THIS REPO | no upstream reference; generated by `../_shared/make_notebooks.py`. |
| Shared files used here: `grid.py`, `lib.py`, `summarize.py`, `score.py`, `t4.sh` | see [`../_shared/README.md`](../_shared/README.md) (d) | `grid.py` re-implements upstream `NaiveReason.py:7-8` + the four memory classes (ADAPTED); the rest is mostly ORIGINAL. |

## (e) How to run

T4 via the `colab` CLI (from your machine; see [`../_shared/README.md`](../_shared/README.md) for the one-time setup and the exact `colab` commands the driver issues):

```bash
uv tool install google-colab-cli && colab sessions            # authorise once
mkdir -p data && uvx gdown 1a0Ya9buk1I0hsGoaRX8bW0QJjsKuN4FR -O data/mpr_dataset.zip
experiments/s1_t4_one_call_grid/session1.sh                   # rerunnable: finished work is skipped
experiments/_shared/t4.sh down                                # ALWAYS: pulls results, stops the VM
colab sessions                                                # must list nothing of yours
python3 experiments/_shared/summarize.py grid --tag s1 --slice results/t4/slices/s900.json
```

Notebook: open [`s1_t4_one_call_grid.ipynb`](s1_t4_one_call_grid.ipynb) in Colab (Runtime > T4 GPU) and run all cells; it clones the repo (private: add a Colab secret `GITHUB_TOKEN`),
installs vLLM, serves the model, runs the same jobs, saves to `/content` and optionally Google Drive. **Not executed as a notebook**: this session was run with the CLI driver, and no free T4 was
available (16 requests returned 503 on 2026-11-01) to smoke-test the notebook. Its cells are syntax-checked and the file is valid nbformat 4 JSON with no outputs.

Expected runtime: clean run about 40-45 minutes after the one-time install (retrieval 210 s, vLLM start about 100 s, throughput 603 s, grid 1,288 s, plus polling); the real session took 76 minutes
from VM creation to the end of the grid because of the debugging above (`timing_log.md`).

## (f) Results

Files: [`results/t4/s1/`](../../results/t4/s1/) (raw jsonl, `timing.jsonl`), [`summary_s1.md`](../../results/t4/summary_s1.md), [`summary_retrieval.md`](../../results/t4/summary_retrieval.md),
[`summary_throughput.md`](../../results/t4/summary_throughput.md), [`retrieval/recall.csv.gz`](../../results/t4/retrieval/recall.csv.gz), [`throughput/throughput.jsonl`](../../results/t4/throughput/throughput.jsonl).

Outcome: same ordering as the paper (Oracle 0.257 > BM25 0.114 about equal to e5 0.109 > no memory 0.041 EM); retrieval beats no memory, and e5 versus BM25 is not detectable overall,
though retrieval recall shows e5 ahead on short hops and BM25 ahead from hop 5. On the 421 leak-free questions the no-memory score drops to 0.014, so part of the no-memory score is arithmetic on the question text.
Absolute numbers are far from the paper's (4-bit 7B model, greedy, our scorer).

Verified at restructuring: `summarize.py grid --tag s1`, `retrieval` and `throughput` regenerate the committed files with only the generator path in line 1 changed; `make_slice.py` reproduces
`s900.json` and `s306.json` byte for byte; `py_compile` and the flat-layout import check pass for `retrieve.py`, `throughput.py`, `make_slice.py`. The T4 jobs themselves were not rerun.
