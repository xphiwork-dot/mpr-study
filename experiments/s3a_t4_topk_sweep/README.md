# S3a - Colab T4 session 3a: top-k sweep (k = 10 / 20 / 30 / 50)

One-call NaiveReason with BM25 and e5 at k = 10, 30, 50 on the 900-question slice (k = 20 is session 1), plus the retrieval recall of the same questions.
See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

- Section 4.3 and **Figure 4** ("Performance of various retrieved statement counts"): heatmaps of accuracy over hop count and retrieved statement count for the four RAG baselines
  (DenseRAG, SparseRAG, TreeRAG, GraphRAG) and four reasoning structures. The retrieval counts on the x-axis of the figure are 10, 30, 50 (read off the figure; the default k is 20, Section 3.4.2).
- Claims in Section 4.3: "in short-hop questions, the accuracy typically improves as k increases" (explained by higher recall), and "in long-hop questions, there exists a peak of accuracy at an intermediate k value"
  (explained by noise and low precision at large k).

We test these two claims for NR with SparseRAG and DenseRAG only, at 7B in 4-bit, on 900 questions (hops 2-4 are the paper's short-hop group, 8-10 the long-hop group of Figure 5; Figure 4's own grouping is by single hop count).

## (b) What we ran

- Hardware and model as in [S1](../s1_t4_one_call_grid/): free-tier Colab T4, `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM, greedy; session `mpr-t4-s3` (created 2026-11-01, T4 granted on the first try, stopped after 80 minutes).
- Slice [`s900.json`](../../results/t4/slices/s900.json) (900 questions, 100 per hop, all 12 users); NaiveReason, memories SparseRAG and DenseRAG, k = 10, 30, 50 (6 new run files, 900 calls each, no failed calls); k = 20 comes from S1.
- Key numbers (copied from [`summary_s3_topk_sweep.md`](../../results/t4/summary_s3_topk_sweep.md); overall EM, n = 900, 95% intervals about +-0.02 are in the file):
  BM25 k = 10 / 20 / 30 / 50: 0.123 / 0.114 / 0.110 / 0.116; e5: 0.109 / 0.109 / 0.108 / 0.117.
  Retrieval recall@k of the same questions rises clearly: BM25 0.376 / 0.478 / 0.530 / 0.595, e5 0.387 / 0.507 / 0.577 / 0.660 (all-gold@k BM25 0.136 / 0.186 / 0.221 / 0.282, e5 0.171 / 0.229 / 0.270 / 0.338).
  Paired differences on the same questions, k = 50 minus k = 10, overall EM: BM25 -0.008 [-0.028, +0.011], e5 +0.008 [-0.011, +0.027]. Short hops 2-4 (n = 300): BM25 +0.010 [-0.027, +0.047], e5 -0.007 [-0.047, +0.033].
  Long hops 8-10 (n = 300): EM stays between 0.053 and 0.080 for every k and both retrievers.
  Of the 60 paired dEM intervals in the file (6 k-pairs x 5 subsets x 2 retrievers) 5 exclude 0, all negative (a larger k scored lower), none positive; the intervals are not corrected for multiple comparisons.
- Time: generation 3,977 s = 66.3 minutes (estimate 58.8), VM wall clock 80 minutes; the first `up` call timed out in the colab client and was resolved by rerunning the driver ([`timing_log.md`](../../results/t4/timing_log.md)).

## (c) High-level code walkthrough

`session3a.sh`: `t4.sh up` -> `retrieve.py` (embedding cache; skipped per user if already there) -> `t4.sh serve` -> three `grid.py` jobs (k = 10, 30, 50) with `--reason naive --memories sparse,dense --tag s3_topk`
-> `summarize.py grid --tag s3_topk` and `summarize.py topk`. `summarize.py topk` loads the k = 20 runs from tag `s1`, the others from `s3_topk`, joins the retrieval recall of the same questions from
`results/t4/retrieval/recall.csv.gz`, and writes the scores, EM by hop, paired differences between k values and BM25 versus e5 (all with bootstrap intervals).

| File | What it does |
|---|---|
| `session3a.sh` | the driver (the reasoning-step half of the old session 3 is in [`../s3b_t4_step_sweep/`](../s3b_t4_step_sweep/)) |
| `s3a_t4_topk_sweep.ipynb` | the same jobs run on the Colab VM itself |
| `../_shared/grid.py` (naive branch, `--topk`) | the runner; `../_shared/summarize.py: topk_sweep, sweep_report` the analysis |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `../_shared/grid.py: Runner.solve` (naive branch) and `Runner.refs` with `--topk` | ADAPTED | upstream `reasonings/NaiveReason.py:7-8` (one call with the question as retrieval key), `baselines/SparseRAG.py:21-26` and `baselines/DenseRAG.py:32-39` (top-k by `self.topk`, taken from `config.py:26` `retrieval_topk = 20` and the `retrieval_config` entries). Prompt: `baselines/prompts.py:2-25` (`Naive_RAG_Prompt_Template`). Ours varies k through `--topk`; upstream has no sweep driver. |
| `../_shared/summarize.py: topk_sweep`, `sweep_report` | ORIGINAL TO THIS REPO | no upstream reference (no analysis code upstream). Paired bootstrap, leak-free subset, per-hop tables are ours; the experiment idea is paper Section 4.3 / Figure 4. |
| `session3a.sh`, `s3a_t4_topk_sweep.ipynb` | ORIGINAL TO THIS REPO | no upstream reference. |
| other shared files | see [`../_shared/README.md`](../_shared/README.md) (d) | |

## (e) How to run

```bash
experiments/s3a_t4_topk_sweep/session3a.sh        # colab CLI, data/mpr_dataset.zip as in S1; rerunnable
experiments/_shared/t4.sh down                    # ALWAYS
colab sessions                                    # must list nothing of yours
python3 experiments/_shared/summarize.py topk     # needs results/t4/s1 and results/t4/retrieval (committed)
```

Notebook: [`s3a_t4_topk_sweep.ipynb`](s3a_t4_topk_sweep.ipynb). **Not executed as a notebook**: the session was run with the CLI driver; no free T4 was available to smoke-test the notebook (see
[S3b](../s3b_t4_step_sweep/)). Expected runtime: about 66 minutes of generation plus about 5 minutes of setup; the real session needed 80 minutes of VM time (`timing_log.md`).

## (f) Results

Files: [`results/t4/s3_topk/`](../../results/t4/s3_topk/) (raw jsonl, `timing.jsonl`), [`summary_s3_topk_sweep.md`](../../results/t4/summary_s3_topk_sweep.md), [`summary_s3_topk.md`](../../results/t4/summary_s3_topk.md).

Outcome: retrieval recall rises clearly with k, but answer EM is flat (BM25 0.110-0.123, e5 0.108-0.117), so the paper's claim that larger k helps short-hop questions is **not reproduced** at this scale; no intermediate-k peak is visible at long hops either
(not reproduced, not refuted: the data cannot see effects below about 0.04 per hop group). One slice and seed, a 4-bit 7B model, greedy decoding, EM floors of 0.05-0.10 at long hops.

Verified at restructuring: `summarize.py grid --tag s3_topk` and `summarize.py topk` regenerate `summary_s3_topk.md` and `summary_s3_topk_sweep.md` identically apart from the generator path in line 1. The T4 jobs were not rerun.
