# Reproduction status

What of the MPR paper (Explicit v.s. Implicit Memory, KDD '26) is reproduced in this repo, what is not, and
where the evidence is. This file is the single source of truth; the README only links here.

Sources for every number below: the committed files under `results/`, or the scout report
(`docs/scout-report.html`, cited by section). The goal is to understand the paper's ideas, not to match its tables.
Setup the results were produced with: user 0 (13,100 statements, 2,700 questions), upstream commit `a24953c`.

Status values: reproduced / partially / not reproduced.

## 1. No-LLM probes (run here, reproducible offline)

| Item | Status | Evidence | Reason / notes |
|---|---|---|---|
| Retrieval recall probe, BM25 (bm25s) | reproduced | `results/probes/retrieval_recall.md` | Recall@20 of gold facts, user 0: hop 2 0.847, hop 10 0.351; all-gold@20 at hop 10 is 0.003. Matches the scout numbers (scout report section 5). |
| Retrieval recall probe, dense e5-base-v2 | reproduced | `results/probes/retrieval_recall.md` | Recall@20: hop 2 1.000, hop 10 0.257; all-gold@20 at hop 10 is 0.000. Dense is ahead on hops 2-3, tied on hop 4 (0.570 vs 0.563), BM25 ahead from hop 5 on. Run without the `query:`/`passage:` prefixes, as upstream does. |
| Leakage check | reproduced, with a rule difference | `results/probes/leak_check.md` | All 12 users, 32,399 questions: any restated number 0.53 overall, 0.82 at hop 10; 3+ digit numbers 0.38 overall, 0.62 at hop 10. The scout report states 0.51, 0.82, 0.61. Hops 5-10 agree within about 0.02; hops 2-4 are higher here (0.13, 0.07, 0.27 vs the scout's 0.08, 0.03, 0.19). The scout's exact number rule was not recorded; this script uses a plain `\d+(\.\d+)?` match. |
| Scorer (`scripts/score.py`) | reproduced (our own code) | `results/farm/summary.md` | Upstream ships no scorer. Ours reports SQuAD-normalised EM and a lenient whole-word "contains". It is not the paper's scorer, so numbers are only comparable between our own runs. |

## 2. Explicit memory on the LLM Farm (small run)

The farm patch (`patches/farm.patch`) adds an Azure OpenAI client for the Bosch LLM Farm to upstream `utils.py` and
`config.py`: credentials read from the environment at run time, `reasoning_effort=low`, and `temperature` dropped
(the farm model rejects the paper's 0.9; `docs/upstream-notes.md` item 3).

| Item | Status | Evidence | Reason / notes |
|---|---|---|---|
| LLM Farm patch | reproduced (our own code) | `patches/farm.patch`, `results/farm/` | Needed because upstream's Qwen server is not shipped. The raw outputs below show it works end to end. |
| Small run, NR (NaiveReason, 1 call per question) | partially | `results/farm/NaiveReason/*.jsonl`, `results/farm/summary.md` | Memory types run: Ignoramus, Oracle, SparseRAG, DenseRAG (all 4). Only 9 questions (user 0, 1 per hop 2-10), k = 20, a farm model instead of Qwen2.5-7B. EM / lenient: Ignoramus 0.11 / 0.11, Oracle 0.89 / 1.00, SparseRAG 0.78 / 0.78, DenseRAG 0.67 / 0.67. One question is 0.11, so only the large gaps mean anything (Oracle >> RAG >> no memory, as in the scout's 45-question run). RAG scores are higher than the scout's (SparseRAG 0.444, DenseRAG 0.311); that is sampling noise on a different 9-question draw, not evidence against the scout. |
| Small run, SR (SequentialReason, 5 calls per question) | partially | `results/farm/SequentialReason/*.jsonl`, `results/farm/summary.md` | Memory types run: Ignoramus, Oracle, SparseRAG (3 of 4). EM / lenient: Ignoramus 0.11 / 0.11, Oracle 0.78 / 0.89, SparseRAG 0.67 / 0.89. SR did not beat NR here, as in the scout's run. We do not claim the paper's SR gain. |
| Farm token usage of the small run | reproduced | `results/farm/usage.md` | 171 calls, 65,471 input and 28,071 output tokens in total, summed from the run logs (the logs are not committed). |
| SR with DenseRAG | not reproduced | none | Not run, to save farm budget. The scout's 45-question grid has it (SR DenseRAG 0.333 EM, scout report section 5.4A). |
| 45-question grid (5 per hop, 8 cells) | not reproduced | none | Not re-run here. The scout run cost 1,080 farm calls (about 466k input + 183k output tokens, scout report section 5.4A). Scout NR EM: Ignoramus 0.200, Oracle 0.756, SparseRAG 0.444, DenseRAG 0.311. `scripts/make_subset.py --per-hop 5` builds the question set. |

## 3. Implicit memory and hybrids

| Item | Status | Evidence | Reason / notes |
|---|---|---|---|
| Implicit memory, LoRA SFT at 7B (50 adapters) | not reproduced | none | Does not fit a 6 GB laptop GPU (scout report, summary and section 5.4B). |
| Implicit memory, 0.5B probe (Qwen2.5-0.5B, AskSFT) | not reproduced here | scout report section 5.4B (original run only) | The scout ran it once: LoRA recalls at most 11% of trained facts, SFT+RAG no better than RAG. Its scripts and adapters were not rebuilt in this repo, so no committed result files exist. |
| HybridMem (BlockSFT: k-means clusters, one LoRA per cluster) | not reproduced | none | Upstream crashes when it trains and evaluates in one run (`ValueError: Adapter 1 not found`, `BlockSFT.py:209`; `docs/upstream-notes.md` item 5). The scout worked around it only at 5 clusters on 0.5B. Not run here. |
| MaskSFT | not reproduced | none | Not run (needs LoRA training). |

## 4. Other explicit-memory methods and reasoning styles

| Item | Status | Evidence | Reason / notes |
|---|---|---|---|
| TreeRAG | not reproduced | none | Builds its index with an LLM over about 13.1k facts per user. Scout estimate: thousands of LLM calls (scout report, risks list). Never measured. |
| GraphRAG | not reproduced | none | Same: LLM entity/relation extraction over about 13.1k facts per user, thousands of calls estimated by the scout, never measured. |
| MR and DR reasoning | not reproduced | none | Not run; only NR and SR were run. |

## 5. Paper numbers

| Item | Status | Evidence | Reason / notes |
|---|---|---|---|
| Paper tables on Qwen2.5-7B (8 A800 GPUs) | not reproduced, not comparable | none | Different model (farm model), tiny subset, our own scorer. Absolute numbers cannot be compared; only the ordering Oracle >> RAG >> no memory and the retrieval pattern carry over. |

## What a Google Colab T4 could add next

Plans, not results. Nothing in this section has been run. A free T4 has 16 GB VRAM, supports fp16 only (no bf16),
and sessions are time-limited and can be reclaimed, so every job needs checkpoints and a resumable output file.

- Explicit-memory baselines with a real 7B: Qwen2.5-7B-Instruct as a 4-bit AWQ build served with vLLM, running
  Ignoramus / Oracle / SparseRAG / DenseRAG with NR and SR on a 300-900 question slice (stratified by hop).
- Top-k sweep and reasoning-step sweep (number of SR steps) on the same slice, to test the retrieval-mismatch story.
- MR and DR reasoning, which were not run here.
- Leak-free rescoring: score the slice again with the questions that restate gold numbers removed (the rule in
  `scripts/leak_check.py`), to separate arithmetic on the question text from real memory use.
- Small-model LoRA (0.5B-3B) on a 300-2000 fact slice with the SFT padding bug fixed (upstream pads every sample to
  4,608 tokens; `docs/upstream-notes.md` item 4), in fp16.
- HybridMem with few clusters (for example 5) and the adapter-name crash fixed (`docs/upstream-notes.md` item 5).

Not planned for a T4: TreeRAG and GraphRAG (the index build cost is in LLM calls, not GPU), and 7B LoRA with 50 adapters.
