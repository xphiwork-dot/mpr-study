# S2 - Colab T4 session 2: five-step chain-of-thought grid (SequentialReason)

The same four memories as session 1, but with the 5-step chain-of-thought structure (SequentialReason, SR) on a 306-question slice, so SR can be
compared with the one-call NR of [`s1`](../s1_t4_one_call_grid/) on the same questions. See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

- Section 3.4.3 defines the reasoning structures (Figure 2(d)): NR, SR (chain of thought, "each inference is based on the results from the last step"), MR, DR; the number of
  reasoning steps is set to 5 unless stated otherwise. Appendix B.2 gives the SR prompts (starting, thinking, answering).
- Section 4.2 and **Figure 3**: "The performance of SR and MR is significantly higher than DR and NR, with an overall improvement of 10% to 20%", and "NR is weaker than other
  multi-hop structures, especially on long-hop questions". Also: SR and MR "mitigate the decline on Oracle", while RAG methods still degrade quickly with the hop count.
- Equation in Section 3.2: in SR the retrieval for step i uses the question and earlier thoughts (`RAG(o_1 || ... || o_{i-1} || q, S)`). Upstream's `SequentialReason` retrieves with the
  previous thought as the query (see (d)); we follow the code, not the equation.

We test only SR versus NR (no MR, no DR) on one model, with four memory types, on 306 questions.

## (b) What we ran

- Hardware and model as in S1: free-tier Colab T4, `Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM, greedy, session `mpr-t4-s2` (created 2026-11-01), k = 20, 5 steps, up to 200 tokens per thought and 64 for the answer, 48 workers.
- Slice: [`results/t4/slices/s306.json`](../../results/t4/slices/s306.json), 34 questions per hop for hops 2-10 over all 12 users, seed 0, a subset of the 900-question slice (the same questions as in S1); 140 of the 306 are leak-free.
- Calls: 1,530 per memory type (306 questions x 5 calls), no failed calls ([`summary_s2.md`](../../results/t4/summary_s2.md)).
- Key numbers (copied from [`summary_s2.md`](../../results/t4/summary_s2.md); EM [95% CI], n = 306): no memory 0.056 [0.033, 0.082], Oracle 0.513 [0.454, 0.569],
  BM25 0.193 [0.150, 0.239], e5 0.222 [0.176, 0.271]; e5 minus BM25 +0.029 [-0.020, +0.078] (not detectable).
- SR versus NR on the same 306 questions, from [`summary_s2_vs_s1.md`](../../results/t4/summary_s2_vs_s1.md) (paired SR minus NR, EM): BM25 +0.078 [+0.033, +0.124], e5 +0.095 [+0.049, +0.137],
  Oracle +0.258 [+0.203, +0.310], no memory +0.013 [-0.010, +0.036]. Leak-free (140 questions): BM25 +0.057 [-0.007, +0.121], e5 +0.093 [+0.029, +0.157], Oracle +0.200 [+0.121, +0.279].
- Time: job walls 586 / 661 / 1,187 / 1,242 s for no memory / Oracle / BM25 / e5 (about 61 minutes of generation), in-job prompt rate 937-1,079 tokens/s;
  the planning estimate was 47 minutes ([`timing_log.md`](../../results/t4/timing_log.md), [`estimates.md`](../../results/t4/estimates.md)). VM wall clock 71 minutes. The first `up` call timed out in the colab client and
  was resolved by rerunning the driver (`timing_log.md`).

## (c) High-level code walkthrough

`session2.sh`: `t4.sh up` -> `retrieve.py` (re-creates the e5 embedding cache; a no-op if restored) -> `t4.sh serve` -> for each memory type one `grid.py` job with
`--reason seq --max-step 5 --topk 20 --tag s2 --workers 48` on `results/slices/s306.json` (cheapest memory first, so a cut session loses least) -> `summarize.py grid --tag s2`.

Inside `grid.py` for one question (SR): retrieve with the question -> **start** thought (prompt `Sequential_Start`) -> for steps 2..n-1 retrieve with the previous thought and ask the
**middle** prompt with all earlier thoughts joined as `(Step i) text` -> retrieve with the last thought and ask the **final** prompt for the answer. The result line stores the response, the gold answer,
the first retrieved ids, the thoughts and token counts.

| File | What it does |
|---|---|
| `session2.sh` | the driver |
| `s2_t4_chain_of_thought.ipynb` | the same jobs run on the Colab VM itself |
| `../_shared/grid.py` (`Runner.solve`, seq branch) | the SR call flow; `../_shared/summarize.py compare` builds the SR-vs-NR table |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `../_shared/grid.py: Runner.solve` (seq branch) | ADAPTED | upstream `reasonings/SequentialReason.py:7-29`: start thought (12-15), middle loop `range(1, max_step-1)` (18-24), final answer (26-29), the `'(Step %d) %s'` join (23, 28), retrieval keyed on the question for the start and on `previous_thoughts[-1]` afterwards (15, 24, 29). Prompt templates are upstream's `baselines/prompts.py:115-206` (`Sequential_{Start,Middle,Final}_{RAG,Ignoramus}_Prompt_Template`), loaded from the pinned clone at run time. Ours calls a vLLM endpoint with 200-token thoughts and a 64-token answer instead of upstream's `LLM.fast_run` (`utils.py:110-119`). Checked: with a stub memory, our prompts and retrieval keys equal upstream's for 2, 3, 5 and 8 steps. |
| `../_shared/grid.py: Runner.refs` | ADAPTED | `baselines/BaseMethod.py:41-43` (`response`: retrieve then infer), `baselines/Oracle.py:15-17`, `baselines/SparseRAG.py:21-26`, `baselines/DenseRAG.py:32-39`, `baselines/Ignoramus.py:14-15`. |
| `../_shared/summarize.py: compare` | ORIGINAL TO THIS REPO | no upstream reference (upstream ships no analysis code); paired bootstrap SR minus NR is ours. |
| `session2.sh`, `s2_t4_chain_of_thought.ipynb` | ORIGINAL TO THIS REPO | no upstream reference. |
| other shared files | see [`../_shared/README.md`](../_shared/README.md) (d) | |

## (e) How to run

```bash
experiments/s2_t4_chain_of_thought/session2.sh          # needs the colab CLI and data/mpr_dataset.zip, as in S1; rerunnable
experiments/_shared/t4.sh down                          # ALWAYS (session2.sh does not stop the VM)
colab sessions                                          # must list nothing of yours
python3 experiments/_shared/summarize.py grid --tag s2 --slice results/t4/slices/s306.json
python3 experiments/_shared/summarize.py compare --tag s2 --nr-tag s1 --slice results/t4/slices/s306.json
```

Notebook: [`s2_t4_chain_of_thought.ipynb`](s2_t4_chain_of_thought.ipynb) (same cell structure as S1; the comparison cell uses the committed `s1` results from the clone). **Not executed as a notebook**:
this session was run with the CLI driver, and no free T4 was available to smoke-test the notebook. Expected runtime: about 60-70 minutes of generation plus about 5 minutes of setup (a free T4 session can be cut; rerun to resume).

## (f) Results

Files: [`results/t4/s2/`](../../results/t4/s2/), [`summary_s2.md`](../../results/t4/summary_s2.md), [`summary_s2_vs_s1.md`](../../results/t4/summary_s2_vs_s1.md).

Outcome: SR beat one-call NR for both retrievers on all questions (BM25 +0.078, e5 +0.095 EM, intervals exclude 0), in the direction of the paper's Section 4.2 claim; the BM25 gain is not clearly nonzero
on the leak-free subset. Oracle gains most (+0.258) and no memory gains nothing, so the RAG gain cannot be credited to re-retrieval alone (SR also makes five calls). One slice, one seed, a 4-bit 7B model: we do not know the size of the gain in general.

Verified at restructuring: `summarize.py grid --tag s2` and `compare --tag s2 --nr-tag s1` regenerate `summary_s2.md` and `summary_s2_vs_s1.md` identically apart from the generator path in line 1; the SR call flow was
checked against upstream with the stub run described in (d). The T4 jobs were not rerun.
