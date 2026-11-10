# S3b - Colab T4 session 3b: reasoning-step sweep

Run on 2026-11-02 on a free-tier Colab T4 (SequentialReason with 2, 3 and 8 steps, BM25 and e5, k = 20, 306 questions). It could not run on 2026-11-01 because no T4 was available (HTTP 503). Together with the
5-step run of [S2](../s2_t4_chain_of_thought/) and the one-call baseline of [S1](../s1_t4_one_call_grid/) it gives the step sweep; the result is in (f). See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

- Section 4.4 and **Figure 5** ("Performance of various reasoning steps", mean and standard deviation): accuracy of SR, MR and DR for different numbers of reasoning steps, per hop group
  (short 2-4, medium 5-7, long 8-10) and per memory baseline; the step counts on the x-axis are 1, 3, 5, 7, 9 (read off the figure; the default is 5 steps, Section 3.4.3).
- Claims in Section 4.4: "model performances improve as the number of reasoning steps increases, but the gains beyond 3 steps are not significant"; DR is more sensitive to the step count than SR and MR;
  long-hop questions are more affected than short and medium ones; Oracle is more affected than the other RAG baselines.

Our planned version: SequentialReason only (no MR, no DR), BM25 and e5, k = 20, **2, 3 and 8 steps** (2 steps = start and final only), compared with the 5-step run of [S2](../s2_t4_chain_of_thought/) and the one-call NR of
[S1](../s1_t4_one_call_grid/), on the 306-question slice. Our step counts (2, 3, 5, 8) differ from the paper's ticks (1, 3, 5, 7, 9).

## (b) What we ran

- `session3b.sh` through the `colab` CLI driver, session `mpr-t4-s3b`, on 2026-11-02: 2, 3 and 8 steps x BM25 and e5, 306 questions each, 6 new run files (7,956 calls, no failed or empty responses). The 5-step and one-call
  settings are not re-run: they are the committed [S2](../s2_t4_chain_of_thought/) (tag `s2`) and [S1](../s1_t4_one_call_grid/) (tag `s1`, restricted to the same 306 questions) runs. Nothing was dropped from the plan.
- Time: 108.3 minutes of generation (estimate 83.8; 29% slower, the 8-step runs took 74 minutes), 127.7 minutes of VM time for the VM that produced the results ([`timing_log.md`](../../results/t4/timing_log.md), "Session 3b run notes").
- Not smooth: three VMs were needed. The first two were lost without a usable result (setup call hung for 30 minutes, then a VM reported lost during the first job; details in the timing log). Nothing from them is in the results.
  The fix to the driver is in `../_shared/t4.sh` (setup runs detached and is polled; `colab exec` calls are time-limited; partial results are pulled about every 10 minutes). The jobs and flags are unchanged.
- Setup: this VM had no vLLM, so vllm 0.32.0 / torch 2.13.0 / transformers 5.19.0 / Python 3.13.16 were installed with `uv pip install vllm ...` (full list in the timing log). vLLM and torch match what session 1 recorded; the
  other package versions of the earlier sessions were not recorded, so a small difference to the session-2 5-step run cannot be excluded. The install procedure is the same as in session 1.
- The earlier failed attempts of 2026-11-01 (16 `colab new` calls answered with 503 over 75 minutes) are described in the timing log, "Session 3b notes, first try".
- A 4-question smoke test of the 2-step SR path ran earlier (`results/t4/smoke_s3/`); it was a code-path check only.

## (c) High-level code walkthrough

`session3b.sh`: `t4.sh up` -> `retrieve.py` (embedding cache) -> `t4.sh serve` -> for each of 2, 3, 8 steps one `grid.py` job (`--reason seq --max-step S --memories sparse,dense --topk 20 --tag s3_steps --workers 48`
on the 306-question slice; `LIMIT=150 session3b.sh` shortens it to the first 150 questions) -> `summarize.py grid --tag s3_steps` and `summarize.py steps`.
`summarize.py steps` builds the step-sweep tables: one call, 2/3/5/8 steps (5 from tag `s2`, the others from `s3_steps`), EM and contains with bootstrap intervals, paired differences between consecutive step counts, 8 vs 2, 8 vs 3, and each against one call.

| File | What it does |
|---|---|
| `session3b.sh` | the driver (copy of the `steps` half of the former `session3.sh`); run |
| `s3b_t4_step_sweep.ipynb` | the same jobs run on the Colab VM itself; a `LIMIT` setting shortens the run; not executed (the real run used the CLI driver) |
| `../_shared/summarize.py: steps_sweep` | the step-sweep analysis |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `../_shared/grid.py: Runner.solve` (seq branch with `--max-step`) | ADAPTED | upstream `reasonings/SequentialReason.py:7-29` (the step loop `range(1, max_step-1)` at line 18; `max_step` comes from `config.py:46`, default 5). See [S2](../s2_t4_chain_of_thought/README.md) (d); the equivalence to upstream was checked for 2, 3, 5 and 8 steps with a stub memory. |
| `../_shared/summarize.py: steps_sweep`, `sweep_report` | ORIGINAL TO THIS REPO | no upstream reference; idea from paper Section 4.4 / Figure 5. |
| `session3b.sh`, `s3b_t4_step_sweep.ipynb` | ORIGINAL TO THIS REPO | no upstream reference. |

## (e) How to run

```bash
T4_SESSION=mpr-t4-s3b experiments/s3b_t4_step_sweep/session3b.sh   # colab CLI as in S1; LIMIT=150 shortens it; rerun after a lost VM, finished questions are skipped
T4_SESSION=mpr-t4-s3b experiments/_shared/t4.sh down               # ALWAYS: the script does not stop the VM
colab sessions                                                     # must list nothing of yours
python3 experiments/_shared/summarize.py steps                     # writes results/t4/summary_s3_steps_sweep.md
```

Notebook: [`s3b_t4_step_sweep.ipynb`](s3b_t4_step_sweep.ipynb). **Not executed as a notebook**; the cells are syntax-checked and the file is valid nbformat 4 JSON with no outputs. The real run used `session3b.sh`.
Runtime seen: 108 minutes of generation plus 5 to 10 minutes of setup and vLLM start (VM time 128 minutes).

## (f) Results

Slice `s306.json`: 306 questions, 34 per hop (hops 2-10), 12 users; greedy decoding, Qwen2.5-7B-Instruct-AWQ, k = 20, our scorer. Intervals are 95% bootstrap over questions (2,000 resamples); "paired" means a paired bootstrap on the same questions.
One call = NaiveReason (S1 run restricted to these questions), 5 steps = S2, 2/3/8 steps = this session. Raw results: `results/t4/s3_steps/*.jsonl`; all tables: `results/t4/summary_s3_steps_sweep.md` (EM, contains, leak-free, per hop, all paired
differences, BM25 minus e5) and `results/t4/summary_s3_steps.md`.

Overall EM (all 306 questions):

| memory | one call | 2 steps | 3 steps | 5 steps (S2) | 8 steps |
|---|---|---|---|---|---|
| BM25 | 0.114 [0.082, 0.150] | 0.199 [0.157, 0.245] | 0.229 [0.183, 0.278] | 0.193 [0.150, 0.242] | 0.203 [0.160, 0.248] |
| e5 | 0.127 [0.092, 0.167] | 0.199 [0.154, 0.245] | 0.245 [0.196, 0.294] | 0.222 [0.173, 0.271] | 0.225 [0.176, 0.271] |

Overall lenient contains: BM25 0.134 / 0.209 / 0.232 / 0.199 / 0.219; e5 0.154 / 0.212 / 0.255 / 0.239 / 0.239 (same column order; intervals in the summary).

Paired EM differences, all 306 questions (A minus B):

| memory | 2 steps - one call | 3 - 2 | 5 - 3 | 8 - 5 | 8 - 3 | 8 - 2 | 8 - one call |
|---|---|---|---|---|---|---|---|
| BM25 | +0.085 [+0.042, +0.131] | +0.029 [+0.000, +0.059] | -0.036 [-0.069, -0.003] | +0.010 [-0.020, +0.042] | -0.026 [-0.065, +0.013] | +0.003 [-0.033, +0.042] | +0.088 [+0.042, +0.137] |
| e5 | +0.072 [+0.029, +0.114] | +0.046 [+0.013, +0.078] | -0.023 [-0.056, +0.007] | +0.003 [-0.029, +0.036] | -0.020 [-0.059, +0.020] | +0.026 [-0.010, +0.065] | +0.098 [+0.052, +0.144] |

What it shows, relative to the paper's claim (Section 4.4: more reasoning steps help, gains beyond 3 steps not significant):

- Any number of steps beats one call: the paired difference to one call is positive with an interval above 0 for all four step counts and both retrievers (overall). This repeats the S2 finding for 2, 3 and 8 steps.
- Between step counts the evidence is thin. Of the 50 paired step-versus-step EM intervals (2 retrievers x 5 step pairs x 5 subsets: all, leak-free, hops 2-4, 5-7, 8-10), 5 exclude 0, all of them involving 3 steps:
  3 vs 2 positive for e5 (overall, hops 5-7) and BM25 (hops 8-10), and 5 vs 3 negative for BM25 (overall, hops 8-10). None of the 8-vs-5, 8-vs-3 or 8-vs-2 intervals excludes 0. The intervals are not corrected for the many comparisons.
- 3 steps has the highest overall EM for both retrievers, but the curve is not monotone: BM25 at 5 steps is below 3 steps (paired -0.036, interval just excludes 0) and 8 steps is nominally below 3 steps (8 - 3: BM25 -0.026 [-0.065, +0.013], e5 -0.020 [-0.059, +0.020]; both include 0). So the data do not support "more steps keep helping" up to 8.
  They are consistent with "a small gain from 2 to 3 steps, then no detectable gain": e5 3 vs 2 is +0.046 [+0.013, +0.078] but BM25 3 vs 2 only reaches +0.029 with a lower bound of 0.000. No gain from 3 to 8 steps is detectable.
- That agrees with the second half of the paper's sentence (no significant gain beyond 3 steps). It does not show the first half beyond what S2 showed (one call versus a chain), and it does not show the paper's pattern that long-hop questions benefit most:
  hops 8-10 gain from 2 to 3 steps for BM25 (+0.059 [+0.020, +0.108]) but not for e5 (+0.069 [+0.000, +0.137], boundary), and no hop group shows a detectable gain from 3 to 8 steps.
- Caveats: one slice (306 questions, 102 per hop group), one seed, greedy decoding, a 4-bit 7B model, SequentialReason only (no MR or DR), our scorer, EM around 0.2 so differences below about 0.04 cannot be seen, and our step counts (2, 3, 5, 8, where 2 = start and
  final only) differ from the paper's ticks (1, 3, 5, 7, 9). The 5-step run is from another VM and day (see (b)). BM25 and e5 do not differ detectably at any step count.
