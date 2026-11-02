# S3b - Colab T4 session 3b: reasoning-step sweep (NOT RUN)

**This experiment was not run.** No free T4 could be obtained, so there is no result file and nothing is claimed about 2, 3 or 8 reasoning steps.
The driver, the summary mode and a Colab notebook exist and are ready. See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

- Section 4.4 and **Figure 5** ("Performance of various reasoning steps", mean and standard deviation): accuracy of SR, MR and DR for different numbers of reasoning steps, per hop group
  (short 2-4, medium 5-7, long 8-10) and per memory baseline; the step counts on the x-axis are 1, 3, 5, 7, 9 (read off the figure; the default is 5 steps, Section 3.4.3).
- Claims in Section 4.4: "model performances improve as the number of reasoning steps increases, but the gains beyond 3 steps are not significant"; DR is more sensitive to the step count than SR and MR;
  long-hop questions are more affected than short and medium ones; Oracle is more affected than the other RAG baselines.

Our planned version: SequentialReason only (no MR, no DR), BM25 and e5, k = 20, **2, 3 and 8 steps** (2 steps = start and final only), compared with the 5-step run of [S2](../s2_t4_chain_of_thought/) and the one-call NR of
[S1](../s1_t4_one_call_grid/), on the 306-question slice. Our step counts (2, 3, 5, 8) differ from the paper's ticks (1, 3, 5, 7, 9).

## (b) What we ran

Nothing for the sweep itself. What exists:

- Failed attempts to get a T4: 16 `colab new -s mpr-t4-s3b --gpu T4` calls between 2026-11-01T17:33:57Z and 18:48:12Z (4 rounds of the driver's 4-attempt retry loop, about 75 minutes) all returned
  HTTP 503 "Service Unavailable"; no VM was created. The same account had been granted a T4 on the first try at 16:10Z for session 3a. Whether the pool was empty or a usage limit applied after 80 minutes of 3a VM time is unknown
  ([`timing_log.md`](../../results/t4/timing_log.md), "Session 3b notes").
- A single further attempt for the restructuring task on 2026-11-01 (`colab new -s mpr-docs-smoke --gpu T4`) also returned 503; no session was created and `colab sessions` listed none, so no notebook could be smoke-tested.
- A 4-question smoke test of the 2-step SR path (BM25 and e5, k = 50) and of the NR top-k path ran on a T4 earlier (`results/t4/smoke_s3/`, 8 calls per SR file, 4 questions each). It checks the code path only, not the sweep.
- A planning estimate of about 84 minutes of generation for the three step counts and two retrievers (83.8 minutes in [`estimates.md`](../../results/t4/estimates.md)); the estimate assumes 5-step token counts per call and so
  underestimates the 8-step run. Together with 3a it exceeds one free-tier session, which is why 3b is a separate script.
- `summarize.py steps` was run on the committed data only (one-call NR from `s1` and the 5-step SR from `s2`; the 2, 3, 8-step settings are skipped when their files are missing) to check that it runs; its output was not kept.

## (c) High-level code walkthrough

`session3b.sh`: `t4.sh up` -> `retrieve.py` (embedding cache) -> `t4.sh serve` -> for each of 2, 3, 8 steps one `grid.py` job (`--reason seq --max-step S --memories sparse,dense --topk 20 --tag s3_steps --workers 48`
on the 306-question slice; `LIMIT=150 session3b.sh` shortens it to the first 150 questions) -> `summarize.py grid --tag s3_steps` and `summarize.py steps`.
`summarize.py steps` builds the step-sweep tables: one call, 2/3/5/8 steps (5 from tag `s2`, the others from `s3_steps`), EM and contains with bootstrap intervals, paired differences between consecutive step counts and against one call.

| File | What it does |
|---|---|
| `session3b.sh` | the driver (copy of the `steps` half of the former `session3.sh`) |
| `s3b_t4_step_sweep.ipynb` | the same jobs run on the Colab VM itself; a `LIMIT` setting shortens the run |
| `../_shared/summarize.py: steps_sweep` | the step-sweep analysis |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `../_shared/grid.py: Runner.solve` (seq branch with `--max-step`) | ADAPTED | upstream `reasonings/SequentialReason.py:7-29` (the step loop `range(1, max_step-1)` at line 18; `max_step` comes from `config.py:46`, default 5). See [S2](../s2_t4_chain_of_thought/README.md) (d); the equivalence to upstream was checked for 2, 3, 5 and 8 steps with a stub memory. |
| `../_shared/summarize.py: steps_sweep`, `sweep_report` | ORIGINAL TO THIS REPO | no upstream reference; idea from paper Section 4.4 / Figure 5. |
| `session3b.sh`, `s3b_t4_step_sweep.ipynb` | ORIGINAL TO THIS REPO | no upstream reference. |

## (e) How to run

```bash
experiments/s3b_t4_step_sweep_not_run/session3b.sh          # colab CLI as in S1; T4_SESSION=mpr-t4-s3b keeps it separate; LIMIT=150 shortens it
experiments/_shared/t4.sh down                              # ALWAYS
colab sessions                                              # must list nothing of yours
python3 experiments/_shared/summarize.py steps              # writes results/t4/summary_s3_steps_sweep.md
```

Notebook: [`s3b_t4_step_sweep.ipynb`](s3b_t4_step_sweep.ipynb). **Not executed as a notebook, and the experiment itself has never been run**; the cells are syntax-checked and the file is valid nbformat 4 JSON with no outputs.
Expected runtime: about 84 minutes of generation (estimate; the 8-step run is probably longer) plus about 5 minutes of setup.

## (f) Results

None. There is no `results/t4/s3_steps/` directory. Status: not run (no T4 available, see (b)); the paper's Section 4.4 claim about the number of reasoning steps is neither supported nor refuted by this repo.
The only related evidence is the 5-step SR versus one-call NR comparison in [S2](../s2_t4_chain_of_thought/).
