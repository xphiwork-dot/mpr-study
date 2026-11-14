# experiments/

One folder per experiment or session. Open a folder's `README.md` to deep-dive it: the paper experiment it relates to, what we ran
(with numbers copied from the committed result files), a code walkthrough, a provenance table, how to run it, and the outcome.

Paper: "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information" (MPR, KDD '26),
`papers/2508.13250v1.pdf` (arXiv v1; section, figure and table numbers below are those of that PDF).
Original code: https://github.com/nuster1128/MPR, pinned at commit `a24953c` ("upstream" below).

| Folder | What | Paper counterpart | Status |
|---|---|---|---|
| [`00_probes_retrieval_leak/`](00_probes_retrieval_leak/) | no-LLM retrieval-recall probe and dataset leakage check | none (our addition; motivated by Sections 3.3.1, 4.2, 4.3) | run |
| [`01_farm_small_run/`](01_farm_small_run/) | 9-question run of upstream `run_qa.py` on the Bosch LLM Farm | Section 4.2, Figure 3 (NR and SR, four memory types) | run, tiny |
| [`s1_t4_one_call_grid/`](s1_t4_one_call_grid/) | T4 session 1: throughput, 12-user retrieval error bars, one-call grid (900 questions), leak-free rescoring | Section 4.2, Figure 3 (NR column); Section 4.6 is related | run |
| [`s2_t4_chain_of_thought/`](s2_t4_chain_of_thought/) | T4 session 2: 5-step chain-of-thought grid (306 questions) and SR vs NR | Section 4.2, Figure 3 (SR vs NR) | run |
| [`s3a_t4_topk_sweep/`](s3a_t4_topk_sweep/) | T4 session 3a: top-k sweep k = 10 / 20 / 30 / 50 (900 questions) | Section 4.3, Figure 4 | run |
| [`s3b_t4_step_sweep/`](s3b_t4_step_sweep/) | T4 session 3b: reasoning-step sweep | Section 4.4, Figure 5 | run (3 steps best by a small margin; no detectable gain from 3 to 8 steps) |
| [`s4_laptop_lora_hybridmem/`](s4_laptop_lora_hybridmem/) | session 4 on the laptop: small LoRA (AskSFT) and HybridMem | Sections 5 and 6, Table 3, Figures 9 and 10 | run, small |
| [`s6_kaggle_dr/`](s6_kaggle_dr/) | session 6 on Colab T4 (Kaggle packaging kept but the Kaggle account gets no accelerator): DecompositionReason (DR) with Qwen2.5-7B-Instruct-AWQ, 306 questions x BM25/e5 | Sections 4.2 (MR/DR) compared with our NR and SR runs | run: 612/612 rows; EM 0.131 (BM25) / 0.118 (e5), below SR and MR on this slice - see `s6_kaggle_dr/report.md` |
| [`_shared/`](_shared/) | helpers used by several experiments, the `colab` driver `t4.sh`, T4 runbook | n/a | n/a |

Raw outputs and summaries stay in [`../results/`](../results/); `docs/reproduction-status.md` is the single status table.

## Provenance labels

Every README has a provenance table with one of three labels per script and non-trivial function:

- **BORROWED**: identical or nearly identical to the original; cites `nuster1128/MPR` at `a24953c` with file and line range.
- **ADAPTED**: derived from original code; says how it differs and cites the original file and lines (and our patch if any).
- **ORIGINAL TO THIS REPO**: no reference exists in the paper's code; the table says so plainly ("no upstream reference").
  Where the code was written from the paper's text only, the table cites the paper section.

No script is labelled BORROWED: the upstream repo has no licence, so nothing from it is copied here. The code is ADAPTED (re-implemented
against the same call flow and prompt templates) or ORIGINAL. The upstream prompt templates are read at run time from the pinned clone
(`upstream/MPR/baselines/prompts.py`, gitignored) and never stored in this repo. `farm.patch` is a diff against two upstream files
(it necessarily shows a few upstream lines as diff context); `setup.sh` applies it to the clone.

The ADAPTED claims were checked against the original, not assumed: the line ranges below are from `git show a24953c:<file>`, and
for the reasoning loops a mock run (upstream `NaiveReason` / `SequentialReason` source from the pinned commit with a stub memory versus
`_shared/grid.py` with a fake model, 2, 3, 5 and 8 steps) gave identical prompts and identical retrieval keys.

## Layout conventions

- Scripts import shared helpers through a relative `../_shared` path, so run them from the repo root (`python experiments/<folder>/<script>.py`).
- T4 drivers (`session*.sh`) and notebooks run the same jobs; the drivers use the `colab` CLI from your machine, the notebooks run on
  the Colab VM itself. `_shared/t4.sh` copies the scripts flat into `/content/t4/scripts/t4/` on the VM.
- `experiments/_shared/make_notebooks.py` regenerates the four notebooks.
