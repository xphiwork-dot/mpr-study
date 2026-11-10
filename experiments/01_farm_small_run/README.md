# 01 - Small run of the authors' code on the Bosch LLM Farm

A 9-question run of the upstream `run_qa.py` with NaiveReason and SequentialReason on the LLM Farm, to check that the authors' pipeline
runs end to end and to see the ordering Oracle > RAG > no memory. See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

- Section 4.1 defines the explicit-memory baselines: SparseRAG, DenseRAG, TreeRAG, GraphRAG, plus Ignoramus (no user statements) and
  Oracle (gold references, "an upper-bound baseline").
- Section 4.2 and Figure 3 ("Overall performances of explicit memory") compare them under four reasoning structures (NR, SR, MR, DR,
  Section 3.4.3 and Figure 2(d)). Claims: SR and MR beat DR and NR by 10% to 20% overall; DenseRAG is best on short hops and SparseRAG on
  long hops; accuracy falls as the hop count grows. Accuracy is Exact Match after normalisation (Section 3.4.1); prompts are in Appendix B.1 (NR) and B.2 (SR).
- The paper uses Qwen2.5-7B on 8 A800 GPUs (Section 3.4.1) and k = 20 (Section 3.4.2).

We run NR and SR only, with Ignoramus, Oracle, SparseRAG and DenseRAG (SR with DenseRAG was not run), a farm-hosted model instead of
Qwen2.5-7B, and only 9 questions. We test only the coarse claim "Oracle >> RAG >> no memory" and whether SR helps; the paper's tables are not reproduced.

## (b) What we ran

- Setup: upstream pinned at `a24953c`, patched by `farm.patch` (Azure-style LLM Farm client; `reasoning_effort=low`; `temperature=0.9` dropped
  because the farm deployment rejects it). User 0, 1 question per hop for hops 2-10 (n = 9), the user's full 13,100-statement memory, k = 20.
  The farm model name is read from `SMALL_MODEL_NAME` at run time and is not recorded in the repo. No GPU was needed for the LLM (it is a hosted API).
- Key numbers, copied from [`results/farm/summary.md`](../../results/farm/summary.md) (EM / lenient contains, n = 9, one question = 0.111):
  NR: Ignoramus 0.11 / 0.11, Oracle 0.89 / 1.00, SparseRAG 0.78 / 0.78, DenseRAG 0.67 / 0.67.
  SR: Ignoramus 0.11 / 0.11, Oracle 0.78 / 0.89, SparseRAG 0.67 / 0.89 (DenseRAG not run).
- Cost, from [`results/farm/usage.md`](../../results/farm/usage.md): 171 calls, 65,471 input and 28,071 output tokens in total.

## (c) High-level code walkthrough

1. `setup.sh` clones upstream at `a24953c` into `upstream/MPR/` (gitignored), applies `farm.patch`, downloads the dataset and calls `make_subset.py`.
2. `make_subset.py` writes `data/subsets/user0_1perhop.json` (all of user 0's statements, 1 sampled question per hop) in the upstream data format.
3. `farm_run.sh Naive|Sequential <memory>` loads credentials from the environment, then runs upstream `run_qa.py <reason> <memory>` inside `upstream/MPR/`
   and prints calls and token totals from the `[Usage]` lines that the patch adds.
4. Upstream writes one JSON line per question under `upstream/MPR/results/<reason>/`; the committed copies are in `results/farm/`.
5. `../_shared/score.py` computes EM and lenient contains per file and per hop group.

| File | What it does |
|---|---|
| `farm.patch` | our diff for upstream `config.py` and `utils.py`: farm client, env-driven paths, usage logging |
| `farm_run.sh` | credential loading (variable names `LLM_FARM_ENDPOINT`, `LLM_FARM_KEY`, `SMALL_MODEL_NAME`; values never written), runs `run_qa.py`, sums token usage |
| `make_subset.py` | stratified question subset with the full user memory |
| `../_shared/score.py` | `normalize`, `em`, `contains`, `score_file`: scorer for the run output |

## (d) Provenance

Upstream code that is **run as is** and is not part of this repo (so it carries no label here): `run_qa.py`, `reasonings/NaiveReason.py`,
`reasonings/SequentialReason.py`, `baselines/{Ignoramus,Oracle,SparseRAG,DenseRAG,BaseMethod}.py`, `baselines/prompts.py`, `utils.py`
(except the lines changed by the patch). They are executed from the gitignored clone.

| Script / function | Label | Reference |
|---|---|---|
| `farm.patch` (config part) | ADAPTED | diff against upstream `config.py`: model paths and `Root_dir` (original lines 7-8), `data_title`/`data_path_dir` read from env (30-31), a new `farm_config`, and `llm_config` of Oracle, Ignoramus, SparseRAG, DenseRAG switched to it (58, 63, 68, 76). |
| `farm.patch` (client part) | ADAPTED | diff against upstream `utils.py`: `AzureOpenAI` import (original line 2), a farm branch in `LLM.__init__` (72-76) and `LLM.run` (90-98, original `temperature=0.9` at line 95 dropped on the farm path), plus usage counters. The Azure client code and `reasoning_effort` are ours (no upstream reference); the surrounding structure is upstream's. Changes are shown with a few upstream lines as diff context. |
| `farm_run.sh` | ORIGINAL TO THIS REPO | no upstream reference. It only calls upstream's command line (`run_qa.py:9-22`, `python run_qa.py <reason> <method>`) and adds credential loading and token accounting. |
| `make_subset.py` | ORIGINAL TO THIS REPO | no upstream reference (upstream ships no dataset or subset tooling). The output follows the data layout that upstream `run_qa.py:36-37` reads. Paper Section 3.4.1 samples 2,700 questions "uniformly across 2-hop to 10-hop"; our per-hop stratification mirrors that idea at 1 per hop. |
| `../_shared/score.py` | ORIGINAL TO THIS REPO | no upstream reference: upstream ships no scorer (`docs/upstream-notes.md` item 2). Paper Section 3.4.1 says EM after normalisation and cites SQuAD [20]; our `normalize` (lowercase, strip punctuation, drop articles a/an/the, collapse spaces) follows that convention. We did not diff it against the SQuAD script, and it is not the paper's scorer, so numbers compare only between our own runs. |
| `../../setup.sh` | ORIGINAL TO THIS REPO | no upstream reference; fetches the upstream commit and the dataset file named in upstream's README (Step 1). |

## (e) How to run

```bash
./setup.sh                                  # needs uv, git
cp env.example .env                         # fill in LLM_FARM_ENDPOINT, LLM_FARM_KEY, SMALL_MODEL_NAME (or export FARM_ENV_FILE=...)
for m in Ignoramus Oracle SparseRAG DenseRAG; do experiments/01_farm_small_run/farm_run.sh Naive $m; done
experiments/01_farm_small_run/farm_run.sh Sequential Oracle      # about 4-5x the tokens of NR; Sequential <memory> likewise
.venv/bin/python experiments/_shared/score.py upstream/MPR/results/*/*.jsonl
```

This spends farm tokens (about 94k for the committed runs, see `results/farm/usage.md`); do not rerun it without a reason. Runtime is minutes
(171 calls). No notebook: the run needs farm credentials and is a 9-question smoke run.

Verified at restructuring (no farm call was made): `bash -n farm_run.sh`; `make_subset.py` builds the subset; `score.py` reproduces the committed
NR rows; `git apply --reverse --check` of `farm.patch` against the existing patched clone passes at its new path. The farm run itself was not repeated.

## (f) Results

Files: [`results/farm/summary.md`](../../results/farm/summary.md), [`results/farm/usage.md`](../../results/farm/usage.md), raw jsonl in `results/farm/NaiveReason/` and `results/farm/SequentialReason/`.

Outcome: with n = 9 only the large gaps mean anything. Oracle (0.89 EM, NR) is far above retrieval (0.67-0.78) and no memory (0.11), the ordering the paper reports.
SR did not beat NR here (Oracle 0.78 vs 0.89, SparseRAG 0.67 vs 0.78); we do not claim the paper's SR gain from this run. The larger T4 runs
([`s1`](../s1_t4_one_call_grid/), [`s2`](../s2_t4_chain_of_thought/)) test the same claims with 900 and 306 questions.
