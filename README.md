# mpr-study

Private study repo for the paper "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over
Personalized Information" (MPR, KDD '26). It holds a study guide with diagrams for a Data Mining final
presentation, plus a small, cheap reproduction of the parts of the paper that can be re-run on a laptop:
the retrieval behaviour of explicit memory (BM25 and dense e5), a dataset leakage check, and an optional
tiny LLM re-run of the paper's baselines on the Bosch LLM Farm. The goal is to understand the paper's ideas,
not to match its tables.

## Start here

1. Open `docs/index.html` (the study guide: paper in 60 seconds, glossary, 4 flow charts, quizzes, talk plan).
2. Run `scripts/retrieval_recall.py`: why one question-only retrieval cannot find the facts of a long chain.
3. Run `scripts/leak_check.py`: why long-hop accuracy partly measures arithmetic on the question text.
4. Skim `docs/upstream-notes.md`, then (optional) run the farm baselines with `scripts/farm_run.sh`.
5. Read upstream, in `upstream/MPR/` after setup: `reasonings/SequentialReason.py`, `baselines/SparseRAG.py`,
   `baselines/DenseRAG.py`, `baselines/BlockSFT.py` (HybridMem), `baselines/SelfAskSFT.py`, `prompts.py`, `utils.py`.

## Layout

```
README.md
setup.sh                  venv (uv) + pinned upstream clone + patch + dataset + question subset
requirements.txt          pinned deps (torch is installed by setup.sh)
env.example               variable names only (farm credentials), no values
patches/farm.patch        our diff for upstream utils.py / config.py (LLM Farm client, env-driven paths)
scripts/
  retrieval_recall.py     no-LLM probe: BM25 / e5 recall@20 of gold facts per hop (user 0)
  leak_check.py           share of questions restating a number from their own gold statements
  score.py                SQuAD-normalised EM + lenient contains for run_qa.py output
  make_subset.py          stratified question subset in upstream format
  farm_run.sh             run run_qa.py on the LLM Farm (credentials read at run time)
  common.py               dataset loading
docs/
  index.html              study guide (diagrams/ holds its 4 iframe charts)
  scout-report.html       full technical notes of the original scouting run
  upstream-notes.md       upstream gaps and bugs, with file and line
upstream/  data/  results/   gitignored: fetched clone, dataset, run outputs
```

## Quick start

```bash
./setup.sh                                  # needs uv and git; TORCH_INDEX=<url> for a GPU wheel
.venv/bin/python scripts/retrieval_recall.py        # BM25 ~3 s; e5 ~3 min incl. the first model download
.venv/bin/python scripts/leak_check.py --show-example
# optional, spends farm tokens (about 95k for the grid below): needs LLM_FARM_ENDPOINT, LLM_FARM_KEY, SMALL_MODEL_NAME
cp env.example .env   # fill in, or: export FARM_ENV_FILE=/path/to/a/file/that/defines them
for m in Ignoramus Oracle SparseRAG DenseRAG; do scripts/farm_run.sh Naive $m; done
.venv/bin/python scripts/score.py upstream/MPR/results/*/*.jsonl
```

`farm_run.sh Sequential <memory>` runs the 5-step chain-of-thought structure (about 4-5x the tokens).
The subset is 1 question per hop (9 questions) on user 0 with the user's full 13,100-statement memory.
`python scripts/make_subset.py --per-hop 5` plus `MPR_DATA_TITLE=user0_5perhop` gives the 45-question
set of the original scout (about 650k tokens).

## What is reproduced, and what is not

Reproduced (numbers from this repo's scripts, user 0, 2,700 questions):

| Probe | Result here | Scout report |
|---|---|---|
| BM25 recall@20, hop 2 / hop 10 | 0.847 / 0.351 | 0.847 / 0.351 |
| e5-base-v2 recall@20, hop 2 / hop 10 | 1.000 / 0.257 | 1.000 / 0.257 |
| Pattern | dense wins on hops 2-3, BM25 wins from hop 5 on; almost no hop-10 question gets all gold facts in one retrieval (all-gold@20: BM25 0.003, e5 0.000) | same |

The full per-hop table is printed by the script and matches the scout's recall columns.

Leakage (`leak_check.py`, all 32,399 non-null questions): any number restated from gold, overall 0.53, hop 10
0.82; numbers with 3+ digits, hop 10 0.62. The scout report states 0.51, 0.82 and 0.61. Hops 5-10 agree
within about 0.02; hops 2-4 come out higher here (0.13, 0.07, 0.27 vs 0.08, 0.03, 0.19). The scout's exact number
rule was not recorded, so this script uses a plain `\d+(\.\d+)?` match; this is a rule difference, not a data difference.

Small LLM re-run on the farm (user 0, 9 questions = 1 per hop 2-10, k=20, model from `SMALL_MODEL_NAME`, reasoning effort low;
about 94k farm tokens in total). EM / lenient contains:

| Reasoning | Ignoramus | Oracle | SparseRAG | DenseRAG |
|---|---|---|---|---|
| NR (1 call) | 0.11 / 0.11 | 0.89 / 1.00 | 0.78 / 0.78 | 0.67 / 0.67 |
| SR (5 calls) | 0.11 / 0.11 | 0.78 / 0.89 | 0.67 / 0.89 | not run (budget) |

With n = 9, one question is 0.11, so only the large gaps mean anything: Oracle >> RAG >> no memory, as in the scout's
45-question run (NR: Ignoramus 0.20, Oracle 0.76, SparseRAG 0.44, DenseRAG 0.31). RAG scores are higher here than in
the scout's run; that is sampling noise on a different 9-question draw, not evidence against the scout. SR did not beat NR,
as in the scout. We do not claim to reproduce the paper's SR gain (+10-20 points) or any Qwen2.5-7B number.

Not reproduced:
- The paper's tables on Qwen2.5-7B (8 A800 GPUs): the farm model is different, so absolute numbers are not comparable.
- Implicit memory (LoRA SFT) and HybridMem at paper scale (7B, 50 adapters): does not fit a 6 GB laptop. The scout's
  0.5B probe (LoRA recalls at most 11% of trained facts; SFT+RAG no better than RAG) is documented in
  `docs/scout-report.html` section 5.4B and its scripts were not rebuilt here.
- TreeRAG, GraphRAG, MaskSFT, MR and DR reasoning: not run.
- DenseRAG with SR and the 45-question grid: not re-run here; see the scout report for those numbers.

Known upstream problems (missing server, no scorer, HybridMem train-then-eval crash, SFT padding, prompt mismatch): `docs/upstream-notes.md`.

## Upstream license

The upstream repo https://github.com/nuster1128/MPR has no LICENSE file. Nothing from it is copied or vendored
here: `setup.sh` clones it at commit `a24953c` into the gitignored `upstream/` and applies `patches/farm.patch`
(our own diff). The dataset (Google Drive zip) is downloaded into the gitignored `data/` and never committed.

## Reference

Zhang, Zhang, Tan, Li, Chen. "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over
Personalized Information". KDD '26, pp. 1964-1974. DOI 10.1145/3770854.3780184. arXiv 2508.13250.
The analysis here used arXiv v1 (2508.13250v1); check the KDD camera-ready for changes before presenting.
