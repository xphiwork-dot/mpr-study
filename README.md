# mpr-study

Private study repo for the paper "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over
Personalized Information" (MPR, KDD '26). It holds a study guide with diagrams for a Data Mining final
presentation, plus a small, cheap reproduction of the parts of the paper that can be re-run on a laptop:
the retrieval behaviour of explicit memory (BM25 and dense e5), a dataset leakage check, and an optional
tiny LLM re-run of the paper's baselines on the Bosch LLM Farm. The goal is to understand the paper's ideas,
not to match its tables.

## Start here

1. Read the paper (arXiv v1): `papers/2508.13250v1.pdf`.
2. Open `docs/index.html` (the study guide: paper in 60 seconds, glossary, 4 flow charts, quizzes, talk plan).
3. Run `scripts/retrieval_recall.py`: why one question-only retrieval cannot find the facts of a long chain.
4. Run `scripts/leak_check.py`: why long-hop accuracy partly measures arithmetic on the question text.
5. Read `docs/reproduction-status.md` (what is reproduced) and skim `docs/upstream-notes.md`, then (optional) run the farm baselines with `scripts/farm_run.sh`.
6. Read upstream, in `upstream/MPR/` after setup: `reasonings/SequentialReason.py`, `baselines/SparseRAG.py`,
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
  t4/                     Colab T4 experiments: driver, 4 session scripts, vLLM runner, summaries (t4/README.md)
results/                  committed evidence (the rest of results/ is gitignored)
  probes/                 retrieval_recall.md, leak_check.md: per-hop tables printed by the scripts
  farm/                   raw run_qa.py jsonl (NaiveReason/, SequentialReason/), summary.md, usage.md
  t4/                     Colab T4 run: raw jsonl, retrieval error bars, throughput, timing log, summary_*.md
docs/
  reproduction-status.md  what is and is not reproduced, with evidence and reasons
  index.html              study guide (diagrams/ holds its 4 iframe charts)
  scout-report.html       full technical notes of the original scouting run
  upstream-notes.md       upstream gaps and bugs, with file and line
upstream/  data/  results/logs/   gitignored: fetched clone, dataset, raw farm logs
```

## Quick start

```bash
./setup.sh                                  # needs uv and git; TORCH_INDEX=<url> for a GPU wheel
.venv/bin/python scripts/retrieval_recall.py        # BM25 ~3 s; e5 ~40 s on a GPU after the first model download (~3 min including it)
.venv/bin/python scripts/leak_check.py --show-example
# optional, spends farm tokens (about 94k for the runs listed in `results/farm/usage.md`): needs LLM_FARM_ENDPOINT, LLM_FARM_KEY, SMALL_MODEL_NAME
cp env.example .env   # fill in, or: export FARM_ENV_FILE=/path/to/a/file/that/defines them
for m in Ignoramus Oracle SparseRAG DenseRAG; do scripts/farm_run.sh Naive $m; done
.venv/bin/python scripts/score.py upstream/MPR/results/*/*.jsonl
.venv/bin/python scripts/score.py results/farm/*/*.jsonl   # the committed outputs of our run
```

Committed results: `results/probes/` (the two probe tables) and `results/farm/` (raw jsonl, `summary.md`, `usage.md`).
To regenerate a probe table, rerun the script and compare with the file; behind a corporate proxy set
`REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` (e5 downloads its model on first use).

`farm_run.sh Sequential <memory>` runs the 5-step chain-of-thought structure (about 4-5x the tokens).
The subset is 1 question per hop (9 questions) on user 0 with the user's full 13,100-statement memory.
`python scripts/make_subset.py --per-hop 5` plus `MPR_DATA_TITLE=user0_5perhop` gives the 45-question
set of the original scout (about 650k tokens).

## Running on a Colab T4

A free-tier T4 serves Qwen2.5-7B-Instruct (4-bit AWQ) with vLLM. Four re-runnable sessions, driven by the `colab` CLI:
`scripts/t4/session1.sh` .. `session4.sh`. Commands, resumability notes and pitfalls: [`scripts/t4/README.md`](scripts/t4/README.md).
Session 1 (throughput, 12-user retrieval error bars, one-call grid on 900 questions, leak-free rescoring) has been run;
results are in `results/t4/` and summarised in [`docs/reproduction-status.md`](docs/reproduction-status.md) section 6.
Sessions 2-4 are scripted and smoke-tested only. Always finish with `scripts/t4/t4.sh down`.

## What is reproduced, and what is not

Full tables (status, evidence file, reason for every gap) and a plan for what a Colab T4 could add:
[`docs/reproduction-status.md`](docs/reproduction-status.md). In short:

- Reproduced: BM25 and e5 retrieval recall per hop (user 0, and with error bars over all 12 users), the leakage check, a scorer,
  a 9-question farm run, and on a Colab T4 the one-call grid (NR with no memory / Oracle / BM25 / e5) on 900 questions with
  Qwen2.5-7B-Instruct AWQ, with leak-free rescoring, and the 5-step chain-of-thought grid (SR) on 306 questions
  (`results/t4/summary_s2_vs_s1.md`).
- Partially reproduced on the laptop: small-model LoRA and HybridMem, with farm-generated training pairs; see
  `results/laptop/sft/summary.md`.
- Run, no support for the paper's top-k claims: top-k sweep k = 10/20/30/50 at 7B (`results/t4/summary_s3_topk_sweep.md`; answer EM is flat in k although retrieval recall rises). Not reproduced: reasoning-step sweep at 7B (scripted, 3b not run: no T4 available), MaskSFT, TreeRAG, GraphRAG,
  MR and DR reasoning, and any paper number (different model setup).

The farm run is tiny (n = 9, one question is 0.11): only the large gaps mean anything.

Known upstream problems (missing server, no scorer, HybridMem train-then-eval crash, SFT padding, prompt mismatch): `docs/upstream-notes.md`.

## Upstream license

The upstream repo https://github.com/nuster1128/MPR has no LICENSE file. Nothing from it is copied or vendored
here: `setup.sh` clones it at commit `a24953c` into the gitignored `upstream/` and applies `patches/farm.patch`
(our own diff). The dataset (Google Drive zip) is downloaded into the gitignored `data/` and never committed.

## Reference

Zhang, Zhang, Tan, Li, Chen. "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over
Personalized Information". KDD '26, pp. 1964-1974. DOI 10.1145/3770854.3780184. arXiv 2508.13250.
The analysis here used arXiv v1 (2508.13250v1); check the KDD camera-ready for changes before presenting.
