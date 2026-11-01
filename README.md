# mpr-study

Private study repo for the paper "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over
Personalized Information" (MPR, KDD '26). It holds a study guide with diagrams for a Data Mining final
presentation, plus a small, cheap reproduction of parts of the paper: retrieval behaviour of explicit memory, a dataset
leakage check, a tiny run of the authors' baselines on the Bosch LLM Farm, Colab T4 runs of Qwen2.5-7B-Instruct (AWQ),
and a toy LoRA / HybridMem run on a laptop. The goal is to understand the paper's ideas, not to match its tables.

## Start here

1. Read the paper (arXiv v1): `papers/2508.13250v1.pdf`.
2. Open `docs/index.html` (the study guide: paper in 60 seconds, glossary, 4 flow charts, quizzes, talk plan).
3. Pick an experiment below and read its folder README: the paper experiment it relates to, what we ran, a code walkthrough, a provenance table, how to run, the outcome.
4. Read `docs/reproduction-status.md` (what is reproduced, with evidence) and skim `docs/upstream-notes.md` (upstream gaps and bugs).

## Experiments

Provenance legend: **BORROWED** = identical to the original (nuster1128/MPR at `a24953c`, cited with lines), **ADAPTED** = derived from it, **ORIGINAL TO THIS REPO** = no upstream reference.
No upstream code is copied here (the repo has no licence); every script in this repo is ADAPTED or ORIGINAL (details: [`experiments/README.md`](experiments/README.md)).

| Folder | What | Paper counterpart | Status |
|---|---|---|---|
| [`experiments/00_probes_retrieval_leak/`](experiments/00_probes_retrieval_leak/) | no-LLM retrieval recall per hop, number-leakage check | our addition (Sections 3.3.1, 4.2, 4.3) | run |
| [`experiments/01_farm_small_run/`](experiments/01_farm_small_run/) | 9-question run of the authors' `run_qa.py` on the LLM Farm | Section 4.2, Figure 3 | run, tiny |
| [`experiments/s1_t4_one_call_grid/`](experiments/s1_t4_one_call_grid/) | T4 session 1: throughput, 12-user retrieval error bars, one-call grid (900 q), leak-free rescoring | Section 4.2, Figure 3 (NR) | run |
| [`experiments/s2_t4_chain_of_thought/`](experiments/s2_t4_chain_of_thought/) | T4 session 2: 5-step chain-of-thought grid (306 q), SR vs NR | Section 4.2, Figure 3 (SR) | run |
| [`experiments/s3a_t4_topk_sweep/`](experiments/s3a_t4_topk_sweep/) | T4 session 3a: top-k sweep 10 / 20 / 30 / 50 (900 q) | Section 4.3, Figure 4 | run |
| [`experiments/s3b_t4_step_sweep_not_run/`](experiments/s3b_t4_step_sweep_not_run/) | T4 session 3b: reasoning-step sweep | Section 4.4, Figure 5 | NOT RUN (no T4 available); scripts and notebook ready |
| [`experiments/s4_laptop_lora_hybridmem/`](experiments/s4_laptop_lora_hybridmem/) | laptop session 4: small LoRA (AskSFT) and HybridMem | Sections 5 and 6, Table 3 | run, small |
| [`experiments/_shared/`](experiments/_shared/) | helpers, `colab` driver `t4.sh`, T4 runbook | n/a | n/a |

Each T4 folder has a Colab notebook (`*.ipynb`, no outputs, committed unexecuted; they have not been executed as notebooks).

## Layout

```
README.md  setup.sh  requirements.txt  env.example      setup.sh: venv (uv) + pinned upstream clone + farm patch + dataset + subset
experiments/            one folder per experiment (scripts, notebook, README); _shared/ holds lib, grid, summarize, score, t4.sh
  01_farm_small_run/farm.patch    our diff for upstream utils.py / config.py (LLM Farm client, env-driven paths)
results/                committed evidence, unchanged: probes/ farm/ t4/ laptop/ (raw jsonl, summaries, timing log)
docs/                   reproduction-status.md, upstream-notes.md, index.html (study guide), scout-report.html
papers/                 the paper PDF
upstream/  data/  results/logs/   gitignored: fetched clone, dataset, raw farm logs
```

## Run commands

```bash
./setup.sh                                                   # needs uv and git; TORCH_INDEX=<url> for a GPU wheel
.venv/bin/python experiments/00_probes_retrieval_leak/retrieval_recall.py     # BM25 ~2 s; e5 ~40 s on a GPU (+ model download)
.venv/bin/python experiments/00_probes_retrieval_leak/leak_check.py --show-example
# spends farm tokens (about 94k for the committed runs): needs LLM_FARM_ENDPOINT, LLM_FARM_KEY, SMALL_MODEL_NAME (see env.example)
cp env.example .env
for m in Ignoramus Oracle SparseRAG DenseRAG; do experiments/01_farm_small_run/farm_run.sh Naive $m; done
.venv/bin/python experiments/_shared/score.py results/farm/*/*.jsonl
# Colab T4 via the colab CLI (details: experiments/_shared/README.md); always finish with t4.sh down
experiments/s1_t4_one_call_grid/session1.sh                  # also session2.sh (s2), session3a.sh (s3a), session3b.sh (s3b, not run)
experiments/_shared/t4.sh down
python3 experiments/_shared/summarize.py grid --tag s1 --slice results/t4/slices/s900.json      # regenerate a summary from raw results
# laptop LoRA / HybridMem (CUDA GPU, farm credentials): see experiments/s4_laptop_lora_hybridmem/README.md
```

Or open a notebook in Colab (Runtime > T4 GPU): `experiments/s1_t4_one_call_grid/s1_t4_one_call_grid.ipynb`, `s2_...`, `s3a_...`, `s3b_...`. The repo is private: add a Colab secret `GITHUB_TOKEN` (read-only) for the clone cell.
Behind a corporate proxy set `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`.

## What is reproduced, and what is not

Full tables (status, evidence file, reason for every gap): [`docs/reproduction-status.md`](docs/reproduction-status.md). In short:

- Reproduced: BM25 and e5 retrieval recall per hop (user 0, and with error bars over all 12 users), the leakage check, a scorer, a 9-question farm run, and on a Colab T4 the one-call grid (no memory / Oracle / BM25 / e5)
  on 900 questions and the 5-step chain-of-thought grid on 306 questions, both with Qwen2.5-7B-Instruct AWQ and leak-free rescoring (`results/t4/summary_s1.md`, `summary_s2_vs_s1.md`).
- Partially reproduced on the laptop: small-model LoRA and HybridMem (`results/laptop/sft/summary.md`).
- Run, no support for the paper's top-k claims: top-k sweep k = 10 / 20 / 30 / 50 at 7B (`results/t4/summary_s3_topk_sweep.md`; answer EM is flat in k although retrieval recall rises).
- Not run: reasoning-step sweep (3b, no T4 available). Not reproduced: MaskSFT, TreeRAG, GraphRAG, MR and DR reasoning, and any paper number (different model setup).

The farm run is tiny (n = 9, one question is 0.11): only the large gaps mean anything.

## Upstream license

The upstream repo https://github.com/nuster1128/MPR has no LICENSE file. Nothing from it is copied or vendored
here: `setup.sh` clones it at commit `a24953c` into the gitignored `upstream/` and applies `experiments/01_farm_small_run/farm.patch`
(our own diff; it shows a few upstream lines as diff context). The dataset (Google Drive zip) is downloaded into the gitignored `data/` and never committed.

## Reference

Zhang, Zhang, Tan, Li, Chen. "Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over
Personalized Information". KDD '26, pp. 1964-1974. DOI 10.1145/3770854.3780184. arXiv 2508.13250.
The analysis here used arXiv v1 (2508.13250v1); check the KDD camera-ready for changes before presenting.
