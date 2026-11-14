# S5 MR (StructuredReason) - report

Status: **COMPLETE** (2026-11-14). All **612/612** expected rows were produced
and validated: 306 unique questions x {BM25 (`SparseRAG`), e5 (`DenseRAG`)}.
The full evaluation ran on a **Colab free-tier Tesla T4** session through
`colab_mr.sh`; Kaggle GPU allocation stayed queued and was abandoned (section
1). Raw rows, run manifest and the generated score table are committed under
`results/kaggle/mr/`.

Headline (same frozen 306-question slice, fixed seed 0):

| run | n | EM [95% CI] | contains [95% CI] |
|---|---|---|---|
| MR SparseRAG | 306 | **0.203** [0.160, 0.248] | 0.209 [0.167, 0.255] |
| NR SparseRAG (s1) | 306 | 0.114 [0.082, 0.150] | 0.134 [0.098, 0.173] |
| SR SparseRAG (s2) | 306 | 0.193 [0.150, 0.242] | 0.199 [0.154, 0.248] |
| MR DenseRAG | 306 | **0.216** [0.170, 0.261] | 0.225 [0.176, 0.271] |
| NR DenseRAG (s1) | 306 | 0.127 [0.092, 0.167] | 0.154 [0.114, 0.193] |
| SR DenseRAG (s2) | 306 | 0.222 [0.173, 0.271] | 0.239 [0.190, 0.288] |

Paired on the same questions, MR minus NR is **+0.088 EM** for both retrievers
(BM25 [+0.039, +0.137], e5 [+0.046, +0.131]; intervals exclude 0), while MR
minus SR is **not detectable** (BM25 +0.010 [-0.026, +0.049], e5 -0.007
[-0.046, +0.033]). MR matches 5-step SR at 12 calls/question versus 5, i.e.
about 2.4x the model calls (75.2 min of generation versus 39.8 min for the
s2 SR grid). The full score table,
subsets and all paired intervals are in
[`../../results/kaggle/mr/summary_mr.md`](../../results/kaggle/mr/summary_mr.md).

## 1. Execution path and what actually ran

| Stage | Result |
|---|---|
| Local tests | `python3 -m unittest tests.test_mr -v` -> 21/21 pass (2026-11-14) |
| Private Kaggle dataset | `xunphi/mpr-mr-inputs` v1, 25 allowlisted flat files, 43,689,643 bytes, all hashes verified; private, no secrets, no upstream source/templates |
| Kaggle kernel | `xunphi/mpr-mr-reasoning` v1 pushed 08:08Z, cancelled by the user while `QUEUED`; v2 pushed 08:20Z with the verified `NvidiaH100` + AIMO competition-source route (`--timeout 14400`) and stayed `QUEUED` for >60 min. Per the orchestrator, the local wrapper was stopped and the job abandoned. `results/kaggle/mr/_kaggle_out` is empty; the kernel is no longer retrievable through the API. Kaggle GPU quota shows **0.20 h used** total (0.01 h before this attempt), with no experiment output. |
| Final run | Colab free-tier session `mpr-mr`, **1x Tesla T4 15,360 MiB, driver 580.82.07**, via `experiments/s5_kaggle_mr/colab_mr.sh` (`MR_COLAB_KEEP=1`, held the shared GPU lock). Kernel launched 09:30:44Z, complete 11:07:26Z: **5,802.5 s (~97 min)** end to end. |
| Inputs of record | flat staged dataset built at commit `44f1e3a`, `git_dirty=false`, 25 files; upstream `a24953c8f342c9f3b2064e562ead49d477d49138`; model `Qwen/Qwen2.5-7B-Instruct-AWQ` rev `b25037543e9394b818fdfca67ab2a00ecc7dd641`; e5 rev `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. |
| vLLM | planned pin `0.32.0` exited rc=2 during startup, so the kernel's one bounded fallback installed **`vllm==0.11.2`** (torch `2.9.0+cu128`, CUDA 12.8, transformers `4.57.6`), which served `qwen` healthily after 350 s. `manifest.json` records `model.vllm=0.11.2`; its `environment.packages` block was captured before the fallback and is stale (it lists 0.32.0/torch 2.13.0). Details in [`deployment.json`](../../results/kaggle/mr/deployment.json). |
| Dataset mount | Colab mounts `/kaggle/input` read-only, so the first launch (09:27Z) exited 2 when it tried to exec from a dataset path there. Fixed by launching `/content/mr_inputs/kernel.py` and passing `MPR_INPUT_ROOT` (now asserted in VM prep); the relaunch is the one above. |
| Staging/pull | `colab_mr.sh` now downloads `results/kaggle/mr` flat plus the kernel/vLLM logs, the embeddings manifest and the packaged tarball. |

The Kaggle private resources are still private and were not used for any
result: <https://www.kaggle.com/datasets/xunphi/mpr-mr-inputs> (v1) and the
aborted kernel <https://www.kaggle.com/code/xunphi/mpr-mr-reasoning>.

## 2. Pre-registered gates (all passed)

- **NR deployment sanity** (`sanity_report.json`): 3 fixed smoke questions x 2
  retrievers, greedy one-call NR vs the committed s1 rows. No empty answers;
  **6/6 exact normalized response matches**; EM drift 0 vs baseline. That is
  direct evidence the deployed model/retrieval stack reproduces the baseline
  deployment, despite the vLLM fallback.
- **MR smoke** (`smoke_report.json`): 6 rows = 3 questions x 2 retrievers,
  exactly 12 calls each (72 calls), 24 selections, 0 selection fallbacks, 3/18
  exact duplicate branch pairs, no empty non-selection response and no empty
  retrieval.
- **Full coverage** (`coverage_report.json`): 306/306 unique successful rows per
  retriever, no missing/extra keys, 12 calls per row, ground truth equal to the
  released QA answer, hop equal to the slice, 0 failure records, `full_run_rc=0`,
  `full_complete=true`, `status.json` status `complete`.
- **Call count**: 7,272 full-stage calls + 72 smoke calls = **7,344**, exactly
  the pre-registered `306 x 2 x 12`; no retries were needed.

## 3. Results on the frozen 306-question slice

Slice: `results/t4/slices/s306.json`, seed 0, 34 questions per hop (2-10), all
12 users, 140 leak-free (`leak_any`), 180 under the 3+ digit rule. Metrics are
the existing `experiments/_shared/score.py` EM and lenient contains, with 95%
percentile bootstrap over questions (2,000 resamples, fixed seeds); paired
differences use the same question keys.

### By retriever and subset (MR)

| retriever | subset | n | EM [95% CI] | contains [95% CI] |
|---|---|---|---|---|
| SparseRAG | all | 306 | 0.203 [0.160, 0.248] | 0.209 [0.167, 0.255] |
| SparseRAG | leak-free | 140 | 0.186 [0.121, 0.257] | 0.193 [0.129, 0.264] |
| SparseRAG | hops 2-4 | 102 | 0.275 [0.196, 0.363] | 0.284 [0.206, 0.373] |
| SparseRAG | hops 5-7 | 102 | 0.147 [0.078, 0.225] | 0.157 [0.088, 0.235] |
| SparseRAG | hops 8-10 | 102 | 0.186 [0.118, 0.265] | 0.186 [0.118, 0.265] |
| DenseRAG | all | 306 | 0.216 [0.170, 0.261] | 0.225 [0.176, 0.271] |
| DenseRAG | leak-free | 140 | 0.236 [0.164, 0.307] | 0.250 [0.179, 0.321] |
| DenseRAG | hops 2-4 | 102 | 0.314 [0.225, 0.412] | 0.333 [0.245, 0.431] |
| DenseRAG | hops 5-7 | 102 | 0.137 [0.078, 0.206] | 0.147 [0.078, 0.225] |
| DenseRAG | hops 8-10 | 102 | 0.196 [0.118, 0.275] | 0.196 [0.118, 0.275] |

### Paired differences vs the historical baselines

| retriever | baseline | subset | n | dEM [95% CI] | dContains [95% CI] |
|---|---|---|---|---|---|
| SparseRAG | NR | all | 306 | +0.088 [+0.039, +0.137] | +0.075 [+0.029, +0.121] |
| SparseRAG | NR | leak-free | 140 | +0.050 [-0.007, +0.114] | +0.043 [-0.021, +0.107] |
| SparseRAG | SR | all | 306 | +0.010 [-0.026, +0.049] | +0.010 [-0.026, +0.049] |
| SparseRAG | SR | leak-free | 140 | -0.007 [-0.057, +0.043] | +0.000 [-0.050, +0.050] |
| DenseRAG | NR | all | 306 | +0.088 [+0.046, +0.131] | +0.072 [+0.029, +0.114] |
| DenseRAG | NR | leak-free | 140 | +0.093 [+0.029, +0.157] | +0.079 [+0.007, +0.150] |
| DenseRAG | SR | all | 306 | -0.007 [-0.046, +0.033] | -0.013 [-0.052, +0.026] |
| DenseRAG | SR | leak-free | 140 | +0.000 [-0.057, +0.057] | -0.014 [-0.071, +0.043] |

BM25 minus e5 on the same questions: EM -0.013 [-0.059, +0.036] overall and
-0.050 [-0.114, +0.014] leak-free; no subset interval excludes 0 (table in
`summary_mr.md`).

### What this does and does not show

- MR beats one-call NR on EM for both retrievers overall; on the leak-free
  subset only the e5 difference is clearly nonzero (+0.093), the BM25 one is
  not (+0.050 with an interval that includes 0).
- MR is **not** distinguishable from 5-step SR here; every MR-minus-SR interval
  includes 0. MR reached the same EM with 12 calls/question instead of 5 and
  2.4x the generation wall time. This is an efficiency/comparability statement,
  not evidence that MR's tree is better.
- MR branch calls are sampled while the historical NR/SR rows were greedy, so
  decoding is **not matched**; no claim is made that branch structure alone
  produced the difference. Absolute numbers are also not the paper's
  A800/full-precision results (4-bit AWQ on one T4, our scorer).

## 4. Trace diagnostics (observed, not assumed)

| diagnostic | SparseRAG | DenseRAG |
|---|---|---|
| rows | 306 | 306 |
| model calls | 3,672 | 3,672 |
| prompt / completion tokens | 2,555,270 / 120,011 | 2,599,385 / 118,167 |
| selection fallbacks | 3 / 1,224 selections (0.25%) | 4 / 1,224 (0.33%) |
| exact duplicate branch pairs | 296 / 918 (32.2%) | 297 / 918 (32.4%) |
| thought calls hitting the 200-token cap | 9 | 8 |
| final selected path depth | 1:122, 2:107, 3:71, 4:6 | 1:106, 2:121, 3:74, 4:5 |
| mean seconds per question (incl. 12 calls) | 263.7 | 258.6 |

The depth distribution is the reference behaviour, not a bug: the buffer update
keeps every old path (including the selected one) and appends two children, so
a shallow path can be selected again. In most questions the final answer uses a
depth-1 or depth-2 path even though the configuration is named `max_step=5`
(the final answer is not appended to the path). Depths 1-4 are all legal and
recorded per row in `trace.selected_path_depths`.

## 5. Cost and timing

- Colab free-tier T4 session: 09:30:44Z-11:07:26Z = **5,802.5 s (96.7 min)**
  including package install, model download, e5 cache, vLLM startup and gates.
- Full generation stage: **7,272 calls in 75.2 min** of job wall time
  (SparseRAG 2,238 s, DenseRAG 2,276 s, 64 workers), plus 6.2 min of smoke.
- Token volume: 5,154,655 prompt tokens and 238,178 completion tokens for the
  full stage. No failed calls, no retried rows, no deadline hit.
- Kaggle: **0.20 h** total GPU quota used by earlier probes/the aborted attempt
  (0.01 h before this work); the aborted job produced no output. Colab free tier
  was used for the evaluation, so the Kaggle quota is essentially untouched.
- Wall-clock cost of the whole worker session (audit, implementation, Kaggle
  attempts, Colab run, analysis) was OMP wall time, not billable GPU time.

## 6. Reference audit and deliberate deviations

The full audit with line references and the frozen settings is in
[`README.md`](README.md) sections (a)-(d). In short: upstream
`reasonings/StructuredReason.py` at `a24953c`, `config.py:49-53`
(`max_step=5`, `branch_size=2`), append-only buffer at line 45, selection from
the whole buffer, retrieval key = last thought, selection has no retrieval.
Deliberate/documented differences: seeded sampling (T=0.7, top_p=1.0) only for
the two `Structured_Middle` candidates, greedy start/selections/final answer;
safe ASCII-digit selection parser instead of `eval`; bounded empty-response
retries; 200/64 token caps; per-question and global deadlines; no gold answer
ever passed to the model. Local stdlib tests cover the upstream call-sequence
parity with a mock model, the buffer semantics, parsing, seeds, duplicate
measurement, no-gold-leak, cut-tail resume and the reused NR/SR helpers.

## 7. Artifacts, hashes and commits

Run artifacts (committed, `results/kaggle/mr/`):

| file | SHA-256 |
|---|---|
| `StructuredReason_SparseRAG_k20_s5.jsonl` | `7c69703a5bb24ffb9a56a30a64320788f7216dbaa4ba17400744e615e3e097ba` |
| `StructuredReason_DenseRAG_k20_s5.jsonl` | `52cf140471972b7d5b64313b6a472b6db9d3b1e2fb8df848597ff418a0235002` |
| `timing.jsonl` | `11c68157ade418197f541396d5f0945b12a153afe51333e0e2cbec6d236919d0` |
| `status.json` | `bafce92d1113aad7982e3f3b588cec3c3594bf997ea208d2f1dae8f4a23abbc7` |
| `manifest.json` | `4e8299295f876d673d6fd29ba0073f2ddf3dd8ea816d6b5eb29a71d3f19ffd80` |
| `summary_mr.md` (generated) | `58729260dea40dc39a3cc9b8719452b2eda4f7b9a239b435292eacc70a390990` |
| `deployment.json` (derived from logs) | this commit |

The run was produced from dataset staging `git_commit=44f1e3a`,
`git_dirty=false`, 25 files, 43,689,643 bytes. `manifest.json` also records the
upstream/model/e5 revisions, prompt-template hashes (no template text), the e5
embedding manifest (12 users, 157,166 statements), inputs hashes, gates,
environment and per-artifact hashes. The kernel/vLLM logs are downloaded but
gitignored (`results/**/*.log`); the facts needed for provenance are transcribed
in `deployment.json`.

Commits on `experiments/mr-kaggle` for this end-to-end attempt:

| commit | content |
|---|---|
| `1fe4f25` | xunphi resource ids, verified `NvidiaH100` + AIMO routing metadata, dataset-ready wait, README recipe |
| `ea90e7b` | Colab driver: PATH export, row-count polling, flat results/log/manifest pull |
| `44f1e3a` | Colab launch from `/content/mr_inputs` (read-only `/kaggle/input` fix); this is the commit the run's input manifest records |
| (this commit) | 612 raw rows, run manifest/status/gates, `deployment.json`, `summary_mr.md`, report/README/reproduction-status updates |

## 8. Reproduce

```bash
# local, no GPU: 21 stdlib tests
python3 -m unittest tests.test_mr -v

# stage the allowlisted private dataset (flat; 25 files)
python3 experiments/s5_kaggle_mr/pack_mr.py --out /tmp/mpr-mr-kaggle/dataset --clean

# one Colab T4 session (holds the shared GPU lock for the whole job,
# resumes from downloaded jsonl rows if the session is preempted)
MR_COLAB_KEEP=1 experiments/s5_kaggle_mr/colab_mr.sh results/kaggle/mr/_colab_out

# score and compare against NR (s1) and SR (s2)
.venv/bin/python experiments/s5_kaggle_mr/analyze_mr.py \
  --results-dir results/kaggle/mr --slice results/t4/slices/s306.json \
  --data-dir ../mpr-study/data/mpr_dataset/question_answer
```

The Kaggle route is still documented (`run_remote.sh`, `kernel-metadata.json`)
for when the account can allocate a GPU; it is not required for these results.

## 9. Caveats and blockers

- **Decoding not matched**: branch-generation calls sampled (T=0.7) vs greedy
  NR/SR; any difference mixes structure with decoding. One slice, one seed,
  exploratory, no multiple-comparison correction.
- **Serving stack differs from the original plan**: vLLM 0.11.2/torch 2.9.0+cu128
  after the 0.32.0 attempt failed to start; e5 statement embeddings were
  prepared before the fallback (torch 2.13/transformers 5.19) and queries were
  encoded after it. The model and e5 revisions are pinned and the NR sanity
  gate matched the committed baseline exactly (6/6), so the deployment is
  healthy; a bit-identical stack is not claimed.
- **T4 4-bit AWQ**, not A800/full precision; our own scorer; absolute numbers
  are not comparable to the paper's tables.
- **Kaggle route blocked/abandoned**: the accelerator request stayed `QUEUED`
  for >60 min and the aborted job left no output. The experiment therefore
  reports Colab T4 execution, not Kaggle. The Kaggle dataset/kernel remain
  private; no Kaggle result is claimed.
- The `max_step=5` name overstates actual depth: retained shallow paths are
  frequently selected (observed denominators above), matching upstream.
- If the peer's DR run is available, the MR-vs-DR comparison should use the
  exported `results/kaggle/mr/summary_mr.md`/rows here; this worktree did not
  modify or depend on the DR worktree.
