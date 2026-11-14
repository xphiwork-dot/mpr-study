# S6 — DecompositionReason (DR) on one Colab T4: final report

**Status: COMPLETE — 612/612 required DR rows were produced, every in-job gate passed, and an
independent local validator and scorer re-checked the downloaded rows.** This report replaces an earlier
version of the same file that recorded a blocker (no accelerator on the shared Kaggle account); the
blocker evidence is preserved in section 9 and the route that worked is section 2.

| Item | Value |
|---|---|
| Worktree / branch | `/home/xphi/Desktop/project/mpr-study-dr`, `experiments/dr-kaggle` |
| Code commit that produced the rows | `eeda20c49c71827bec6aafd89741c6a3126d5813` (`manifest_out.json`) |
| Method | DecompositionReason, faithful to upstream `a24953c` with one documented parser deviation |
| Model / serving | Qwen2.5-7B-Instruct-AWQ `b25037543e9394b818fdfca67ab2a00ecc7dd641`, vLLM 0.11.2, greedy, 8192 ctx |
| Retrieval | BM25 (SparseRAG) and e5-base-v2 (DenseRAG), k = 20, 306-question `s306` slice |
| Hardware | one Colab free-tier Tesla T4, 15,360 MiB, driver 580.82.07, 2 vCPU, 13.3 GB RAM |
| Successful DR rows | **612 / 612** (306 per memory, 0 failures, 0 invalid rows) |
| Kernel wall | **2,720 s = 45.3 min** (full 612-row evaluation 2,051 s of that) |
| Colab session | created 11:25:52Z, stopped 12:19:09Z (53.3 min); session cap 9,000 s, not reached |
| Kaggle GPU quota spent | **0.00 h** (the Kaggle route was not used; see section 9) |
| Local tests | `python3 -m unittest discover -s tests` → **57 tests, OK** (9 skip without numpy); the same 57 pass with `.venv/bin/python` |
| Independent validation | `results/kaggle/dr/validation_local.json`: `complete_612=true`, `ok=true` |

## 1. What was actually run

Upstream `reasonings/DecompositionReason.py` at `a24953c` (read-only audit and upstream-parity mocks in
`README.md` (b) and `tests/test_dr.py`): **Divide** (1 call, no retrieval, 5 sub-question limit),
**Solve** one sub-question at a time in order (1 retrieval + 1 call each, seeing earlier answers),
**Merge** (1 retrieval keyed on the answered state + 1 call). `decompose_num = 5`, so 3-7 calls per
question; observed mean **3.89** sub-questions, i.e. fewer than the 5-call cap.

Frozen settings (`settings.json`, sha256 `3ab140653fb77b4c318bf01da6687e1eca427027143f1d21de734d8add410f48`,
recorded in every row): greedy decoding, caps divide 200 / solve 200 / merge 64 tokens, top-k 20,
workers 40, parser `ref_faithful_drop_blank`, slice `results/t4/slices/s306.json` (306 questions, 34 per
hop 2-10, all 12 users, 140 leak-free by the any-number rule). No hyperparameter was tuned on this set.

The parser deviation never fired here: **0** fallback rows and **0** rows with dropped blank lines in both
memories, so all 612 rows are pure upstream division behaviour.

## 2. Execution route: authenticated Colab CLI, Tesla T4 (why not Kaggle)

The Kaggle kernel and dataset are packaged and tested, but four independent accelerator probes on the
shared Kaggle account (two kernel types, three request shapes) all ran CPU-only while the server stored
`enable_gpu: true` / `machine_shape: NvidiaTeslaT4` (section 9). The `colab` CLI on this machine is
authenticated and allocates a live **Tesla T4** in seconds, and the peer MR experiment already produced its
612 rows that way, so the **same committed `kernel.py`** was run on a Colab VM by
`experiments/s6_kaggle_dr/colab_run.sh`. Only the input staging differs: Colab mounts `/kaggle/input`
read-only, so the packaged dataset is unpacked at `/content/dr_inputs` and handed to the kernel through
`MPR_INPUT_ROOT` (with `MPR_WORK=/kaggle/working`); the kernel still verifies every sha256 and still
refuses a manifest that declares a different dataset.

Observed environment (committed evidence):

| Item | Value | Source |
|---|---|---|
| GPU | `GPU 0: Tesla T4`, 15,360 MiB, driver `580.82.07` | `preflight.json` |
| VM | 2 vCPU, 13.3 GB RAM, Python 3.13.16 | `preflight.json` |
| vLLM | **0.11.2** (pre-installed by the driver), healthy after 380.1 s, described model `qwen` | `server.json` |
| torch / CUDA | `2.9.0+cu128` / CUDA 12.8 | `deps.json` |
| transformers / sentence-transformers | 4.57.6 / 5.7.0 | `deps.json` |
| bm25s | 0.2.9 (installed in-job in 2.8 s) | `setup_pip.log` |
| Embeddings | e5-base-v2, 12 users, 47.2 s on the GPU, cached fp16 | `embeddings.json` |
| Upstream | `a24953c8f342c9f3b2064e562ead49d477d49138`, `prompts.py` sha256 `daf087ed…2405` verified | `upstream.json` |

The vLLM pin is deliberately the one that **actually served the MR rows** (MR's own deployment record shows
0.32.0 failing at server startup on this driver and 0.11.2 serving), so the DR and MR engines match. The
kernel's own driver-aware 0.32.0 → 0.11.2 fallback was therefore not exercised; `colab_run.sh` pre-installs
0.11.2 and the kernel then only adds `bm25s==0.2.9` and removes `torchao` (the historical T4 recipe).

## 3. Gates (all passed, observed before the 612-row run)

| Gate | Result | Evidence |
|---|---|---|
| Input hash verification | 19/19 files match `manifest.json`; resolved root `/content/dr_inputs` (`strategy=explicit`) | `input_probe.json`, `manifest_out.json` |
| Upstream pin | commit and `prompts.py` sha256 match the audited file; job would abort otherwise | `upstream.json` |
| Retrieval cross-check | 9/9 checks: BM25 and e5 top-20 identical to freshly built indexes (users 0/3/7); cached embedding cosines 0.9997-1.0003 | `retrieval_gate.json` |
| Divide gate | one real divide call returned 3 usable sub-questions | `divide_gate.json` |
| DR smoke (12 rows) | calls = 1+N+1, N non-empty sub-answers, every solve/merge retrieval top-20, prompts ≤ 8192 | `smoke_dr.json` |
| NR deployment sanity | **5/6** byte-identical NaiveReason responses per memory and **6/6** questions with 20/20 identical retrieved ids; pre-registered floor for a non-0.32.0 engine is 2/6 | `sanity_naive.json` |
| Runtime projection | 1,035 s estimated vs 7,131 s remaining → start | `runtime_estimate.json` |
| In-job validation | `complete_612=true`, 0 missing/extra/duplicate/failure/invalid per memory | `validation.json` |

The sanity gate compares against the archived S1 rows, which were produced with vLLM **0.32.0**; this run
served **0.11.2**, so `engine_matches_reference=false` and the 2/6 floor applied. One response per memory
differs at the token level (`2023-04-21` vs `2023-04-19` for `2:RC000159#02` on SparseRAG; `24` vs
`24 (days)` for `1:RC000150#05` on DenseRAG) while retrieval was 20/20 identical, which is the drift the
gate was designed to bound, not evidence of a different deployment.

## 4. Results (all numbers from the committed rows; 95% percentile bootstrap, 2,000 resamples)

Slice `s306.json`: 306 questions, 140 leak-free. `results/kaggle/dr/summary_dr.md` / `.json`.

| run | n | EM [95% CI] | contains [95% CI] |
|---|---|---|---|
| DR SparseRAG | 306 | **0.131 [0.095, 0.167]** | 0.131 [0.095, 0.167] |
| DR DenseRAG | 306 | **0.118 [0.082, 0.157]** | 0.121 [0.085, 0.160] |
| DR SparseRAG, leak-free | 140 | 0.143 [0.086, 0.207] | 0.143 [0.086, 0.207] |
| DR DenseRAG, leak-free | 140 | 0.129 [0.079, 0.186] | 0.136 [0.086, 0.193] |

| run | EM 2-4 (n=102) | EM 5-7 (n=102) | EM 8-10 (n=102) | calls | 3/4/5/6/7-call rows |
|---|---|---|---|---|---|
| DR SparseRAG | 0.216 | 0.069 | 0.108 | 1,802 | 4/41/80/41/140 |
| DR DenseRAG | 0.176 | 0.078 | 0.098 | 1,800 | 4/41/81/41/139 |

EM by hop 2→10, SparseRAG: 0.32, 0.21, 0.12, 0.15, 0.03, 0.03, 0.21, 0.12, 0.00.
DenseRAG: 0.35, 0.15, 0.03, 0.12, 0.03, 0.09, 0.15, 0.15, 0.00.
The two retrievers are within noise of each other; both fall to 0.00 at hop 10, and DR's `contains` equals
EM on SparseRAG (no lenient-only matches).

## 5. Paired differences against the archived NR and SR runs

Same questions on both sides, DR minus baseline (`results/t4/s1` = one-call NR, `results/t4/s2` = 5-step
SR), paired bootstrap over the shared questions. Full table (all subsets) in `summary_dr.md`.

| comparison | subset | n | dEM [95% CI] | dContains [95% CI] |
|---|---|---|---|---|
| DR SparseRAG − NR SparseRAG | all | 306 | +0.016 [−0.023, +0.056] | −0.003 [−0.042, +0.036] |
| DR DenseRAG − NR DenseRAG | all | 306 | −0.010 [−0.046, +0.026] | −0.033 [−0.072, +0.003] |
| DR SparseRAG − SR SparseRAG | all | 306 | −0.062 [−0.105, −0.016] | −0.069 [−0.111, −0.023] |
| DR DenseRAG − SR DenseRAG | all | 306 | −0.105 [−0.154, −0.056] | −0.118 [−0.167, −0.069] |
| DR SparseRAG − SR SparseRAG | leak-free | 140 | −0.050 [−0.107, +0.007] | −0.050 [−0.107, +0.007] |
| DR DenseRAG − SR DenseRAG | leak-free | 140 | −0.107 [−0.179, −0.036] | −0.129 [−0.200, −0.057] |

So on this slice DR is **statistically indistinguishable from one-call NR** (intervals include 0 for both
memories on all questions; DenseRAG's `contains` is −0.033 with an interval just touching 0) and **below
the 5-step SR** for both retrievers on all 306 questions. That is the opposite direction of the paper's
MR/DR gains, and it is a statement about this deployment (4-bit 7B, greedy, T4, our scorer), not about the
A800/full-precision paper setup.

## 6. Mutual MR-vs-DR comparison (both directions)

Both sides were read with the same rules (`experiments/_shared/score.py`, last row wins per `uid:rc_id`,
paired bootstrap seed 1, 2,000 resamples) by `compare_mr_dr.py`; no questions are missing on either side
(`n=306` everywhere). MR rows are the committed Colab export now at
`/home/xphi/Desktop/project/mpr-study/results/kaggle/mr/` (`StructuredReason_{SparseRAG,DenseRAG}_k20_s5.jsonl`).

| run | n | EM [95% CI] | contains [95% CI] |
|---|---|---|---|
| MR SparseRAG | 306 | 0.203 [0.160, 0.248] | 0.209 [0.167, 0.255] |
| MR DenseRAG | 306 | 0.216 [0.170, 0.261] | 0.225 [0.176, 0.271] |

| direction | memory | subset | n | dEM [95% CI] | dContains [95% CI] |
|---|---|---|---|---|---|
| DR − MR | SparseRAG | overall | 306 | **−0.072 [−0.118, −0.029]** | −0.078 [−0.124, −0.033] |
| DR − MR | DenseRAG | overall | 306 | **−0.098 [−0.147, −0.052]** | −0.105 [−0.154, −0.056] |
| DR − MR | SparseRAG | leak-free | 140 | −0.043 [−0.100, +0.007] | −0.050 [−0.107, +0.007] |
| DR − MR | DenseRAG | leak-free | 140 | −0.107 [−0.186, −0.029] | −0.114 [−0.193, −0.036] |
| DR − MR | SparseRAG | 2-4 / 5-7 / 8-10 | 102 each | −0.059 / −0.078 / −0.078 | −0.069 / −0.088 / −0.078 |
| DR − MR | DenseRAG | 2-4 / 5-7 / 8-10 | 102 each | −0.137 / −0.059 / −0.098 | −0.147 / −0.069 / −0.098 |

Artifacts: `results/kaggle/dr/summary_mr_vs_dr.md` / `.json` (DR minus MR) and
`results/kaggle/dr/summary_mr_minus_dr.md` / `.json` (the same tool run with the sides swapped).

Mutuality was checked, not assumed:

1. **MR side reproduces MR's own analysis.** The EM/contains and intervals computed here from MR's raw rows
   are identical to MR's committed `summary_mr.md` (0.203 [0.160, 0.248] / 0.216 [0.170, 0.261]), so the two
   experiments are compared by one scorer, not by two different reports.
2. **Reverse direction mirrors exactly.** Re-running the tool with A=MR and B=DR gives `n` unchanged,
   dEM/dContains sign-flipped and the interval reflected (`[-hi, -lo]`) for all 10 paired rows; checked
   programmatically over `summary_mr_vs_dr.json` vs `summary_mr_minus_dr.json`.
3. **Comparability conditions.** Same model revision, same slice, same retrievers, k=20, same T4 class, same
   vLLM/torch pin (0.11.2 / 2.9.0+cu128), both runs gold-standard-checked against the released data.
   The one real difference is decoding: DR is fully greedy, MR samples its two branch candidates per middle
   round (T=0.7, top_p=1.0) and is greedy elsewhere. MR-vs-DR is therefore **not decoding-matched**, and DR
   also uses far fewer sampled tokens (62k completion vs MR's 235k).

## 7. Calls, tokens and wall time (observed)

| item | SparseRAG | DenseRAG | total |
|---|---|---|---|
| rows | 306 | 306 | 612 |
| calls | 1,802 | 1,800 | 3,602 |
| prompt tokens | 1,097,511 | 1,136,907 | 2,234,418 |
| completion tokens | 30,995 | 31,019 | 62,014 |
| summed request time | 37,865 s | 39,349 s | 77,214 s (at 40 concurrent questions) |
| full-run wall | 1,006.8 s | 1,043.8 s | 2,050.6 s (34.2 min) |
| engine throughput | 1.79 calls/s | 1.72 calls/s | — |

Setup before the full run: 668.7 s (30.7 s deps, 47.2 s embeddings, 380.1 s vLLM start, divide/smoke/NR
sanity gates). Kernel total 2,720.1 s; session creation→stop 3,197 s including the 6-minute vLLM pre-install
and the output download. The 9,000 s cap and the 8,400 s watchdog were never approached.

## 8. Artifacts

Curated evidence (committed) under `results/kaggle/dr/`:

- `runs/DecompositionReason_{SparseRAG,DenseRAG}_k20.jsonl` — 306 rows each, with the full call trace,
  prompts' sha256, sub-questions/answers, solve and merge retrieval ids, parser metadata and token counts;
  `runs/failures_*.jsonl` — empty (0 failures).
- `status.json`, `DONE`, `manifest_out.json` (software versions, 19 verified input hashes, settings),
  `summary_dr.md`/`.json`, `summary_mr_vs_dr.md`/`.json`, `summary_mr_minus_dr.md`/`.json`,
  `validation.json`, `validation_local.json` (independent re-check), `score_dr.json`.
- Gate evidence: `preflight.json`, `input_probe.json`, `upstream.json`, `deps.json`, `embeddings.json`,
  `retrieval_gate.json`, `divide_gate.json`, `smoke_dr.json`, `sanity_naive.json`/`.jsonl`,
  `runtime_estimate.json`, `server.json`, `run_summary.json`, `run.log`, `dr_kernel.log`,
  `setup_pip.log`, `setup_upstream.log`.
- `raw/` (gitignored) holds the unedited Colab download bundle, including the full `vllm.log`.
- Historical blocker evidence, preserved unedited: `attempt1_pi_setup_failure/`, `mount_probe/`,
  `gpu_availability/`, and `reference_nr_vs_sr.*` (the scorer sanity check on archived rows).

No credentials, tokens, embedding caches or model weights are committed. `results/kaggle/dr/raw/` and
`results/kaggle/dr/vllm.log` are gitignored; dataset and sparse model files never leave the VM.

## 9. History: the blocked Kaggle phase (why this run is on Colab)

**Attempt 1** (`phinguynxun/mpr-dr-reasoning` v1, Pi-era worker, 2026-11-12 17:58 UTC) died after 0.0 s with
`FileNotFoundError: manifest.json not found under /kaggle/input/mpr-dr-inputs`. A private CPU probe kernel
proved the dataset was mounted at `/kaggle/input/datasets/phinguynxun/mpr-dr-inputs/` instead
(`mount_probe/mountprobe.json`, all 20 files, expected sizes, manifest sha256). `kernel.py:
discover_input_root()` now tries explicit candidates, walks `/kaggle/input` and retries for 90 s, and
`tests/test_kernel_stage.py` covers both layouts and the foreign-manifest rejection.

**Attempt 2 was stopped on a hard blocker**: four independent accelerator probes (script and notebook,
`enable_gpu` only, `machine_shape`, `--accelerator NvidiaTeslaT4`, latest image pin) all ran CPU-only —
no `/dev/nvidia*`, `torch 2.11.0+cpu`, `GPU used 0.00h`, `has_ever_run: false` — while the saved kernel
metadata carried the GPU request. The account therefore provisions no accelerator to this job even with an
untouched 30 h/week GPU allowance, and a CPU-only vLLM run is not the frozen T4 deployment the NR/SR
baselines were measured on. Details and raw probe output: `gpu_availability/` and the earlier revision of
this report in git history. Nothing was pushed, so no Kaggle GPU quota was spent (`0.00 h`).

The same committed code was then run on the authenticated Colab T4 route (section 2). The private Kaggle
dataset/kernel IDs in `settings.json` and `package.py` move to the `xunphi/mpr-dr-{inputs,reasoning}` IDs used
by the MR experiment's committed packaging, so a future Kaggle push stays consistent; they are only IDs and
nothing was uploaded for this run.

## 10. Caveats

- One slice (306 questions), one seed, one 4-bit 7B model on one T4: exploratory, not a reproduction of the
  paper's A800/full-precision tables. Absolute numbers are only comparable with our own runs.
- Engines differ from the archived NR/SR rows (vLLM 0.11.2 / torch 2.9.0+cu128 here, vLLM 0.32.0 there).
  The NR sanity gate bounds the drift at 5/6 byte-identical responses and 6/6 identical retrievals, but
  token-level decoding differences cannot be excluded.
- MR samples branch candidates; DR is fully greedy. The MR-vs-DR table compares complete methods under
  different decoding, not equal compute (DR: 3,602 calls / 62k completion tokens; MR: 7,272 candidate
  calls / 235k completion tokens).
- DR's calls per question are data-dependent (3-7; mean 5.89, with 8 of 612 rows producing only 3 calls), so the
  method's compute is not fixed across questions; hop groups are not matched on call count.
- Intervals are percentile bootstraps over questions; no multiple-comparison correction is applied to the
  subset tables (10 paired rows per comparison, 5 subsets).
- The Colab free tier can reclaim sessions; this one completed without a deadline hit, and every row is
  flushed/fsynced so a cut session would resume rather than duplicate.

## 11. How to reproduce

```bash
# 0. prerequisites: pinned upstream clone, released dataset, gitignored data (see experiments/_shared/README.md)
git -C upstream/MPR rev-parse HEAD        # a24953c8f342c9f3b2064e562ead49d477d49138
ls data/mpr_dataset/question_answer/question_answer_0.json

# 1. local checks (stdlib; the parity tests execute the upstream clone read-only)
python3 -m unittest discover -s tests -v

# 2. one locked Colab T4 job: builds the dataset from a CLEAN tree, stages it at /content/dr_inputs,
#    pre-installs vllm 0.11.2, runs the unchanged kernel.py, pulls /kaggle/working into
#    results/kaggle/dr/raw and stops the session (DR_COLAB_KEEP=1 keeps it for debugging)
experiments/s6_kaggle_dr/colab_run.sh

# 3. curate the downloaded evidence and re-check it independently
cp results/kaggle/dr/raw/results/kaggle/dr/runs/*.jsonl results/kaggle/dr/runs/
python3 experiments/s6_kaggle_dr/validate_dr.py --runs results/kaggle/dr/runs
uv run --with numpy python3 experiments/s6_kaggle_dr/summarize_dr.py \
  --dr results/kaggle/dr/runs --slice results/t4/slices/s306.json \
  --nr results/t4/s1 --sr results/t4/s2 --out results/kaggle/dr/summary_dr.md

# 4. mutual comparison with MR (both directions; same scorer and paired conventions)
uv run --with numpy python3 experiments/s6_kaggle_dr/compare_mr_dr.py \
  --a-dir results/kaggle/dr/runs --a-label DR \
  --b-dir /home/xphi/Desktop/project/mpr-study/results/kaggle/mr \
  --b-pattern 'StructuredReason_{mem}_k20_s5.jsonl' --b-label MR \
  --slice results/t4/slices/s306.json --out results/kaggle/dr/summary_mr_vs_dr.md
# then swap --a-* and --b-* and write summary_mr_minus_dr.md
```

The kernel enforces the same gates before any evaluation row is written (hash verification, upstream pin,
retrieval cross-check, divide gate, 12-row smoke, NR deployment sanity, closed-loop runtime projection) and
stops submitting 1,200 s before the session cap. The Kaggle path (`kaggle_run.sh`, `package.py`) remains
packaged and tested but is blocked on the account, not on the code (section 9).

## 12. Commits on `experiments/dr-kaggle`

- Implementation, kernel, packaging, tests and Kaggle driver (Pi-era, preserved): `56c096a`, `520bd22`,
  `3bb161a`, `0975d3d`.
- Kaggle-phase hardening and evidence (preserved): `1a76ce4` mount discovery, `e78b59d` driver-aware vLLM
  pin, `25eb76c` blocker/quota record, `cb56b39` runtime-projection tests, `739b4c9` validator/summarizer
  tests, `55dc7e0` engine-aware sanity floor, `47d88fd` proven-vs-cause split, `5e8e0bb` DR-vs-MR tool,
  `a427c09` serve-time pin fallback + UI-session kernel, `9f96e27`/`ee9feb7`/`2ee5a1c`/`b33568b`/`38bd4b7`
  docs and audits.
- `eeda20c` — Colab T4 route (`colab_run.sh`, `xunphi` resource IDs); **this is the commit the run manifest
  records**.
- `e81635d` — curated results and comparison artifacts under `results/kaggle/dr/`.
- The commit carrying this report and the README/docs updates.
