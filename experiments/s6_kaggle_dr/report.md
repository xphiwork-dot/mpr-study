# S6 — DecompositionReason (DR) Kaggle GPU run: final status report

**Status: BLOCKED — no accelerator is provisioned on the shared Kaggle account.**
0 of the 612 required DR rows exist. No EM/contains number, no NR/SR comparison and no MR-vs-DR
comparison is reported for DR, because none was measured. Nothing in this report is inferred from
implementation: every number below was observed.

| Item | Value |
|---|---|
| Worktree / branch | `/home/xphi/Desktop/project/mpr-study-dr`, `experiments/dr-kaggle` |
| Head at report time | see `git log -1` (this file is committed with the code it describes) |
| Method | DecompositionReason, faithful to upstream `a24953c` with one documented parser deviation |
| Model / serving | Qwen2.5-7B-Instruct-AWQ `b25037543e9394b818fdfca67ab2a00ecc7dd641`, vLLM, greedy, 8192 ctx |
| Retrieval | BM25 (SparseRAG) and e5-base-v2 (DenseRAG), k=20, 306-question `s306` slice |
| Local tests | `python3 -m unittest discover -s tests` → **57 tests, OK** (9 skip without numpy); the same 57 pass with `.venv/bin/python` |
| Kaggle GPU quota spent | **0.00 h** (`has_ever_run: false`); 30 h/week allowance untouched |
| Successful DR rows | **0 / 612** |
| Blocker | accelerator requests are accepted and then silently not honoured (CPU-only container) |

## 1. Attempt 1 (Pi-era worker, 2026-11-12 17:58 UTC): setup failure, diagnosed

`phinguynxun/mpr-dr-reasoning` version 1 died after 0.0 s of script time:
`FileNotFoundError: manifest.json not found under /kaggle/input/mpr-dr-inputs`. It produced no rows and is
not an evaluation. Evidence, unedited: `results/kaggle/dr/attempt1_pi_setup_failure/`
(`DONE`, `status.json`, `manifest_out.json`, `run_log.txt`, `kernel_log.json`).

**Root cause (verified, not hypothesised).** A private CPU-only probe kernel
(`phinguynxun/mpr-dr-mount-probe`, source `experiments/s6_kaggle_dr/diagnostics/mount_probe.py`, output
`results/kaggle/dr/mount_probe/mountprobe.json`, 0 GPU quota) enumerated the real mount:

```
/kaggle/input/                             -> contains exactly one entry: "datasets"
/kaggle/input/datasets/phinguynxun/mpr-dr-inputs/   <- all 20 files present
manifest.json sha256 3fdfc315da42fe1d5e0d7c8b784959ac9ded014e47e6f42e09ebf531ab0a9b04
```

So `/kaggle/input/mpr-dr-inputs` never existed; the dataset is mounted under `datasets/<owner>/<slug>`.
This is a layout change, not dataset-processing lag (the API listed all 20 files with the expected sizes
at the same time).

**Fix (committed, tested).** `kernel.py: discover_input_root()` tries `MPR_INPUT_ROOT`,
`/kaggle/input`, `/kaggle/input/<slug>` and `/kaggle/input/datasets/<owner>/<slug>`, then walks
`/kaggle/input` to depth 4, retrying for 90 s, and publishes the resolved root, the candidate list and a
bounded tree to `input_probe.json` and `status.json`. `stage_inputs()` also refuses a manifest that
declares a different Kaggle dataset (so the MR dataset can never be staged by mistake). Covered by
`tests/test_kernel_stage.py::test_stage_inputs_discovers_nested_datasets_mount_layout` and the
failure-message and foreign-manifest tests.

## 2. Attempt 2 was prepared, then stopped on a hard blocker: no accelerator

The corrected job (mount fix + driver-aware vLLM pin + a 2.5 h session cap) was built and its private
input dataset updated to version 2 (later 3 and 4, the last one adding `kernel.py` for UI-started sessions)
but was **not** pushed: four independent accelerator probes proved
that this account receives no GPU at all.

| Probe | Kernel | Request | Result |
|---|---|---|---|
| 1 | `phinguynxun/mpr-dr-gpu-probe` (script) | `enable_gpu` + `machine_shape: NvidiaTeslaT4` + `--accelerator NvidiaTeslaT4` | no `nvidia-smi` (rc 127), no `/dev/nvidia*`, no `/proc/driver/nvidia`, `torch 2.11.0+cpu`, `cuda_available=false` |
| 2 | `phinguynxun/mpr-dr-gpu-probe2` (script) | `enable_gpu` only, no `machine_shape`, no `--accelerator` | identical |
| 3 | `phinguynxun/mpr-dr-gpu-probe3` (script) | `machine_shape` + `docker_image_pinning_type: latest` | identical |
| 4 | `phinguynxun/mpr-dr-gpu-probe-nb` (notebook) | `enable_gpu` + `--accelerator NvidiaTeslaT4` | identical |

Server-side saved metadata for all four carries `enable_gpu: true` and
`machine_shape: NvidiaTeslaT4` (`results/kaggle/dr/gpu_availability/metadata/`), so the request was
accepted; the runtime simply ignored it. Every session ran `RUNNING → COMPLETE` in ~20 s on CPU, and
`kaggle quota` stayed `GPU used 0.00h`, `quota_view().gpu_quota.time_used = 0`, `has_ever_run = false`
(`results/kaggle/dr/gpu_availability/quota_view.json`, raw endpoint response in `quota_api_raw.json`:
`totalTimeAllowed` 108000 s, `timeUsed` 0, `hasEverRun` false). A GPU capacity shortage queues a session
rather than silently downgrading it, and the image-pin variant rules out the known pinning pitfall. The MR
worker independently reproduced the same result for GPU and TPU (`TpuV5E8` sessions without `/dev/accel*`
or libtpu).

What is **established** is that this account is not being given an accelerator while it holds an unused
30 h/week GPU allowance and the server accepts its GPU settings. What is **not** established from
CLI/runtime evidence alone is the cause: an account-side eligibility requirement (Kaggle requires one-time
phone/identity verification for accelerators) or a platform-side allocation problem are both consistent
with the observations, and only the account owner or Kaggle support can distinguish them.

Why this stops the run instead of degrading: the frozen deployment is a T4 vLLM server
(`--quantization awq`, fp16 compute, 8192 context) that the NR/SR baselines were measured on; a CPU-only
vLLM run is not that deployment, cannot complete 4,284 greedy calls inside any sanctioned budget, and
would produce numbers that could not be compared with NR/SR. Pushing the 2.5 h job would therefore burn
the attempt on a setup failure.

## 3. What was verified locally (no GPU)

```
python3 -m unittest discover -s tests              # 57 tests, OK (9 numpy-dependent skips)
.venv/bin/python -m unittest discover -s tests     # 57 tests, OK (bootstrap/paired included)
python3 experiments/s6_kaggle_dr/package.py build  # dataset+kernel staged, manifest hashes, clean tree
```

* DR call flow, prompts, retrieval keys and current-state strings are mock-checked prompt-for-prompt
  against the upstream classes at `a24953c` (`tests/test_dr.py::UpstreamParityTests`).
* parser policy, bounded fallbacks, sequential solve, merge re-retrieval, gold-answer leakage guard,
  resume after a cut JSONL line, duplicate keys, deadlines, circuit breaker and concurrent unique-key
  coverage are covered.
* new in this attempt: mount-layout discovery, foreign-manifest rejection, driver-aware vLLM pinning
  (`vllm==0.32.0` needs driver ≥ 580; `0.11.2`/`0.10.2` are the CUDA-12 fallbacks, and the deployment
  sanity gate then uses a pre-registered 2/6 exact-response floor instead of 3/6 so a valid run is not
  aborted by token-level text drift), the closed-loop
  runtime projection (per-call latency × calls / worker slots, no more dividing the 612-row token total by
  the 12-row smoke wall clock), and tests for the two result tools (`tests/test_dr_scoring.py`, 11 tests:
  validator row invariants, missing/extra-key reporting, deduplication, paired-comparison denominators and
  zero-difference CIs, and the DR-vs-MR comparison tool with unequal coverage and no-overlap cases).
  Writing those tests found and fixed two validator crashes on out-of-slice rows.

## 4. Costs and artifacts

* GPU: 0.00 h used, 0.00 h reserved. CPU-only probe kernels: seconds each.
* Private dataset `phinguynxun/mpr-dr-inputs` version 4 (v1..v4; v4 adds `kernel.py` for UI sessions), private kernel
  `phinguynxun/mpr-dr-reasoning` version 1 (failed setup, preserved) and four private probe kernels.
* No credentials, tokens, embedding caches or model weights were committed. The probe JSONs that printed
  per-session `KAGGLE_*` tokens were sanitized before being stored.

## 5. Commits on `experiments/dr-kaggle`

* `56c096a`, `520bd22`, `3bb161a`, `0975d3d` — implementation, kernel, packaging, tests, driver (Pi-era,
  preserved).
* `1a76ce4` — mount-layout discovery, fail-loud input diagnostics, attempt-1 evidence, budget docs.
* `e78b59d` — driver-aware vLLM pin (`0.32.0` only on driver ≥ 580).
* plus the commit carrying this report (runtime projection, workers 40, quota correction, probe evidence).

## 6. Resume procedure (when an accelerator is available)

If API/CLI pushes are still being downgraded to CPU after verification, run the same committed code in an
**interactive** notebook (the private dataset now carries `kernel.py`, hash-covered by the manifest):

```python
import glob, shutil, subprocess, sys
src = glob.glob("/kaggle/input/**/kernel.py", recursive=True)[0]
shutil.copyfile(src, "/kaggle/working/kernel.py")
sys.exit(subprocess.call([sys.executable, "-u", "/kaggle/working/kernel.py"]))
```

with the dataset attached, internet on, T4 selected, and the shared lock held for the session; the code
enforces the same 9000 s cap, hashes and gates.

```bash
# 0. confirm the account now gets a device (2 min, no full job)
kaggle kernels push -p build/kaggle/dr_gpuprobe --accelerator NvidiaTeslaT4
kaggle kernels output phinguynxun/mpr-dr-gpu-probe -p /tmp/gpuprobe -o
jq '.torch, .devices' /tmp/gpuprobe/gpu_probe.json    # expect a device name and /dev/nvidia0

# 1. rebuild inputs/manifest from the final commit and upload them (private dataset v3)
python3 experiments/s6_kaggle_dr/package.py build
kaggle datasets version -p build/kaggle/dr_dataset -m "DR inputs for <commit>" --keep-tabular

# 2. one locked, capped GPU job (holds the shared flock until the kernel is terminal + downloaded)
experiments/s6_kaggle_dr/kaggle_run.sh                # T4, --timeout 9000, waits 4 h for the lock

# 3. independently validate and score the downloaded rows
python3 experiments/s6_kaggle_dr/validate_dr.py --runs results/kaggle/dr/runs \
  --slice results/t4/slices/s306.json --qa-dir data/mpr_dataset/question_answer
python3 experiments/s6_kaggle_dr/summarize_dr.py --dr results/kaggle/dr/runs \
  --slice results/t4/slices/s306.json --nr results/t4/s1 --sr results/t4/s2 \
  --out results/kaggle/dr/summary_dr.md
```

Completion requires exactly 306 unique successful rows per memory (612 total); anything less must be
reported as partial. The in-job gates (mount/hash verification, divide gate, 12-row smoke, NR deployment
sanity, runtime projection) all run before the full evaluation and stop it on failure.

## 7. Required user action

1. Check/complete phone/identity verification of the `phinguynxun` Kaggle account
   (<https://www.kaggle.com/settings>) — the most likely account-side cause — and/or open a Kaggle support
   ticket with the probe evidence; then rerun the 2-minute probe in §6 against a real device; **or**
2. explicitly authorize a different T4 provider for the frozen deployment (the repo already has the
   Colab T4 vLLM 0.32.0 recipe used by the earlier sessions); that is outside the current authorization.

Everything else in the experiment is implemented, tested, packaged and committed; no DR row exists and no
result is claimed.
