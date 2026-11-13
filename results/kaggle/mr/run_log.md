# MR Kaggle run log (OMP worker, 2026-11-13)

Branch `experiments/mr-kaggle`, base commit `c1ad941` plus the fixes below. All
times UTC. Diagnostics only; no evaluation rows were produced.

## 1. Package pre-flight (local, no GPU)

- Pinned versions exist on PyPI: `vllm==0.32.0` (requires `torch==2.13.0`,
  `transformers<5.20.0,>=5.16.1`), `sentence-transformers==6.3.0`,
  `bm25s==0.2.9`.
- `pip install --dry-run vllm==0.32.0` resolves on Linux x86_64 (torch 2.13.0 +
  CUDA 13.5 wheels). This matches the prior T4 recipe in
  `experiments/_shared/t4.sh` (vLLM 0.32.0, torch 2.13).
- Private dataset `phinguynxun/mpr-mr-inputs` v1 downloaded and re-verified:
  all 24 allowlisted files present, every SHA-256 matches `input_manifest.json`
  (built at commit `c1ad941`, `git_dirty=false`). `mr.py`, `lib.py` and
  `score.py` are byte-identical to this worktree.

## 2. Kaggle dataset mount layout (root cause of the peer's failed run)

A CPU probe kernel on this account shows the dataset at

```
/kaggle/input/datasets/phinguynxun/mpr-mr-inputs/   (24 flat files)
```

not `/kaggle/input/mpr-mr-inputs`. `kernel.py` previously globbed
`/kaggle/input/*/s306.json`, which cannot match the nested layout, so the
kernels would have failed exactly like the peer's DR run
(`FileNotFoundError: manifest.json not found under /kaggle/input/mpr-dr-inputs`).
The peer's independent CPU probe found the same nested layout for the DR dataset.

Fix in `experiments/s5_kaggle_mr/kernel.py`: candidate roots
(`MPR_INPUT_ROOT`, flat legacy, `/kaggle/input/datasets/<owner>/<slug>`), then a
bounded `os.walk` of `/kaggle/input`, a 90 s retry window and a logged mount
tree on failure. Regression tests: `tests/test_mr.py::KernelMountTest`.

## 3. GPU provisioning: Kaggle runs every accelerator request on CPU

Quota before the probes (`kaggle quota`, 2026-11-13T02:0xZ):

```
GPU  used 0.00h  remaining 30.00h  total 30.00h  refresh 2026-11-14T00:00Z
TPU  used 0.00h  remaining 20.00h  total 20.00h  refresh 2026-11-14T00:00Z
```

The new `api.kaggle.com` quota object (CLI `KaggleApi.quota_view()`) stores
the allowance as a timedelta; its `.seconds` component is 21600 s but the full
value is 1 day + 6 h = 108,000 s = the same 30 h/week, with `timeUsed=0s`,
`hasEverRun=False`, `is_frozen=True`. The quota was therefore not exhausted;
allocation, not quota, is what failed.

Probe results (all private kernels on `phinguynxun`, each `enable_gpu` /
`enable_tpu` set in `kernel-metadata.json` and passed via `--accelerator`):

| # | shape requested | result |
|---|---|---|
| 1 | `NvidiaTeslaT4` (script, zone asia-east1-a) | no `/dev/nvidia*`, no `/proc/driver/nvidia`, `torch 2.11.0+cpu`, `torch.cuda.is_available() == False` |
| 2 | `NvidiaTeslaT4` (notebook, same metadata) | identical CPU result |
| 3 | `TpuV5E8` (zone us-central1-c) | no `/dev/accel*`, no `libtpu`/`torch_xla` |
| 4 | `NvidiaTeslaT4`, `docker_image_pinning_type: latest`, no dataset (zone europe-west4-c) | `torch 2.11.0+cpu`, no GPU |

`kaggle quota` still reports `used 0.00h` after all probes: no accelerator
session was ever started. The internal quota object's `hasEverRun=False`
confirms the account has never had an accelerator session.

Hosted references: Kaggle documents that free GPU/TPU access requires one-time
phone verification, and kaggle-cli issue #1197 documents that push-based
accelerator requests can be accepted while the session silently runs in the
standard CPU/GPU image. Here the failure is not the pinned `docker_image`
(probe 4 pinned `latest` and still had no device) and not the zone (three
zones). The evidence establishes only CPU-only provisioning; possible causes
are account accelerator entitlement (verification or hold) or a Kaggle
platform fault. No cause is concluded, and none can be fixed from inside the
kernel.

Probe kernels (private, kept as evidence):
`phinguynxun/mpr-mr-mount-probe` (v1 CPU mount probe, v2 CPU environment probe,
v3 GPU script probe, v4 GPU device probe, v5 TPU probe),
`phinguynxun/mpr-mr-gpu-notebook-probe` (v1 notebook GPU probe),
`phinguynxun/mpr-mr-gpu-probe2` (v1 latest-pinning GPU probe).

## 4. Work completed before the blocker

- Mount-resolution fix + CUDA post-install check (`check_cuda`) in `kernel.py`.
- Session budget aligned with the shared 6 h conservative quota view: 3 h job
  timeout, 20 min reserve, 64 workers (matches `--max-num-seqs 64`).
- 18 local tests pass (`python -m unittest tests.test_mr -v`).
- No GPU session, no evaluation rows, no scores: the experiment is blocked at
  accelerator provisioning, not at the method or the packaging.

## 5. Local end-to-end wiring check (stub model, no GPU; not a result)

The real CLI path was exercised against the released dataset, the pinned
upstream templates and a local OpenAI-compatible stub (no model weights):

```bash
.venv/bin/python experiments/s5_kaggle_mr/mr.py run --stage full \
  --slice results/t4/slices/s306.json --data-dir ../mpr-study/data/mpr_dataset/question_answer \
  --cache-dir /tmp/mr-local/cache --output-dir /tmp/mr-local/out \
  --memories sparse --topk 20 --max-step 5 --branch-size 2 --expected-calls 12 \
  --seed 0 --workers 2 --limit 2 --device cpu
```

Observed: 2 unique rows, 24 calls (12 per row, no failures), real BM25 top-20
ids per retrieval call, selection buffer sizes 1/3/5/7 (old paths retained),
selected final path depth 2 (a retained shallow path was chosen, as upstream
allows), 0 duplicate branch pairs, coverage rc=0. `analyze_mr.py
--allow-partial` then produced the summary and `analysis_validation.json`
without error. These are wiring checks only; the stub's strings are not
experiment results and nothing from this run is committed.

## 6. Re-check after the user reported access fixed (2026-11-13T02:47-02:52Z)

The user confirmed Kaggle accelerator access was fixed. Two requests were made
after that confirmation, both still CPU-only:

- `phinguynxun/mpr-mr-gpu-probe2` v2, `--accelerator NvidiaTeslaT4`, run at
  02:47Z in `us-central1-c`: no `/dev/nvidia*`, no `/proc/driver/nvidia`,
  `torch 2.11.0+cpu`, `cuda False`.
- `phinguynxun/mpr-mr-gpu-fresh-2026-11-13` v1, a brand-new kernel id with
  `enable_gpu: true`, `--accelerator NvidiaTeslaT4`, started
  2026-11-13T02:52:05Z in `asia-east1-c`: same result.

`kaggle quota` stayed at GPU 0.00h used / 30.00h total and the API quota object
still reports `hasEverRun=false`, so no accelerator session was started at all.
Sanitized machine-readable evidence: `gpu_availability/probes.json`.

The MR kernel `phinguynxun/mpr-mr-reasoning` was created with `--no-run` only
(version 1, correct GPU metadata and dataset source; no experiment code ran and
no quota was used). Its save-validation session ended in
`KernelWorkerStatus.ERROR` while converting the saved script; a later running
push creates a new version and is unaffected.

At this point the local coordinator asked for direct UI evidence (a private
notebook with T4 enabled and successful `nvidia-smi` output) before any
full-job submission, because the push/API path still provisions CPU only.
No further probes or submissions were made.

## 7. UI-executable code shipped in the private dataset (no GPU)

Dataset `phinguynxun/mpr-mr-inputs` v2 adds `kernel.py` (SHA-256
`82641cb08066c9e2f04a0c35b4ff836b3022316dcfde01523b34acb5c01408d4`, recorded
in `input_manifest.json`; `git_commit=2132ac7`, `git_dirty=false`, 25 files).
A UI-started private notebook with the dataset attached, internet on and GPU
T4 x2 can exec the exact committed entrypoint:

```python
import glob, shutil, subprocess, sys
src = glob.glob('/kaggle/input/**/kernel.py', recursive=True)[0]
shutil.copyfile(src, '/kaggle/working/kernel.py')
sys.exit(subprocess.call([sys.executable, '-u', '/kaggle/working/kernel.py']))
```

This is code delivery only; no session was started.
