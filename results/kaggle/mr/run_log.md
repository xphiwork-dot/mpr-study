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
