# Accelerator availability on the shared Kaggle account (2026-11-13)

All four kernels were pushed as **private**, each with `enable_gpu: true`, and each ran to
`KernelWorkerStatus.COMPLETE` on a **CPU-only** container. `kaggle quota` reported `GPU used 0.00h` after
every run, and `api.quota_view().gpuQuota.hasEverRun` stayed `false`, so no accelerator-second was ever
allocated or billed. Files here are sanitized: the per-session `KAGGLE_*` tokens that the probe printed
were removed before committing (`kaggle_env` deleted).

| Probe | Kernel | Type | Requested | Runtime result | Evidence |
|---|---|---|---|---|---|
| 1 | `phinguynxun/mpr-dr-gpu-probe` | script | `enable_gpu` + `machine_shape: NvidiaTeslaT4` + `--accelerator NvidiaTeslaT4` | no `nvidia-smi` (rc 127), no `/dev/nvidia*`, no `/proc/driver/nvidia`, `torch 2.11.0+cpu`, `cuda_available=false` | `probe1_script_accelerator.json` |
| 2 | `phinguynxun/mpr-dr-gpu-probe2` | script | `enable_gpu` only, **no** `machine_shape`, no `--accelerator` | identical CPU-only result | `probe2_script_enable_gpu_only.json` |
| 3 | `phinguynxun/mpr-dr-gpu-probe3` | script | `machine_shape: NvidiaTeslaT4` + `docker_image_pinning_type: latest` (the TPU-issue workaround) | identical CPU-only result | `probe3_script_latest_image_pin.json` |
| 4 | `phinguynxun/mpr-dr-gpu-probe-nb` | **notebook** | `enable_gpu` + `--accelerator NvidiaTeslaT4` | identical CPU-only result | `probe4_notebook_accelerator.json` |

The server **stored** the GPU request in every case (`metadata/*.json`: `enable_gpu: true`,
`machine_shape: NvidiaTeslaT4`, script and notebook), so the request was accepted and then silently not
honoured. The same account's CPU-only mount probe (`phinguynxun/mpr-dr-mount-probe`,
`../mount_probe/mountprobe.json`) ran without any accelerator request, as expected.

## Interpretation

The failure is account-level, not a code or metadata problem:

* not our metadata: the server-side saved metadata carries the accelerator request (four kernels, two
  kernel types, three request shapes);
* not the image pin: clearing/pinning the image changes nothing (probe 3 vs 1);
* not capacity/queueing: a GPU shortage queues a session (`QUEUED`) or fails it; these sessions went
  `RUNNING` → `COMPLETE` in ~20 s on CPU and consumed zero GPU quota;
* not a single-kernel fluke: four independent kernels plus the MR worker's three probes
  (`mpr-mr-mount-probe`, notebook, TPU) reproduce it, including `TpuV5E8` sessions without
  `/dev/accel*` or `libtpu`.

The observable matches Kaggle's one-time phone/identity verification requirement for accelerator use:
the account has never had an accelerator session (`hasEverRun: false`) yet the CLI reports a nominal GPU
allowance, and push-time requests are accepted without error.

## What this blocks

The approved experiment needs a real T4 (Qwen2.5-7B-Instruct-AWQ over vLLM, fp16, 8192 context, 612 rows
x 7 calls). Running the same job on CPU is not a valid reproduction of the frozen deployment and cannot
finish in any sanctioned budget, so the DR evaluation was **not** started: 0 of 612 rows exist and no
EM/contains number is reported. No GPU quota was spent (`GPU used 0.00h`).

## Required to unblock

1. Phone/identity verification of the `phinguynxun` Kaggle account at <https://www.kaggle.com/settings>
   (the account owner must do this), then re-run the probe and, if it reports a device, the prepared job
   (procedure in `../../../../experiments/s6_kaggle_dr/report.md` §Resume); or
2. An explicit user decision to run the frozen deployment on a different T4 provider (the repo already
   has the Colab T4 recipe used by the earlier sessions), which is outside the current authorization.
