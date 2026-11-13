# Kaggle GPU docs audit - independent second-pass review

Reviewer: mpr-mr (t1f), 2026-11-13. Read-only: no probes, submissions, Colab,
auth or edits to the DR worktree. Reviewed artifact:
`mpr-study-dr/experiments/s6_kaggle_dr/kaggle_gpu_docs_audit.md` (commit
`b33568b`). All retrieval times UTC.

## (a) Conclusion

The audit's core conclusion **agrees with my independent check**: the current
documented GPU request is field-for-field what we already sent, no documented
correction exists, and the observed CPU-only containers contradict documented
behaviour. The cause stays **unresolved**; no new accelerator probe is
justified. Three claims need narrowing: the phone-verification wording, the
queue inference, and the "only post-release push change" statement. The one
documented path not yet exercised is the interactive UI accelerator selection,
which needs the account owner.

## (b) Primary sources re-fetched and checked (2026-11-13, 05:02-05:05Z)

| Source | Version / date | Verified here |
|---|---|---|
| `raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/kernels.md` | main `e3583ad` (2026-11-12T21:23:54Z) | `kernels push` uploads and runs; `--accelerator NvidiaTeslaT4` documented as "GPU T4 x2, default GPU"; list of 8 accelerators; "Some of these are only available to participants of specific competitions, and some are only available to Kaggle admins"; `--no-run` saves without executing |
| `.../docs/kernels_metadata.md` | same commit | `enable_gpu` bool (default false), `machine_shape` e.g. `NvidiaTeslaT4` for GPU T4 x2; `docker_image` / pinning keys are **not** documented here |
| `https://www.kaggle.com/docs/notebooks.md` | live page, 05:03Z | GPU added via Settings -> Accelerator -> GPU; 12 h CPU/GPU session limit; T4 x2 specs; "Free GPU availability is limited: in busy times, you might be placed in a queue"; page still advertises a single P100 (stale) |
| `https://www.kaggle.com/docs/efficient-gpu-usage.md` | live page, 05:03Z | 30 h/week GPU quota; recommends the Kaggle API/batch runs; no accelerator restriction or verification statement |
| `https://www.kaggle.com/docs/api.md` | live page, 05:03Z | OAuth scopes; no accelerator-specific API requirement |
| PyPI `kaggle`, `kagglesdk` | 2.2.5 (2026-10-22T22:01:21Z), 0.1.38 (2026-10-21T22:01:50Z) | installed CLI is 2.2.5 and the SDK is 0.1.38, both current |
| `raw.githubusercontent.com/Kaggle/docker-python/main/README.md` | main, 05:03Z | two families: CPU-only `gcr.io/kaggle-images/python`, GPU `gcr.io/kaggle-gpu-images/python`; our sessions used the CPU-only digest |
| `api.github.com/repos/Kaggle/kaggle-cli/pulls/1219`, commit `42f85b6` | merged 2026-10-26T20:52:41Z | clears stale image pins **for TPU pushes only**; derives `enable_tpu` from an accelerator prefix; no GPU scheduling change |
| `api.github.com/repos/Kaggle/kaggle-cli/commits?path=src/kaggle/api/kaggle_api_extended.py` | after v2.2.5 (2026-10-22T22:03:24Z) | post-release push-path commits include `42f85b6` (#1219, TPU pins) **and `87ea99e` (#1232, restrict kernel push to python)** plus pull/delete/auth/exit-code fixes; none changes GPU scheduling |

## (c) Wire conversion, not just attributes (offline)

I built `ApiSaveKernelRequest` the way `kernels_push` does for the real MR
metadata and serialized it with the installed SDK's `to_json()`:

```json
{"id": 144700338, "slug": "mpr-mr-reasoning", "newTitle": "MPR MR reasoning",
 "text": "...", "language": "python", "kernelType": "script",
 "datasetDataSources": ["phinguynxun/mpr-mr-inputs"], "isPrivate": true,
 "enableGpu": true, "enableInternet": true, "enableTpu": false,
 "sessionTimeoutSeconds": 14400, "machineShape": "NvidiaTeslaT4"}
```

`kernelExecutionType` appears only for `--no-run` (`QUICK_SAVE`); a normal push
leaves the server default. `dockerImage`/`dockerImagePinningType` are omitted
when unset (DR's capture showed empty strings; that is a metadata difference,
not a missing field). This matches the documented push request and the fields
DR captured, so there is no documented wire-level omission.

## (d) Agreement with the audit

- The documented procedure is what we sent: `machine_shape` (T4 x2) plus the
  deprecated `enable_gpu`, script and notebook, internet on, timeout within the
  limit, batch/API as the recommended path.
- No current formal doc describes a silent CPU fallback, and the docs do show
  queueing under load and accelerator shapes restricted to some accounts.
- The absence of `/dev/nvidia*` and the NVIDIA driver means no GPU was attached;
  this is not an unused GPU.
- Docs are stale (P100 on the notebook page, eight accelerators on the CLI
  page), so the SDK docstring's three-shape list is not authoritative.
- Saved metadata is not execution proof: `pull -m` returns saved config,
  `kernels list` does not project `enable_gpu`/`machine_shape` (false/empty for
  every kernel), and `get_kernel.last_run_time` was identical across kernels
  created on different days.
- No documented correction was found; a repeat T4 probe would repeat a
  documented request.

## (e) Corrections and unsupported claims

1. **Phone verification is overstated.** The statement "no current official
   document states that phone/identity verification is required for GPUs" is
   true only for the three formal pages fetched, which are demonstrably stale
   and silent on account eligibility. Official Kaggle staff/community guidance
   does state it: `kaggle.com/product-feedback/268333` ("phone verify your
   account ... after that, you should be able to enable GPU") and
   `kaggle.com/discussions/general/97939` ("your account must be verified to
   see the Accelerator option", 2023). These are supporting, dated and not
   current formal docs, so verification remains an **unproven possible cause**;
   `hasEverRun=false` says nothing about it. It should not be described as
   disproven or as "not a requirement".
2. **The queue inference is not evidence.** A short `QUEUED`/`RUNNING` phase
   does not prove queueing was absent, nor does it establish account/platform
   causation. The decisive observation is container-level: no device nodes and
   no driver. Session timing should not be used either way.
3. **"The only post-release push change is #1219" is imprecise.** `87ea99e`
   (#1232, restrict kernel push to python) is also post-release and touches the
   push path; it is irrelevant to GPU scheduling, so the conclusion that
   upgrading would not change GPU behaviour still stands.
4. **"Image pinning optional" is CLI-source behaviour, not documented.** The
   metadata docs do not document `docker_image` at all, and there is no
   documented way to force the GPU image family. The server-assigned CPU-only
   image is consistent with a CPU session; it is not a documented knob we
   missed.
5. **Minor:** DR's row "no timeout on probes" vs the real kernel's 9000/14400 s
   does not affect acceleration; and the audit's claim that execution "works"
   means the script ran, not that it ran accelerated.

## (f) Was a documented execution step missed?

No. Command, metadata keys and JSON types, accelerator identifier, CLI-vs-
metadata precedence, script/notebook support, `--no-run` semantics and the
batch/API recommendation all match. The docs do not state that API pushes lack
GPU support; the opposite is recommended. The only documented GPU-selection
path not exercised is the interactive UI (Settings -> Accelerator -> GPU),
which requires a signed-in browser session owned by the account holder.

## (g) One precise next action

Account owner starts **one private notebook in the Kaggle UI with
Accelerator = GPU T4 x2** and runs only:

```python
!nvidia-smi -L
import torch; print(torch.cuda.is_available())
```

If a device appears, re-test the already-prepared private MR API push exactly
once under the existing 4 h cap and shared lock; if it does not, the remaining
hypotheses are account eligibility (an unverified/flagged flag not exposed by
the API) or platform-side provisioning, which this documentation audit cannot
resolve. No probe is proposed or run here.

## (h) Evidence limits

Read-only review. No Kaggle probes, submissions, Colab or auth commands were
run. No credentials, authorization headers or KAGGLE_* session tokens were
read or copied. DR's worktree was read, not modified.
