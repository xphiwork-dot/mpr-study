# Kaggle GPU documentation audit — current official sources vs our actual requests

Prepared by mpr-dr (t1g) as the sole primary investigator, 2026-11-13. Bounded research only: no probes,
no submissions, no provider execution, no Colab/auth flow. The main experiment remains **BLOCKED**.
Independent second-pass review: `mpr-study-mr/experiments/s5_kaggle_mr/kaggle_gpu_docs_review.md`
(mpr-mr, 2026-11-13); the four corrections it requested are applied here and logged in (g).

## (a) Conclusion first

1. **Every material step in the current official GPU instructions is already what we sent.** The documented
   API path is: `kernel-metadata.json` with `machine_shape` (T4 = `NvidiaTeslaT4`, i.e. GPU T4 ×2) plus
   `enable_internet`, then `kaggle kernels push -p DIR --accelerator NvidiaTeslaT4 --timeout <s>`. Our five
   accelerator probes (script and notebook, three request shapes, including a fresh kernel pushed with
   `docker_image_pinning_type: latest`) serialized exactly those fields, and the runs executed.
2. **The observed outcome contradicts the documented expectation.** Docs describe queueing when capacity is
   tight and do not document a silent CPU substitute. Our sessions completed without any GPU device:
   no `/dev/nvidia*`, no `/proc/driver/nvidia`, no `nvidia-smi`, `torch 2.11.0+cpu` (the CPU-only image),
   while `GET /api/v1/kernels/quota` reported GPU `timeUsed=0`, `totalTimeAllowed=108000` s,
   `hasEverRun=false`. Session timing is recorded as an observation only and is **not** used as evidence
   about queueing or causation.
3. **No documented correction was found.** Image pinning is CLI-source behaviour, not a documented metadata
   key (see (c)), and the closest control available (fresh kernel + `docker_image_pinning_type: latest`, no
   `docker_image`) was in fact already exercised (probe 3) and also completed CPU-only. The post-release
   push-path changes are #1232 (push restricted to python) and #1219 (TPU image pins only); neither changes
   GPU scheduling. **No validation command is proposed** (see (f)); the cause remains unresolved from
   documentation alone.
4. **Correction to our own earlier narrative (narrowed after review):** none of the three formal
   documentation pages reviewed states a phone/identity verification requirement for GPUs, but those pages
   are stale and silent on account eligibility. Dated official staff/community guidance (source 11) does say
   verification is needed before the Accelerator option/GPU becomes usable. That guidance is not current
   formal documentation, so verification remains an **unproven possible cause** — neither established nor
   disproven; `hasEverRun=false` says nothing about it. The formal docs do establish that some accelerators
   are restricted (competition participants/admins).

## (b) Sources, versions and retrieval times (all UTC, 2026-11-13)

| # | Source | Version / commit / date | Retrieved | What it establishes |
|---|---|---|---|---|
| 1 | Kaggle CLI docs `docs/kernels.md` — https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/kernels.md | `Kaggle/kaggle-cli` main `e3583ad834dc` (2026-11-12T21:23:54Z); latest release `v2.2.5` (2026-10-22T22:03:24Z) | 04:50:06Z | `kernels push` semantics, `--accelerator`, `--timeout`, `--no-run`; accelerator list "as of Sep 2026": `NvidiaTeslaT4` (GPU T4 ×2), `NvidiaTeslaA100`, `NvidiaL4`, `TpuV5E8`, `NvidiaL4X1`, `TpuV6E8`, `NvidiaH100`, `NvidiaRtxPro6000`; "Some of these are only available to participants of specific competitions, and some are only available to Kaggle admins." |
| 2 | Kaggle CLI docs `docs/kernels_metadata.md` — same repo | same commit | 04:50:06Z | Exact metadata keys; `machine_shape` "accelerator/GPU type (e.g. `NvidiaTeslaT4` for GPU T4 ×2, `NvidiaL4`, `TpuV5E8`)"; `enable_gpu` bool; `is_private` defaults true |
| 3 | Official notebook docs — https://www.kaggle.com/docs/notebooks.md | live page | 04:53:56Z | GPU is added in the editor: Settings → Accelerator → GPU; "Free GPU availability is limited: in busy times, you might be placed in a queue"; 12 h limit for CPU and GPU sessions; "T4 x2 GPU Specifications — 2 Nvidia Tesla T4 GPUs"; page still says "a single NVIDIA Tesla P100" (stale) |
| 4 | Official GPU-usage docs — https://www.kaggle.com/docs/efficient-gpu-usage.md | live page | 04:53:56Z | 30 h/week GPU quota ("resets weekly and is 30 hours or sometimes higher depending on demand"); recommends batch/API runs: "Consider using the Kaggle-API to avoid interactive sessions entirely"; discusses cancelling batch sessions |
| 5 | Official API docs — https://www.kaggle.com/docs/api.md | live page | 04:53:58Z | OAuth scopes incl. `kernels.execute`; no accelerator-specific API requirement documented |
| 6 | PyPI `kaggle` / `kagglesdk` | kaggle **2.2.5** (2026-10-22T22:01:21Z); kagglesdk **0.1.38** (2026-10-21T22:01:50Z); both installed versions match | 04:50:04Z | Installed client is current; the SDK request schema is the one the CLI serializes |
| 7 | GitHub commits since v2.2.5 | `src/kaggle/api/kaggle_api_extended.py`: 10 commits, latest `e3583ad` (2026-11-12); `src/kagglesdk/kernels/types/kernels_api_service.py`: 0 commits | 04:57:58Z | Post-release push-path commits include `42f85b6` #1219 "clear stale image pins for TPU pushes" (2026-10-26) and `87ea99e` #1232 "Restrict kernel push to python" (2026-10-22), plus pull/delete/auth/exit-code fixes. #1219 derives `enable_tpu` from the accelerator prefix and clears `docker_image` + notebook `dockerImageVersionId` **for TPU only**; #1232 does not touch GPU scheduling; the request schema is unchanged, so upgrading from 2.2.5 would not change a GPU request |
| 8 | Kaggle/docker-python README — https://raw.githubusercontent.com/Kaggle/docker-python/main/README.md | main, tags `v173-{CPU,GPU,TPU}-…` | 04:55:10Z | Two image families: CPU-only `gcr.io/kaggle-images/python`, GPU `gcr.io/kaggle-gpu-images/python`; our runs used the CPU-only digest and `torch 2.11.0+cpu` |
| 9 | Official issue tracker (supporting only) — Kaggle/kaggle-cli | #1197 closed (2026-09-17), #1219 merged 2026-10-26 (commit `42f85b6`), #1151/#1156 closed (P100 + image), #1196/#1198 closed (T4 = GPU T4 ×2), #490 open since 2023, #907/#1029 (CLI `--gpu`, `kaggle quota`) | 04:55:2xZ | Maintainers confirmed a pinned CPU/GPU `docker_image` can override accelerator provisioning (TPU case) and auto-clear stale pins **for TPU pushes only**; docs clarified `NvidiaTeslaT4` = GPU T4 ×2 |
| 10 | Official service status | no official endpoint found: `status.kaggle.com` does not resolve; `kaggle.com/status` is a user profile | 04:52Z | No documented outage source to check |
| 11 | Independent second-pass review (MR) — `mpr-study-mr/experiments/s5_kaggle_mr/kaggle_gpu_docs_review.md` | reviewer re-fetched the cited sources 2026-11-13 05:02–05:05Z | 05:06Z | Confirms the core conclusion and corrects four claims: phone-verification wording, the queue-timing inference, the "only post-release change" statement, and the undocumented image-pinning keys; cites dated official staff/community guidance <https://www.kaggle.com/product-feedback/268333> and <https://www.kaggle.com/discussions/general/97939> |

Staleness note: the installed `kagglesdk` docstrings still list only `NvidiaTeslaT4`, `NvidiaTeslaP100`,
`Tpu1VmV38` as machine shapes, while current CLI docs list eight accelerators. The SDK docstrings are not a
reliable source; the CLI docs at main are. Conversely the notebook docs page still advertises P100 while
CLI docs describe T4 ×2 — both pages are partly stale.

## (c) Current supported GPU procedure (per sources 1–4, 8)

Exact command:

```bash
kaggle kernels push -p <kernel-dir> --accelerator NvidiaTeslaT4 --timeout 9000
```

Exact `kernel-metadata.json` (keys and JSON types as documented; `enable_gpu` is marked DEPRECATED in the
SDK in favour of `machine_shape`, but still accepted):

```json
{
  "id": "phinguynxun/<slug>",
  "title": "<title>",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Precedence and behaviour, as established by the installed CLI source (2.2.5; post-release push changes are
#1232 and #1219, neither affecting GPU scheduling): `--accelerator` overrides `machine_shape`;
`enable_gpu`/`enable_tpu` are read from metadata and are deprecated; `--timeout` becomes
`session_timeout_seconds`; `--no-run` sets `QUICK_SAVE` and **does not execute**; a plain push sets
`KERNEL_EXECUTION_TYPE_UNSPECIFIED` (server default; executes, as our runs did).

`docker_image` and `docker_image_pinning_type` are **not documented** in `docs/kernels_metadata.md` at all:
the key names, the `original|latest` values (`valid_push_pinning_types`) and the "used in the last session"
semantics come from the CLI source, the SDK schema and `kernels pull -m` output. There is also no documented
way to force the GPU image family (`gcr.io/kaggle-gpu-images/python`). The server-assigned CPU-only image in
our saved metadata is therefore consistent with a CPU session — it is not a documented knob we missed.

Verification of real hardware (required before any result is believed): `/dev/nvidia*`, `/proc/driver/nvidia`,
`nvidia-smi -L`, and `torch.cuda.is_available()` **inside the session**, not the saved metadata.

## (d) Documented requirements vs what we actually sent

Request fields below were captured offline by intercepting `save_kernel` for the same metadata/folders we
pushed (no network, no new submission): `enable_gpu=True`, `machine_shape="NvidiaTeslaT4"`,
`session_timeout_seconds=9000`, `kernel_execution_type=UNSPECIFIED`, `docker_image=""`,
`docker_image_pinning_type=""` or `"latest"`, `is_private=True`.

| Documented requirement | Our probes | Match? |
|---|---|---|
| `machine_shape` (or `--accelerator`) names the accelerator; T4 = `NvidiaTeslaT4` (GPU T4 ×2) | probe 1/3/4 sent `machine_shape: NvidiaTeslaT4`; probe 1/4 also passed `--accelerator NvidiaTeslaT4` | yes |
| Legacy `enable_gpu` accepted (deprecated) | all probes sent `enable_gpu: true` | yes |
| `enable_internet` for model/data downloads | true in every probe and the DR kernel | yes |
| script **and** notebook both supported | probe 1/2/3 script; probe 4 notebook (`kernel_type: notebook`, real `.ipynb`) | yes |
| image pinning keys are **not documented** in the metadata docs; issue #1197 documents a TPU-only override caused by stale pins | probe 3: **fresh kernel**, no `docker_image`, `docker_image_pinning_type: latest` (CLI-source value) | n/a — undocumented key; exercised anyway |
| timeout ≤ global maximum | 9000 s requested on the DR kernel; probes used the server default (no `--timeout`) | yes |
| API/batch runs are the recommended GPU path | used `kaggle kernels push` throughout | yes |
| Busy periods → documented queue rather than a CPU substitute | sessions recorded `RUNNING → COMPLETE` with no GPU device; no queue phase observed | observation only — timing is not used as causal evidence |
| Accelerators may be restricted to some accounts | docs say this only for "some" of the list; T4 is described as the default GPU; no documented account requirement | **unknown** |

Runtime outcome after those exact requests (existing sanitized artifacts, same evidence base as
`results/kaggle/dr/gpu_availability/`): all probes `COMPLETE`; `nvidia_smi_L.rc=127` ("not found"),
`devices=[]`, no `/proc/driver/nvidia/version`, `torch 2.11.0+cpu`, `cuda_available=false`, `device_count=0`;
`kaggle quota` GPU `used 0.00h`; quota API `timeUsed=0`, `hasEverRun=false`. A DR attempt with the same
metadata reached setup and failed only on the (now-fixed) input path: the batch execution path runs the
script, but that run was never accelerated either.

Additional evidence limits found during this audit (why saved metadata cannot settle the question):

* `kernels pull -m` reflects the **saved configuration** (`enable_gpu: true`, `machine_shape:
  NvidiaTeslaT4`, server-assigned `docker_image: gcr.io/kaggle-images/python@sha256:0717…`, the CPU-only
  image family). Metadata being saved is not proof of accelerated execution.
* `kernels list` (`ApiKernelMetadata`) returned `enable_gpu=False` and `machine_shape=''` for **every**
  kernel in the account, including ones whose saved config says GPU — the list projection does not carry
  these fields; it must not be read as "no GPU was requested".
* `get_kernel` reported the identical `last_run_time` `2026-11-10T04:52:09` for all kernels, including one
  created on 2026-11-13 — that timestamp field is unreliable. No official endpoint was found that reports
  the machine actually allocated to a finished batch session.

## (e) Supported conclusions vs unknowns

**Supported by the sources (facts):**

1. The documented API/CLI GPU procedure is the one we used, including both `machine_shape` and the
   deprecated `enable_gpu`, both script and notebook, with and without `docker_image_pinning_type: latest`.
   Post-release push-path changes are #1232 (push restricted to python) and #1219 (TPU image pins only);
   neither changes GPU scheduling, so upgrading would not change a GPU request.
2. The documented operational notes are: queueing when capacity is tight (a queue is documented; our
   recorded timings are not used as evidence one way or the other); accelerator shapes restricted for some
   accounts; stale image pins overriding accelerator provisioning (documented for the TPU case, auto-cleared
   only for TPU by #1219).
3. None of the formal documentation pages reviewed states a GPU verification requirement (they are stale and
   silent on account eligibility), no formal page documents a silent CPU fallback, and no formal page
   documents a way to force the GPU image family. Dated official staff/community guidance on verification is
   treated as an unproven hypothesis in the unknowns below.
4. The absence of device nodes and driver inside the container means the session had no GPU attached; it is
   not merely "an unused GPU".
5. Our requests match the documented ones field-for-field; therefore "we forgot a parameter" is not
   supported by the evidence.

**Unknown (and not claimed):**

* Why the backend created CPU sessions for correctly serialized GPU requests. Candidates, all unproven:
  an account-level eligibility flag not documented or exposed by the API; a platform/zone provisioning
  problem with silent downgrade; a backend bug specific to batch sessions on this account. Phone/identity
  verification is a **dated staff-guidance** hypothesis (source 11, not current formal documentation:
  `kaggle.com/product-feedback/268333`, `kaggle.com/discussions/general/97939`); it is neither established nor disproven, and `hasEverRun=false` is not evidence about it.
* Whether the interactive UI path (Settings → Accelerator → GPU) allocates a device on this account — that
  is the documented user-facing path and the only officially described way to select a GPU that we have not
  exercised; it requires a signed-in browser session.
* Whether any documented control can force the GPU image family: none is documented (`docker_image` and the
  pinning keys are CLI-source only), so this cannot be checked from the docs.
* Whether the CLI-source `docker_image_pinning_type: latest` selects the GPU image for a GPU shape (the SDK
  description is only "which docker image to use for executing new versions").

**Rejected as undocumented fixes (do not repeat as new evidence):** pushing the same T4 shape again; adding
`enable_gpu`; switching script↔notebook; `docker_image_pinning_type: latest`; L4/L4X1 shapes; TPU shapes —
all were exercised or are irrelevant to the observed deviceless runtime.

## (f) Proposed minimal validation command

**None is proposed.** No genuine documented correction exists beyond what was already executed: there is no
documented key that forces the GPU image family, and the image-pinning variant that exists only in CLI
source was already exercised (probe 3), so another capped probe would only repeat a documented request. The single untested documented path is the interactive
UI accelerator selection (source 3), which is an account-owner action, not a kernel command; if a UI session
shows `nvidia-smi` output, the API path can then be re-tested under the same capped scope.

If the account owner or Kaggle support can state which account eligibility flag is involved, the minimal
private check that would settle it is a ≤60 s script kernel whose only job is to print
`/dev/nvidia*`, `/proc/driver/nvidia/version`, `nvidia-smi -L`, `torch.cuda.is_available()` — i.e. exactly
the probe already recorded in `results/kaggle/dr/gpu_availability/`; it needs no new code and is not
proposed for execution here.

## (g) Corrections applied after independent review

Reviewer: mpr-mr, `mpr-study-mr/experiments/s5_kaggle_mr/kaggle_gpu_docs_review.md`. Four corrections
applied, core conclusion preserved (**no missing documented GPU step/field; cause unresolved**):

1. **Phone verification** — narrowed to "not stated in the three formal pages reviewed"; the dated official
   staff/community guidance is now cited as an *unproven eligibility hypothesis*, not as a disproven one.
2. **Queue timing** — removed the inference that fast completion contradicts queueing or implies a cause;
   the decisive observation is container-level (no device nodes, no driver), and timing is observation only.
3. **Post-release push changes** — added #1232 (push restricted to python) alongside #1219 (TPU image pins);
   neither changes GPU scheduling, so the "upgrading would not help" conclusion stands.
4. **Image pinning** — separated CLI-source behaviour from formally documented metadata keys: the pinning
   keys are undocumented in `docs/kernels_metadata.md`, and no documented control forces the GPU image
   family; the server-assigned CPU-only image is consistent with a CPU session, not a missed documented step.

The reviewer also independently re-serialized the push request with the SDK's `to_json()` and confirmed no
wire-level field omission; that is cited as supporting evidence, not as a new probe.

## Evidence paths (sanitized, no tokens)

* `results/kaggle/dr/gpu_availability/` — probe JSONs, server-metadata snapshots, quota API response
* `results/kaggle/dr/attempt1_pi_setup_failure/` — the Pi-era run that failed on the input path
* Installed CLI/SDK sources cited above; request-field captures reproduced offline from
  `experiments/s6_kaggle_dr/package.py`-built kernel dirs (no submission)
