# Kaggle GPU documentation audit — current official sources vs our actual requests

Prepared by mpr-dr (t1g) as the sole primary investigator, 2026-11-13. Bounded research only: no probes,
no submissions, no provider execution, no Colab/auth flow. The main experiment remains **BLOCKED**.

## (a) Conclusion first

1. **Every material step in the current official GPU instructions is already what we sent.** The documented
   API path is: `kernel-metadata.json` with `machine_shape` (T4 = `NvidiaTeslaT4`, i.e. GPU T4 ×2) plus
   `enable_internet`, then `kaggle kernels push -p DIR --accelerator NvidiaTeslaT4 --timeout <s>`. Our five
   accelerator probes (script and notebook, three request shapes, including a fresh kernel pushed with
   `docker_image_pinning_type: latest`) serialized exactly those fields, and the runs executed.
2. **The documented behaviour is contradicted by what happened.** Official docs describe a GPU *queue* when
   capacity is tight; they do not document a silent CPU substitute. Our sessions went `RUNNING → COMPLETE`
   in ~20 s with no `/dev/nvidia*`, no `/proc/driver/nvidia`, no `nvidia-smi`, and `torch 2.11.0+cpu`
   (the CPU-only image), while `GET /api/v1/kernels/quota` reported GPU `timeUsed=0`,
   `totalTimeAllowed=108000` s, `hasEverRun=false`.
3. **No documented correction was found.** The one configuration that current docs do not explicitly cover
   (fresh kernel + `docker_image_pinning_type: latest`, no image pin) was in fact already exercised
   (probe 3) and also ran CPU-only. **No validation command is proposed** (see (f)); the cause remains
   unresolved from documentation alone.
4. **Correction to our own earlier narrative:** current official docs do **not** state that phone/identity
   verification is required to use Kaggle GPUs, and `hasEverRun=false` does not prove verification is
   missing. That hypothesis must be treated as unverified, not as the working conclusion. What the docs do
   establish is that some accelerators are restricted (competition participants/admins) and that free GPU
   capacity can be queued.

## (b) Sources, versions and retrieval times (all UTC, 2026-11-13)

| # | Source | Version / commit / date | Retrieved | What it establishes |
|---|---|---|---|---|
| 1 | Kaggle CLI docs `docs/kernels.md` — https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/kernels.md | `Kaggle/kaggle-cli` main `e3583ad834dc` (2026-11-12T21:23:54Z); latest release `v2.2.5` (2026-10-22T22:03:24Z) | 04:50:06Z | `kernels push` semantics, `--accelerator`, `--timeout`, `--no-run`; accelerator list "as of Sep 2026": `NvidiaTeslaT4` (GPU T4 ×2), `NvidiaTeslaA100`, `NvidiaL4`, `TpuV5E8`, `NvidiaL4X1`, `TpuV6E8`, `NvidiaH100`, `NvidiaRtxPro6000`; "Some of these are only available to participants of specific competitions, and some are only available to Kaggle admins." |
| 2 | Kaggle CLI docs `docs/kernels_metadata.md` — same repo | same commit | 04:50:06Z | Exact metadata keys; `machine_shape` "accelerator/GPU type (e.g. `NvidiaTeslaT4` for GPU T4 ×2, `NvidiaL4`, `TpuV5E8`)"; `enable_gpu` bool; `is_private` defaults true |
| 3 | Official notebook docs — https://www.kaggle.com/docs/notebooks.md | live page | 04:53:56Z | GPU is added in the editor: Settings → Accelerator → GPU; "Free GPU availability is limited: in busy times, you might be placed in a queue"; 12 h limit for CPU and GPU sessions; "T4 x2 GPU Specifications — 2 Nvidia Tesla T4 GPUs"; page still says "a single NVIDIA Tesla P100" (stale) |
| 4 | Official GPU-usage docs — https://www.kaggle.com/docs/efficient-gpu-usage.md | live page | 04:53:56Z | 30 h/week GPU quota ("resets weekly and is 30 hours or sometimes higher depending on demand"); recommends batch/API runs: "Consider using the Kaggle-API to avoid interactive sessions entirely"; discusses cancelling batch sessions |
| 5 | Official API docs — https://www.kaggle.com/docs/api.md | live page | 04:53:58Z | OAuth scopes incl. `kernels.execute`; no accelerator-specific API requirement documented |
| 6 | PyPI `kaggle` / `kagglesdk` | kaggle **2.2.5** (2026-10-22T22:01:21Z); kagglesdk **0.1.38** (2026-10-21T22:01:50Z); both installed versions match | 04:50:04Z | Installed client is current; the SDK request schema is the one the CLI serializes |
| 7 | GitHub commits since v2.2.5 | `src/kaggle/api/kaggle_api_extended.py`: 10 commits, latest `e3583ad` (2026-11-12); `src/kagglesdk/kernels/types/kernels_api_service.py`: 0 commits | 04:57:58Z | The only post-release change in the push path is `42f85b6` "clear stale image pins for TPU pushes" (#1219, 2026-10-26): it derives `enable_tpu` from the accelerator prefix and clears `docker_image` + notebook `dockerImageVersionId` **for TPU only**. It changes nothing for GPU pushes (`enable_tpu` stays false; `docker_image` still read from metadata), so installed 2.2.5 is equivalent for our case |
| 8 | Kaggle/docker-python README — https://raw.githubusercontent.com/Kaggle/docker-python/main/README.md | main, tags `v173-{CPU,GPU,TPU}-…` | 04:55:10Z | Two image families: CPU-only `gcr.io/kaggle-images/python`, GPU `gcr.io/kaggle-gpu-images/python`; our runs used the CPU-only digest and `torch 2.11.0+cpu` |
| 9 | Official issue tracker (supporting only) — Kaggle/kaggle-cli | #1197 closed (2026-09-17), #1219 merged (2026-10-15), #1151/#1156 closed (P100 + image), #1196/#1198 closed (T4 = GPU T4 ×2), #490 open since 2023, #907/#1029 (CLI `--gpu`, `kaggle quota`) | 04:55:2xZ | Maintainers confirmed a pinned CPU/GPU `docker_image` can override accelerator provisioning (TPU case) and auto-clear stale pins **for TPU pushes only**; docs clarified `NvidiaTeslaT4` = GPU T4 ×2 |
| 10 | Official service status | no official endpoint found: `status.kaggle.com` does not resolve; `kaggle.com/status` is a user profile | 04:52Z | No documented outage source to check |

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

Precedence and behaviour, as established by the installed CLI source (2.2.5; the only post-release push change is #1219, TPU image pins only):
`--accelerator` overrides `machine_shape`; `enable_gpu`/`enable_tpu` are read from metadata and are
deprecated; `--timeout` becomes `session_timeout_seconds`; `--no-run` sets `QUICK_SAVE` and **does not
execute**; a plain push sets `KERNEL_EXECUTION_TYPE_UNSPECIFIED` (server default; executes, as our runs did);
`docker_image` is optional and `docker_image_pinning_type` accepts `original|latest`.

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
| image pinning optional; stale pins can break accelerators (issue #1197 for TPU) | probe 3: **fresh kernel**, no `docker_image`, `docker_image_pinning_type: latest` | yes (strongest form) |
| timeout ≤ global maximum | 9000 s requested on the DR kernel, no timeout on probes | yes |
| API/batch runs are the recommended GPU path | used `kaggle kernels push` throughout | yes |
| Busy periods → queue | sessions never queued: `QUEUED → RUNNING → COMPLETE` in ~20 s | **contradicted** |
| Accelerators may be restricted to some accounts | docs say this only for "some" of the list; T4 is described as the default GPU; no documented account requirement | **unknown** |

Runtime outcome after those exact requests (existing sanitized artifacts, same evidence base as
`results/kaggle/dr/gpu_availability/`): all probes `COMPLETE`; `nvidia_smi_L.rc=127` ("not found"),
`devices=[]`, no `/proc/driver/nvidia/version`, `torch 2.11.0+cpu`, `cuda_available=false`, `device_count=0`;
`kaggle quota` GPU `used 0.00h`; quota API `timeUsed=0`, `hasEverRun=false`. A DR attempt with the same
metadata reached setup and failed only on the (now-fixed) input path, so execution itself works.

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
   The newest unreleased CLI change (#1219) only affects TPU image pins, so upgrading would not change a GPU request.
2. The documented failure modes are: queueing when capacity is tight; accelerator shapes restricted for
   some accounts; stale image pins overriding accelerator provisioning (documented for TPU, auto-cleared
   only for TPU by #1219).
3. No current official document states that GPU use requires phone/identity verification, and none documents
   a silent CPU fallback for a correctly requested accelerator.
4. The absence of device nodes and driver inside the container means the session had no GPU attached; it is
   not merely "an unused GPU".
5. Our requests match the documented ones field-for-field; therefore "we forgot a parameter" is not
   supported by the evidence.

**Unknown (and not claimed):**

* Why the backend created CPU sessions for correctly serialized GPU requests. Candidates, all unproven:
  an account-level eligibility flag not documented or exposed by the API; a platform/zone provisioning
  problem with silent downgrade; a backend bug specific to batch sessions on this account. Phone
  verification is **not** established as the cause by any source read here, and `hasEverRun=false` is not
  evidence about verification.
* Whether the interactive UI path (Settings → Accelerator → GPU) allocates a device on this account — that
  is the documented user-facing path and the only officially described way to select a GPU that we have not
  exercised; it requires a signed-in browser session.
* Whether `docker_image_pinning_type: latest` selects the GPU image for a GPU shape (its documented meaning
  is only "which docker image to use for executing new versions").

**Rejected as undocumented fixes (do not repeat as new evidence):** pushing the same T4 shape again; adding
`enable_gpu`; switching script↔notebook; `docker_image_pinning_type: latest`; L4/L4X1 shapes; TPU shapes —
all were exercised or are irrelevant to the observed deviceless runtime.

## (f) Proposed minimal validation command

**None is proposed.** No genuine documented correction exists beyond what was already executed, so another
capped probe would only repeat a documented request. The single untested documented path is the interactive
UI accelerator selection (source 3), which is an account-owner action, not a kernel command; if a UI session
shows `nvidia-smi` output, the API path can then be re-tested under the same capped scope.

If the account owner or Kaggle support can state which account eligibility flag is involved, the minimal
private check that would settle it is a ≤60 s script kernel whose only job is to print
`/dev/nvidia*`, `/proc/driver/nvidia/version`, `nvidia-smi -L`, `torch.cuda.is_available()` — i.e. exactly
the probe already recorded in `results/kaggle/dr/gpu_availability/`; it needs no new code and is not
proposed for execution here.

## Evidence paths (sanitized, no tokens)

* `results/kaggle/dr/gpu_availability/` — probe JSONs, server-metadata snapshots, quota API response
* `results/kaggle/dr/attempt1_pi_setup_failure/` — the Pi-era run that failed on the input path
* Installed CLI/SDK sources cited above; request-field captures reproduced offline from
  `experiments/s6_kaggle_dr/package.py`-built kernel dirs (no submission)
