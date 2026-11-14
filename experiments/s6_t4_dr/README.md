# S6 - DecompositionReason (DR) on a Colab T4

DR is the paper's decomposition reasoning structure (Section 3.4.3, Figure 2(d)): a question is broken
into sub-questions, the sub-questions are answered one at a time with memory retrieval, and a final call
merges the answers. This folder runs upstream's `DecompositionReason` (ADAPTED) on the frozen 306-question
slice for the two retrievers the existing T4 sessions use (BM25/SparseRAG and e5/DenseRAG, k = 20), on one
tesla-T4 GPU session capped at 2.5 h, then scores it against the archived one-call (`results/t4/s1`) and
five-step chain-of-thought (`results/t4/s2`) runs and against the peer MR experiment.

**Status: complete.** 612/612 rows on 2026-11-14 on a Colab free-tier Tesla T4 via `colab_run.sh`
(vLLM 0.11.2), keyed to code commit `eeda20c`; see `report.md` for the numbers. Every Kaggle accelerator
request on the shared account ran CPU-only (four probes), so the run went through the authenticated
Colab CLI — the same route the MR experiment used. The probe bundle and the `kaggle_run.sh` wrapper were
removed in the repository cleanup; `package.py` and the runtime behaviour it packages are retained, and
the historical `phinguynxun/...` names below document the blocked phase. The private resource IDs are
`xunphi/mpr-dr-{inputs,reasoning}`.

Deliverables: this README, `report.md` (final report), `dr.py` (the DR runner), `kernel.py` (the entrypoint,
unchanged on both routes), `package.py` (private dataset/kernel packaging), `colab_run.sh` (the proven Colab
T4 driver), `summarize_dr.py`, `validate_dr.py`, `compare_mr_dr.py`, `settings.json` (frozen settings),
`tests/`, and the evidence under `results/t4/s6_dr/`.

## (a) What we run, and the limits we accept

- 306 questions: `results/t4/slices/s306.json`, seed 0, 34 questions per hop (2-10), all 12 users,
  140 leak-free questions (the slice's `leak_any` flag). Unchanged, not resampled.
- Qwen/Qwen2.5-7B-Instruct-AWQ at revision `b25037543e9394b818fdfca67ab2a00ecc7dd641` (same as
  `experiments/_shared/t4.sh`), fp16 compute, `--max-model-len 8192`, `--gpu-memory-utilization 0.80`,
  `--max-num-seqs 64`, `--generation-config vllm`, served as `qwen`, greedy decoding, one T4 replica.
- DR settings: `decompose_num = 5` (upstream `config.py:39-42`), so 1 divide + up to 5 solve + 1 merge =
  **3 to 7 calls per question**, at most 7 x 306 x 2 = 4,284 calls. Caps: divide 200 tokens, solve 200,
  merge 64 (the historical NR/SR convention: 200-token intermediate steps, 64-token final answer).
- Retrieval: BM25 via `bm25s==0.2.9` with English stopwords and e5-base-v2 with normalised cosine,
  no `query:`/`passage:` prefixes, ordered statement ids, cached fp16 embeddings; top-k 20.
- Scope: the full 306 x 2 = 612 unique question/memory rows are the completion criterion. Ignoramus and
  Oracle are not part of the denominator (their NR/SR counterparts already exist); the Ignoramus prompt
  path is still mock-tested. No farm credentials are used; the only inference engine is the local vLLM
  server in the Kaggle kernel.

## (b) Reference audit (upstream `nuster1128/MPR` at `a24953c`)

Read at the pinned commit (the clone is gitignored; no code or template text is copied into this repo):

| Upstream file | Lines | What it does | Consequence for us |
|---|---|---|---|
| `reasonings/DecompositionReason.py` | 9-12 | **Divide**: `knowledge_model.inference('Decomposition_Divide', {'question', 'max_sub_question': decompose_num})`, then `.split('\n')[:decompose_num]` | no retrieval; no JSON list parser; no validation; the first five newline entries are the sub-questions |
| same | 17-32 | **Solve**: a Python loop over the sub-questions, in order | sequential, not parallel: later sub-questions see earlier answers |
| same | 18-26 | numbered `current_state`: earlier lines `'%d. %s (Answer: %s)'`, current `'%d. %s (Current Sub-question)'`, later `'%d. %s'` | we reproduce these exact strings |
| same | 27-31 | `knowledge_model.response('Decomposition_Solve', ..., ref, f'(question) {question}\n (sub-question) {sub_question}')` | retrieval key is the question **plus the current sub-question**, not the sub-question alone and not earlier answers; the single space before `(sub-question)` is literal |
| same | 34-40 | **Merge**: the answered state is rebuilt, then `response('Decomposition_Merge', ..., ref, current_state)` | merge *re-retrieves* using the answered state as the key; it is not a closed-book synthesis |
| `baselines/BaseMethod.py` | 41-43 | `RAGMethod.response` retrieves with the supplied `knowledge_key`, stores it as `param_dict['references']`, then calls `inference` | our DR `call` does the same and the resulting references are the prompt's `{references}` |
| same | 45-50 | `inference` formats `eval(f'{call_type}_RAG_Prompt_Template')` | the RAG prompt suffix is chosen by call type; we use `getattr` instead of `eval` |
| `baselines/Ignoramus.py` | 14-21 | `response` is `inference` only (no retrieval), formats `{call_type}_Ignoramus_Prompt_Template` | with no memory there are no retrieval calls at all; mock-tested |
| `baselines/SparseRAG.py` | 12, 14-26 | `topk` from `retrieval_config`; bm25s index over the user's statement texts with `stopwords="en"`; query tokenised the same way, top-k ids joined with newlines | matches `_shared/lib.UserIndex.sparse` |
| `baselines/DenseRAG.py` | 11-16, 18-24, 26-39 | e5-base-v2, `normalize_embeddings=True`, no prefixes, score `matmul`, sort descending, top-k, join | matches `_shared/lib.UserIndex.dense` and the cached embeddings |
| `baselines/prompts.py` | 28-35, 37-44 | `Decomposition_Divide_{RAG,Ignoramus}` (the two texts are identical) ask for 1 to `{max_sub_question}` concise sub-questions, one per line | divide prompt has **no** `{references}` placeholder, so the no-retrieval call formats cleanly |
| same | 47-63, 65-78 | `Decomposition_Solve_*`: question, numbered sub-questions, current sub-question, information (RAG only), one-line concise answer | we use the template as shipped, loaded from the clone at run time |
| same | 80-96, 98-111 | `Decomposition_Merge_*`: question, sub-questions and answers, information (RAG only), concise final answer with the date/number rules | same |
| `config.py` | 26 | `retrieval_topk = 20` | k = 20 |
| same | 39-42 | `Decomposition_default = {'reason_method': 'DecompositionReason', 'decompose_num': 5}` | N <= 5 |
| same | 66-80 | SparseRAG / DenseRAG defaults use `retrieval_topk` | k = 20 both |
| `utils.py` | 6, 71-119 | `LLM.run` for local Qwen models posts `messages` + `kwargs={}` to the model server (no temperature); `fast_run` retries until the response is truthy and returns the string `'None'` after five tries | the upstream decoding temperature for the local Qwen server is **not specified anywhere**; the historical NR/SR runs in this repo froze greedy decoding and we keep it. We also use 5 transport retries but treat a persistent failure as a failed question (never as `'None'` text) |
| `run_qa.py` | 36-37, 45, 47-60 | one user's data per run, `store(data['message'])`, loop over hops and questions, Oracle passes the gold `message` list as `ref`, output fields `k_hop`, `rc_id`, `response`, `ground_truth`, `fold_tag` | our slice covers all 12 users; the result rows keep those fields plus trace fields |

Semantics confirmed with executable mocks, not by reading alone (`tests/test_dr.py:UpstreamParityTests`):
the test imports the real upstream `DecompositionReason`, `RAGMethod`, `Ignoramus` and `NaiveReason`
classes, plugs in the same fake memory index and canned engine outputs as our runner, and asserts
element-wise equality of **every prompt string in order**, **every retrieval key in order**, the final
response, and the call count, for 2 and 5 sub-questions, for the RAG and the Ignoramus paths. The
upstream parser rule itself (`split('\n')[:5]`, including empty lines) is a separate assertion.

### Deliberate deviations from the reference (all pre-registered in `settings.json`)

1. **One slice across users, not one user's file** (`run_qa.py:36-37,45`): same as the historical S1-S3
   runs; the questions come from the frozen slice, and all 12 users' memories are loaded.
2. **Greedy decoding** (upstream local-server `kwargs={}` is unspecified): matches the historical NR/SR
   runs so that the comparison is decoding-matched. Divide, solve and merge are all greedy.
3. **Token caps** 200/200/64: upstream does not cap. 200 for intermediate text matches the
   SequentialReason thoughts, 64 for the final answer matches NR/SR; all three are recorded per call.
4. **Parser `ref_faithful_drop_blank`**: upstream `text.split('\n')[:decompose_num]` only; we additionally
   drop whitespace-only lines (counted per row, `parser.dropped_blank`) because they cannot be
   sub-questions, and if nothing non-blank remains we use the original question as the single
   sub-question (`parser.fallback = true`, 3 calls). Numbering and all other text are kept verbatim; no
   dedup, no repair, no length truncation. The raw divide response is stored in every row. The
   `upstream_strict` policy is kept in `dr.py` only for the parser mock; it is not used for the run.
   This is the one place where we do not claim exact parity when the model emits blank lines.
5. **Leading/trailing whitespace stripped from every engine response** (as `_shared/lib.py:chat` already
   does for NR/SR) instead of upstream returning the raw string.
6. **Bounded failure treatment**: a transport/engine failure after retries is written to
   `failures_<memory>.jsonl` with the stage and partial trace, the question is not written to the result
   file, and a rerun resumes it. There is no silent fallback to NaiveReason. Twenty consecutive failures
   trip a circuit breaker and stop the run. A run deadline stops submitting new questions ~25 minutes
   before the session timeout and drains in-flight calls.
7. **Runtime** environment: one T4 (16 GB, fp16/AWQ) reached through the Colab CLI rather than the Kaggle
   kernel launcher, because the account provisions no accelerator to Kaggle jobs (section (e)). One GPU
   replica; a second T4, if present, is idle. The vLLM/torch versions are recorded in `manifest_out.json`
   (0.11.2 / torch 2.9.0+cu128 for the completed run, the same pin that served the MR rows), and the
   historical vLLM 0.32.0 difference is bounded by the NaiveReason deployment sanity gate below
   (observed 5/6 byte-identical responses, 20/20 identical retrievals per memory).

## (c) Frozen settings and seed policy

`settings.json` is the single source of truth, hashed into every result row (`settings_sha256`). It was
written before scoring and does not depend on any observed result. Key values: `decompose_num` 5, top-k
20, greedy, caps 200/200/64, parser `ref_faithful_drop_blank`, workers 40, slice s306, 612 expected rows,
smoke = 6 questions (2 at hops 2, 5 and 9), sanity gate thresholds, and a session deadline of 9,000 s with
submission stop at 7,800 s and a watchdog exit at 8,400 s (the kernel reads `MPR_SESSION_TIMEOUT_S`, default
9,000 s; the completed Colab run used 2,720 s of it).

Seeds: DR is greedy, so there is no sampling seed. Seed 0 fixes only the deterministic smoke/sanity key
selection (slice order) and the bootstrap conventions used later (`summarize_dr.py`: seed 0 for
intervals, seed 1 for paired differences, 2,000 resamples). No hyperparameter is tuned on this evaluation
set.

## (d) Provenance

| File / function | Label | Reference |
|---|---|---|
| `dr.py: Runner.solve` (divide -> sequential solve -> merge) | ADAPTED | upstream `reasonings/DecompositionReason.py:9-42`; prompt suffixes from `baselines/BaseMethod.py:45-46` and `baselines/Ignoramus.py:17-21`; retrieval from `baselines/SparseRAG.py:21-26` / `baselines/DenseRAG.py:32-39`. Mock-checked prompt-for-prompt and key-for-key (README (b)). |
| `dr.py: parse_division` | ADAPTED, with a labelled deviation | upstream `.split('\n')[:decompose_num]` at `DecompositionReason.py:12`; `ref_faithful_drop_blank` drops blank lines and has one documented fallback (README (b) item 4) |
| `dr.py: build_current_state`, `build_answered_state`, `solve_retrieval_key`, `merge_retrieval_key` | ADAPTED | exact strings from `DecompositionReason.py:18-26, 31, 35-36, 40`; these are short format strings, verified by the parity mock |
| `dr.py: Runner.refs`, `Runner.user`, `Runner.format_prompt`, `Runner.call` | ADAPTED | `_shared/grid.py:Runner.refs/call/user` with the same memories; `baselines/BaseMethod.py:42`, `baselines/Ignoramus.py:14-21` |
| `dr.py: Runner.solve_naive` | ADAPTED | upstream `reasonings/NaiveReason.py:8`; used only for the deployment sanity check against archived S1 rows; mock-checked against upstream `NaiveReason` |
| `dr.py: Runner.run/run_item`, `unique_rows`, `ensure_trailing_newline`, CLI | ORIGINAL TO THIS REPO | no upstream reference (upstream has no runner, resume or deadline handling); the resume/jsonl idea follows `_shared/grid.py` / upstream `utils.py:58-61` |
| `kernel.py` | ORIGINAL TO THIS REPO | no upstream reference; Kaggle orchestration, gates, preflight, watchdog |
| `package.py` | ORIGINAL TO THIS REPO | no upstream reference; private dataset/kernel packaging and hash manifest |
| `summarize_dr.py`, `validate_dr.py` | ORIGINAL TO THIS REPO | no upstream reference; scoring conventions copied from `_shared/summarize.py` (itself ours), metrics from `_shared/score.py` |
| `settings.json` | ORIGINAL TO THIS REPO | values cited to upstream `config.py` where they come from it |
| `tests/test_dr.py` | ORIGINAL TO THIS REPO (test harness) | executes upstream classes read-only as the parity oracle |
| prompt templates | upstream, fetched at run time | `baselines/prompts.py:28-111`; never stored in this repo |

Shared files reused without modification: `experiments/_shared/lib.py` (data, prompts loader, jsonl
writer, vLLM chat, indexes, encoder) and `experiments/_shared/score.py` (EM/contains). Their sha256 is in
the run manifest; neither is edited by this branch.

## (e) Infrastructure: one GPU session, embedded gates, two possible routes

One driver runs the **committed `kernel.py`**: `colab_run.sh` (proven, used for the results). The
historical `kaggle_run.sh` wrapper was removed in the cleanup; it held the same shared host lock
`~/.local/state/mpr-study/kaggle-gpu.lock`, so a DR and an MR job could never overlap.

**Colab route (used).** `colab_run.sh` builds the packaged dataset from a *clean* git tree (so
`manifest.json` records the exact commit), tars it, creates a fresh `mpr-dr` session with `--gpu T4`, uploads
and unpacks it at `/content/dr_inputs`, checks `nvidia-smi -L`, pre-installs `vllm==0.11.2` (the pin that
actually served the MR rows on driver 580; the kernel's CUDA-generation choice would otherwise try 0.32.0
first and pay a bounded fallback), launches `kernel.py` detached with `MPR_INPUT_ROOT=/content/dr_inputs`
and `MPR_WORK=/kaggle/working`, polls the exit file/log every 60 s (pushing `hc` milestones on gate pass,
every 20 min and at exit), packs `/kaggle/working`, downloads it into `results/t4/s6_dr/raw`, and stops the
session on exit (`DR_COLAB_KEEP=1` keeps it). Colab mounts `/kaggle/input` read-only, which is exactly why
the kernel resolves `MPR_INPUT_ROOT` first and only then the Kaggle mount candidates.

**Kaggle route (blocked; probe wrappers removed, packaging code retained).**

- Private dataset `xunphi/mpr-dr-inputs` (built locally as `build/kaggle/dr_dataset`): the 12 released `question_answer_*.json` files, the s306
  slice, `dr.py`, `lib.py`, `score.py`, `settings.json`, the archived S1 NaiveReason rows restricted to
  the slice (for the sanity gate), and `manifest.json` with a sha256 of every file plus the code commit,
  upstream commit and model revision. Only this explicit allowlist is uploaded; no `.git`, credentials,
  upstream sources or other agents' state.
- Private script kernel `xunphi/mpr-dr-reasoning`, `kaggle kernels push --timeout 9000
  --accelerator NvidiaTeslaT4`; internet on, needed for the pinned upstream clone and the model/e5
  downloads; one GPU replica (`CUDA_VISIBLE_DEVICES=0`), the second T4 (if allocated) is recorded and idle.
  The 2.5-hour cap is a shared-account constraint, not a method choice: both workers serialize on the
  host GPU lock, so MR takes 3 h and DR 2.5 h (see (e2)). The Colab driver passes the same 9,000 s cap
  (`MPR_SESSION_TIMEOUT_S`); the completed run used 2,720 s of it.
- The kernel **discovers** the read-only input directory instead of assuming it (`kernel.py:
  discover_input_root`): it tries `MPR_INPUT_ROOT`, `/kaggle/input`, `/kaggle/input/<slug>` and
  `/kaggle/input/datasets/<owner>/<slug>`, then walks `/kaggle/input` to depth 4, retrying for 90 s, and
  writes the resolved path, candidate list and a bounded tree to `input_probe.json`. This is the fix for
  attempt 1 (see (e1)).
- The kernel stages, verifies every input hash, records `nvidia-smi`, installs `bm25s==0.2.9` and
  sentence-transformers/vLLM if absent, clones upstream at the pinned commit and **refuses to run if
  `prompts.py` does not hash to the audited file**, builds the e5 embedding cache before starting vLLM,
  serves the model with the historical flags, and then:
  - **vLLM pin is chosen from the GPU's CUDA generation** (`vllm_specs`), because it is not a free
    choice: 0.32.0 (the historical T4 version, `results/t4/timing_log.md`) depends on
    `torch==2.13.0` + `humming-kernels[cu13]` and therefore needs driver >= 580, while 0.11.2
    (`torch==2.9.0`) and 0.10.2 (`torch==2.8.0`) are CUDA-12 pins that work on driver 5xx. The driver
    major, every install attempt, the import probe of `vllm`/`torch` and `torch.version.cuda` are written
    to `deps.json`, and a pin is accepted only if a separate process can import both; the kernel fails
    loudly if no pin is importable. A CUDA-12 pin is a documented deviation from the historical
    deployment, not a silent one.
  - **divide gate**: one real `Decomposition_Divide` call must be non-empty and parse to 1-5 lines;
  - **DR smoke**: 6 questions (2 per hop group, hops 2/5/9) x 2 memories, written into the main result
    files because the settings are identical; each smoke row is checked for calls = 1+N+1, N sub-answers,
    top-20 retrievals for every solve and merge, non-empty answers, and prompts below `max_model_len`;
  - **deployment sanity gate**: NaiveReason is re-run on the same 6 questions for both memories and
    compared with the archived S1 rows. We require, per memory, at least 5/6 questions with >= 18/20
    identical retrieved ids and at least 3/6 byte-identical responses **on the historical vLLM 0.32.0**;
    when the installed engine is a different build (the CUDA-12 pins 0.11.2/0.10.2, see the vLLM bullet),
    the exact-response floor is 2/6 (`sanity_gates()`), because greedy text can differ by a token on
    another build while prompts, model revision and retrieval are unchanged and independently verified.
    Both versions and the effective thresholds are written to `sanity_naive.json`. Failure stops the full
    DR run (an unexplained engine/model/retrieval drift would invalidate the NR/SR comparison);
  - **runtime gate**: the full-run estimate from the smoke stage's token throughput (x1.3 safety) must fit
    in the remaining session budget, else the full run does not start;
  - **full run**, then validation (unique coverage, ground truth, row invariants) and EM/contains overall,
    per hop group and leak-free; `status.json`, `DONE`, `manifest_out.json` and `run.log` are written for
    download. A watchdog exits at +8,400 s and the runner stops submitting at +7,800 s so the output is
    committed before the 2.5-hour limit.
- Every result line is flushed and fsynced (shared `lib.JsonlWriter`); a cut line is ignored on resume and
  a missing final newline is repaired before appending (`dr.ensure_trailing_newline`). The job is one
  session: no retries, no replacement job without new user approval. A host advisory `flock`
  (`~/.local/state/mpr-study/kaggle-gpu.lock`) serialises this job with the peer's MR job on the shared
  Kaggle account.

## (e1) Attempt 1 failed in setup: the dataset was mounted under a different path

Attempt 1 (`phinguynxun/mpr-dr-reasoning`, Pi-era worker, 2026-11-12 17:58 UTC) was a script kernel that
died after 0.0 s of script time with `FileNotFoundError: manifest.json not found under
/kaggle/input/mpr-dr-inputs`. Its probe/status bundle was removed in the repository cleanup. It produced
**no** experiment rows and is not an evaluation.

Root cause, verified with the private CPU-only kernel `phinguynxun/mpr-dr-mount-probe`
(2026-11-13, 0 GPU quota; source and probe output were removed in the cleanup): the attached dataset is
mounted at `/kaggle/input/datasets/phinguynxun/mpr-dr-inputs/`, not `/kaggle/input/mpr-dr-inputs/`.
`/kaggle/input` contained exactly one entry, `datasets`. All 20 expected files were present with the
expected sizes, and `manifest.json` hashed to
`3fdfc315da42fe1d5e0d7c8b784959ac9ded014e47e6f42e09ebf531ab0a9b04`.

Fix (tested in `tests/test_kernel_stage.py`): `discover_input_root()` resolves the mount from explicit
candidates plus a bounded walk and retries, `stage_inputs()` publishes the probe and the resolved root to
the runtime `status.json`/`input_probe.json` files, and it refuses a manifest that declares a different
Kaggle dataset (so the MR dataset can never be staged by mistake).

## (e2) Budget: 2.5-hour cap, no Kaggle GPU quota used, serialized with the MR worker

**Observed for the completed run: 0.00 Kaggle GPU-hours and one Colab free-tier T4 session of 53 min
(2,720 s from kernel start to finish, 34 min of it the 612-row evaluation).** The numbers below describe
the Kaggle allowance the route would have used; they remain true and untouched.

Both experiments were planned on one Kaggle account. `kaggle quota` on 2026-11-13 reported GPU `used 0.00h`,
`remaining 30.00h`, refresh 2026-11-14T00:00Z, and the SDK quota object confirms 30 h/week
(`time_used` 0, `has_ever_run` false) — an earlier read of `timedelta.seconds` (21600 s) was mistaken for
a 6 h allowance; the correct value is 1 day + 6 h = 108000 s. MR may take 3 h and DR 2.5 h
(`--timeout 9000`), comfortably inside that allowance, and the two cannot starve each other. The two jobs are
serialized by the host advisory lock `~/.local/state/mpr-study/kaggle-gpu.lock` (the historical
`kaggle_run.sh` waited up to 4 h for it and held it until its kernel was terminal and downloaded).
The runtime gate inside the job still refuses to start the full evaluation when its own token-throughput
projection does not fit the remaining session, and the runner stops submitting 1200 s before the hard
limit, so a slow job yields partial rows instead of being killed mid-write. There is no automatic retry
and no replacement job.

## (f) How to run (reproduce)

```bash
# 0. prerequisites (already used for this run): pinned upstream clone + released dataset
git -C upstream/MPR rev-parse HEAD                      # a24953c8f342c9f3b2064e562ead49d477d49138
ls data/mpr_dataset/question_answer/question_answer_0.json   # gitignored data, see experiments/_shared/README.md

# 1. local checks (stdlib only; no GPU, no network; parity tests execute the upstream clone)
python3 -m unittest discover -s tests -v

# 2. one locked Colab T4 job: builds the dataset from a CLEAN tree, stages /content/dr_inputs,
#    pre-installs vllm 0.11.2, runs the unchanged kernel.py, downloads /kaggle/working into
#    results/t4/s6_dr/raw and stops the session (DR_COLAB_KEEP=1 keeps it for debugging)
experiments/s6_t4_dr/colab_run.sh

# 3. curate the downloaded rows, then validate and score locally (independent re-check)
#    (the raw bundle under results/t4/s6_dr/raw/ is gitignored; copy its runs/*.jsonl into runs/)
python3 experiments/s6_t4_dr/validate_dr.py --runs results/t4/s6_dr/runs
uv run --with numpy python3 experiments/s6_t4_dr/summarize_dr.py \
  --dr results/t4/s6_dr/runs --slice results/t4/slices/s306.json \
  --nr results/t4/s1 --sr results/t4/s2 --out results/t4/s6_dr/summary_dr.md

# 4. mutual comparison (DR vs the MR export, then the reverse direction)
uv run --with numpy python3 experiments/s6_t4_dr/compare_mr_dr.py \
  --a-dir results/t4/s6_dr/runs --a-label DR \
  --b-dir results/t4/s5_mr \
  --b-pattern 'StructuredReason_{mem}_k20_s5.jsonl' --b-label MR \
  --slice results/t4/slices/s306.json --out results/t4/s6_dr/summary_mr_vs_dr.md
# rerun with --a-* and --b-* swapped and --out results/t4/s6_dr/summary_mr_minus_dr.md

# 5. Kaggle alternative (still blocked on the account; no accelerator is provisioned).
#    package.py can stage/version the inputs, but the push/monitor wrapper was removed:
python3 experiments/s6_t4_dr/package.py build
kaggle datasets version -p build/kaggle/dr_dataset -m "DR inputs for <commit>" --keep-tabular
kaggle kernels push -p build/kaggle/dr_kernel --timeout 9000 --accelerator NvidiaTeslaT4
```

`compare_mr_dr.py` only needs the shared row schema (`uid`, `rc_id`, `k_hop`, `response`,
`ground_truth`), so it compares DR with any other reasoning method without assuming a call flow; it
drops questions missing on either side and reports the shared `n` for every paired row.

`package.py` refuses to build from a dirty tracked tree so the manifest's code commit matches the code
that ran. `summarize_dr.py` needs numpy (bootstrap only); run it with the repo venv or `uv run --with numpy`
if the system Python has none. `settings.json` deliberately keeps the historical
`"experiment": "s6_kaggle_dr"` label because every run row pins its SHA-256; renaming the folder did not
rewrite the 612 committed rows.

### UI-started Kaggle session (unused fallback)

On 2026-11-13 the account's API/CLI pushes (`kaggle kernels push --accelerator NvidiaTeslaT4`, script and
notebook, all request shapes) ran CPU-only while the server stored the GPU settings (see (e1); the probe
bundle was removed in the cleanup). **This path was not used: the completed run went through the Colab
CLI (f).** If a future interactive Kaggle session *does* get a T4, the same code can run there
without pasting it: the packaged dataset also carries `kernel.py` (identical to the pushed kernel's
`code_file`, hash in `manifest.json`). In the UI notebook, one cell is enough:

```python
import glob, shutil, subprocess, sys
src = glob.glob("/kaggle/input/**/kernel.py", recursive=True)[0]   # works for both mount layouts
shutil.copyfile(src, "/kaggle/working/kernel.py")
sys.exit(subprocess.call([sys.executable, "-u", "/kaggle/working/kernel.py"]))
```

Attach the private dataset `xunphi/mpr-dr-inputs`, enable internet, pick T4. The job self-caps at
9000 s (`SESSION_TIMEOUT_S`, watchdog exit 600 s before that, stop-submitting 1200 s before), writes
its runtime `status.json`/`DONE`/evidence to `/kaggle/working`, and the same validation/scoring commands
apply after downloading the output. Hold the shared GPU lock as the historical `kaggle_run.sh` did, and
keep the capped scope and hash checks unchanged.

## (g) Results (observed 2026-11-14, Colab T4, vLLM 0.11.2)

**612/612 rows complete, 0 failures, all gates and the independent validator passed.** Full report and
tables: [`report.md`](report.md); generated artifacts: `results/t4/s6_dr/`.

| run | n | EM [95% CI] | contains [95% CI] |
|---|---|---|---|
| DR SparseRAG | 306 | **0.131 [0.095, 0.167]** | 0.131 [0.095, 0.167] |
| DR DenseRAG | 306 | **0.118 [0.082, 0.157]** | 0.121 [0.085, 0.160] |
| DR SparseRAG, leak-free | 140 | 0.143 [0.086, 0.207] | 0.143 [0.086, 0.207] |
| DR DenseRAG, leak-free | 140 | 0.129 [0.079, 0.186] | 0.136 [0.086, 0.193] |

Paired differences (same 306 questions, DR minus baseline), all overall: DR is indistinguishable from
one-call NR (SparseRAG +0.016 [−0.023, +0.056]; DenseRAG −0.010 [−0.046, +0.026]) and below the 5-step SR
(SparseRAG −0.062 [−0.105, −0.016]; DenseRAG −0.105 [−0.154, −0.056]). Against MR (same slice, retrievers,
model revision and T4/vLLM pin; MR samples its branch candidates, DR is greedy): DR − MR is
**−0.072 [−0.118, −0.029]** EM on SparseRAG and **−0.098 [−0.147, −0.052]** on DenseRAG, with the reverse
direction committed as `summary_mr_minus_dr.md`. MR-side numbers reproduce MR's own `summary_mr.md`
exactly, and the reverse run mirrors the forward deltas/interval bounds to the last digit.

Observed budget: 3,602 calls (1,802 + 1,800), 2,234,418 prompt + 62,014 completion tokens, 2,051 s of
full-run wall at 40 concurrent questions (1.72-1.79 calls/s), 2,720 s kernel wall; 0.00 Kaggle GPU-hours.

Historical blocker summary (the probe bundles were removed from `results/t4/s6_dr/` in the cleanup):

- attempt 1 died in setup because the dataset was mounted at `/kaggle/input/datasets/<owner>/<slug>`, not
  `/kaggle/input/<slug>`; this is fixed in `kernel.py: discover_input_root`.
- the four private accelerator probes (two kernel types, three request shapes) all ran CPU-only with
  `enable_gpu: true` stored server-side, `torch 2.11.0+cpu`, no `/dev/nvidia*`, and `GPU used 0.00h` /
  `has_ever_run: false`. The Kaggle job was therefore not pushed; the Colab T4 route produced the rows.
