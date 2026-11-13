# S6 - DecompositionReason (DR) on a Kaggle T4

DR is the paper's decomposition reasoning structure (Section 3.4.3, Figure 2(d)): a question is broken
into sub-questions, the sub-questions are answered one at a time with memory retrieval, and a final call
merges the answers. This folder runs upstream's `DecompositionReason` (ADAPTED) on the frozen 306-question
slice for the two retrievers the existing T4 sessions use (BM25/SparseRAG and e5/DenseRAG, k = 20), on one
private Kaggle T4 GPU job capped at four hours, then scores it against the archived one-call
(`results/t4/s1`) and five-step chain-of-thought (`results/t4/s2`) runs.

Deliverables: this README, `report.md` (final report after the job), `dr.py` (the DR runner),
`kernel.py` (the Kaggle entrypoint), `package.py` (private dataset/kernel packaging), `summarize_dr.py`,
`validate_dr.py`, `settings.json` (frozen settings), `tests/test_dr.py`, and the evidence under
`results/kaggle/dr/`.

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
7. **Runtime** environment: a Kaggle T4 (fp16/AWQ/Triton/Marlin as on Colab) instead of a Colab VM.
   The vLLM/torch versions are recorded in `manifest_out.json`; the historical Colab versions were not
   pinned, so small decoding differences are possible even with greedy decoding. A
   NaiveReason deployment sanity gate (below) is our guard against unexplained drift.

## (c) Frozen settings and seed policy

`settings.json` is the single source of truth, hashed into every result row (`settings_sha256`). It was
written before scoring and does not depend on any observed result. Key values: `decompose_num` 5, top-k
20, greedy, caps 200/200/64, parser `ref_faithful_drop_blank`, workers 32, slice s306, 612 expected rows,
smoke = 6 questions (2 at hops 2, 5 and 9), sanity gate thresholds, session deadline 14,400 s with
submission stop at 12,900 s and a watchdog exit at 13,800 s.

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

## (e) Infrastructure: one private Kaggle job, embedded gates

- Private dataset `phinguynxun/mpr-dr-inputs`: the 12 released `question_answer_*.json` files, the s306
  slice, `dr.py`, `lib.py`, `score.py`, `settings.json`, the archived S1 NaiveReason rows restricted to
  the slice (for the sanity gate), and `manifest.json` with a sha256 of every file plus the code commit,
  upstream commit and model revision. Only this explicit allowlist is uploaded; no `.git`, credentials,
  upstream sources or other agents' state.
- Private script kernel `phinguynxun/mpr-dr-reasoning`, `kaggle kernels push --timeout 9000
  --accelerator NvidiaTeslaT4`; internet on, needed for the pinned upstream clone and the model/e5
  downloads; one GPU replica (`CUDA_VISIBLE_DEVICES=0`), the second T4 (if allocated) is recorded and idle.
  The 2.5-hour cap is a shared-account constraint, not a method choice: the account has 6 GPU-hours per
  refresh and both workers serialize on the host GPU lock, so MR takes 3 h and DR 2.5 h (see (e2)).
- The kernel **discovers** the read-only input directory instead of assuming it (`kernel.py:
  discover_input_root`): it tries `MPR_INPUT_ROOT`, `/kaggle/input`, `/kaggle/input/<slug>` and
  `/kaggle/input/datasets/<owner>/<slug>`, then walks `/kaggle/input` to depth 4, retrying for 90 s, and
  writes the resolved path, candidate list and a bounded tree to `input_probe.json`. This is the fix for
  attempt 1 (see (e1)).
- The kernel stages, verifies every input hash, records `nvidia-smi`, installs `bm25s==0.2.9` and
  sentence-transformers/vLLM if absent (vLLM unpinned to match the image's torch; exact versions
  recorded), clones upstream at the pinned commit and **refuses to run if `prompts.py` does not hash to
  the audited file**, builds the e5 embedding cache before starting vLLM, serves the model with the
  historical flags, and then:
  - **divide gate**: one real `Decomposition_Divide` call must be non-empty and parse to 1-5 lines;
  - **DR smoke**: 6 questions (2 per hop group, hops 2/5/9) x 2 memories, written into the main result
    files because the settings are identical; each smoke row is checked for calls = 1+N+1, N sub-answers,
    top-20 retrievals for every solve and merge, non-empty answers, and prompts below `max_model_len`;
  - **deployment sanity gate**: NaiveReason is re-run on the same 6 questions for both memories and
    compared with the archived S1 rows. We require, per memory, at least 3/6 byte-identical responses and
    at least 5/6 questions with >= 18/20 identical retrieved ids. Failure stops the full DR run (an
    unexplained engine/model/retrieval drift would invalidate the NR/SR comparison);
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
/kaggle/input/mpr-dr-inputs`. Its evidence is preserved unedited in
`results/kaggle/dr/attempt1_pi_setup_failure/` (`DONE`, `status.json`, `manifest_out.json`,
`run_log.txt`, `kernel_log.json`). It produced **no** experiment rows and is not an evaluation.

Root cause, verified with the private CPU-only kernel `phinguynxun/mpr-dr-mount-probe`
(source `diagnostics/mount_probe.py`, output `results/kaggle/dr/mount_probe/mountprobe.json`,
2026-11-13, 0 GPU quota): the attached dataset is mounted at
`/kaggle/input/datasets/phinguynxun/mpr-dr-inputs/`, not `/kaggle/input/mpr-dr-inputs/`.
`/kaggle/input` contained exactly one entry, `datasets`. All 20 expected files were present with the
expected sizes, and `manifest.json` hashed to
`3fdfc315da42fe1d5e0d7c8b784959ac9ded014e47e6f42e09ebf531ab0a9b04`.

Fix (tested in `tests/test_kernel_stage.py`): `discover_input_root()` resolves the mount from explicit
candidates plus a bounded walk and retries, `stage_inputs()` publishes the probe and the resolved root to
`status.json`/`input_probe.json`, and it refuses a manifest that declares a different Kaggle dataset (so
the MR dataset can never be staged by mistake).

## (e2) Shared-account GPU budget: one 2.5-hour job, serialized with the MR worker

Both experiments run on one Kaggle account. `kaggle quota` on 2026-11-13 reported GPU `used 0.00h`,
`remaining 30.00h`, refresh 2026-11-14T00:00Z, while the API view (`quota_view().totalTimeAllowed`) used
by the MR worker reported 21600 s = 6 h. We budget against the stricter 6 h: MR may take 3 h and DR 2.5 h
(`--timeout 9000`), which fits with a 30-minute margin and cannot starve either job. The two jobs are
serialized by the host advisory lock `~/.local/state/mpr-study/kaggle-gpu.lock`
(`kaggle_run.sh`, which waits up to 4 h for it and holds it until its kernel is terminal and downloaded).
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

# 2. freeze/commit, then build the private inputs + manifest and upload them
python3 experiments/s6_kaggle_dr/package.py build
kaggle datasets create -p build/kaggle/dr_dataset --keep-tabular     # or: package.py dataset-version
kaggle kernels push -p build/kaggle/dr_kernel --timeout 9000 --accelerator NvidiaTeslaT4

# 3. monitor, then download the outputs (do not stop at "push")
kaggle kernels status phinguynxun/mpr-dr-reasoning
kaggle kernels output phinguynxun/mpr-dr-reasoning -p results/kaggle/dr/raw -o

# 4. validate and score locally (independent re-check against the released data)
python3 experiments/s6_kaggle_dr/validate_dr.py --runs results/kaggle/dr/runs
python3 experiments/s6_kaggle_dr/summarize_dr.py \
  --dr results/kaggle/dr/runs --slice results/t4/slices/s306.json \
  --nr results/t4/s1 --sr results/t4/s2 --out results/kaggle/dr/summary_dr.md
```

`package.py` refuses to build from a dirty tracked tree so the manifest's code commit matches the code
that ran. `summarize_dr.py` needs numpy (bootstrap only); run it with the repo venv if the system Python
has none.

## (g) Results

See [`report.md`](report.md) for the final status, the run's numbers, paired differences, costs, kernel
URL, commit hashes and caveats, and `results/kaggle/dr/` for the evidence (`runs/*.jsonl`, failures,
`validation_local.json`, `summary_dr.md`, `summary_dr.json`, `manifest_out.json`, `status.json`,
`sanity_naive.json`, `smoke_dr.json`, logs).
