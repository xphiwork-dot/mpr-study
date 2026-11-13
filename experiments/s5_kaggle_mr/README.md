# S5 - MR (StructuredReason) on a private Kaggle T4

This experiment runs the paper's **multi-branch reasoning** structure (MR,
upstream class `StructuredReason`) with the two retrieval baselines the other
sessions use: BM25 (`SparseRAG`) and e5-base-v2 (`DenseRAG`), k = 20, on the
frozen 306-question slice (`results/t4/slices/s306.json`). It is a new,
experiment-local implementation; the NR/SR path and shared helpers are not
modified.

Settings were frozen and written down here **before** any scored run. Results
and the final status are in [`report.md`](report.md); raw rows are under
`results/kaggle/mr/`.

## (a) Reference audit (pinned upstream, no code copied)

Upstream: <https://github.com/nuster1128/MPR> at
`a24953c8f342c9f3b2064e562ead49d477d49138` (inspected read-only; the clone is
gitignored and not shipped). The upstream repository has **no licence**, so no
upstream code or prompt text is stored in this repository. Prompt templates are
loaded at runtime from the pinned clone.

| Reference | What was verified |
|---|---|
| `reasonings/StructuredReason.py:22-53` | `solve_qa` is the MR flow. Start at 26-30, three middle rounds at 32-45, final selection/answer at 47-53 for `max_step=5`. |
| `reasonings/StructuredReason.py:33` | `range(1, max_step-1)` gives **three** middle rounds at `max_step=5`, not five. |
| `reasonings/StructuredReason.py:34-35` | Each middle round first selects a path from the **entire current buffer**; selection is `knowledge_model.inference` only (no retrieval, no reference block). |
| `reasonings/StructuredReason.py:37-44` | Exactly `branch_size` candidate continuations of the selected path; each uses `previous_thoughts` formatted by `(Step %d) %s`, and retrieves with `current_path[-1]` (line 41). |
| `reasonings/StructuredReason.py:45` | Buffer update is `path_buffer[:pid] + path_buffer[pid:] + new_path_list`, i.e. the whole old buffer **including the selected path**, then the two new paths are appended. It does not replace the selected path, prune, or force the selected path to grow. A shallow retained path can be selected again. |
| `reasonings/StructuredReason.py:7-20` | Selection prompt lists every path as `Path %d: (Step 1) ... (Step 2) ...` (10). `res.isdigit()` + `eval` (13-14); non-digit or out-of-range falls back to path **0** (15-20). |
| `config.py:49-53` | `Structured_default`: `max_step = 5`, `branch_size = 2`. |
| `baselines/BaseMethod.py:41-43` | A `response` call retrieves first with `knowledge_key`, then adds the joined references and calls `inference`; `inference` formats `{call_type}_RAG_Prompt_Template` (46). |
| `baselines/prompts.py:208-219, 231-245, 260-277, 295-305` | Exact `Structured_Start`, `Structured_Middle`, `Structured_Final`, `Structured_Selection` RAG templates. Selection has no `{references}` placeholder, confirming inference-only. |
| `baselines/SparseRAG.py:14-26`, `baselines/DenseRAG.py:22-39` | BM25 with English stopwords; e5 `normalize_embeddings=True`, **no** `query:`/`passage:` prefixes, cosine top-k. |
| `utils.py:90-96, 110-119` | Hosted-client path samples at `temperature=0.9`; `fast_run` retries empty results and returns `'None'` after five tries. These decoding settings are not part of the local-server path. |
| `run_qa.py:47-60` | Evaluation iterates all hops/questions and records `k_hop`, `rc_id`, `response`, `ground_truth`. |

Consequences for the reference-faithful primary run:

- At the default settings a question makes
  `1 start + 3*(1 selection + 2 candidates) + 1 final selection + 1 answer = 12`
  model calls. `tests/test_mr.py` checks this against the upstream class with a
  mocked knowledge model, including every formatted prompt and retrieval key.
- The nominal total for the full evaluation is
  `306 questions x 2 retrievers x 12 = 7,344` calls, before retries and smoke.
- The maximum selected-path depth is 4 (the final answer is not appended to a
  path); shallow paths are legal and are recorded, not "fixed".

## (b) Frozen settings

| Item | Value |
|---|---|
| Evaluation slice | `results/t4/slices/s306.json`, seed 0, 306 questions, 34 per hop (2-10), all 12 users, 140 leak-free (any-number rule), 180 leak-free (3+ digit rule); SHA-256 `2e7b307bdcef4298b885305e0d56c837298de3136a3253f9c5d1941da3f8e831` |
| Model | `Qwen/Qwen2.5-7B-Instruct-AWQ`, revision `b25037543e9394b818fdfca67ab2a00ecc7dd641`, served name `qwen`, fp16 compute, AWQ, nominal max model length 8192, `--gpu-memory-utilization 0.80`, `--max-num-seqs 64` |
| Retrieval | BM25 `bm25s==0.2.9` English stopwords; e5-base-v2 revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, fp16, normalized cosine, no `query:`/`passage:` prefixes; k = 20; ordered statement ids; cached float16 embeddings |
| MR | `max_step=5`, `branch_size=2`, three middle rounds, selection from the whole buffer, append-only buffer exactly as upstream |
| Thought/answer caps | 200 max new tokens for start/middle candidates; 64 for selections and the final answer |
| Greedy calls | start, all three round selections, final selection, final answer: `temperature=0`, `top_p`/`seed` omitted |
| Branch calls | only the two `Structured_Middle` candidates per round: `temperature=0.7`, `top_p=1.0`, stable per-(question, round, branch) seed derived from experiment seed 0 by SHA-256 (`stable_seed`), independent of thread order |
| Empty responses | bounded retry: 3 attempts for non-selection calls then fail the question; selection falls back to path 0 after 3 attempts, like upstream's `'None'` |
| Selection parser | strip surrounding whitespace, ASCII digits only, safe `int`, out-of-range/non-digit -> path 0; no `eval` |
| Other | no gold answer is passed to the runner; ground truth is attached after the answer returns |
| Seeds | experiment seed 0; per-question seeds are derived from `uid`, `rc_id`, round and branch, not from scheduling order. No accuracy-driven tuning or unbounded retry-until-distinct. |
| Hardware budget | exactly one **private** Kaggle GPU job, `NvidiaTeslaT4` (T4 x2 shape), CLI timeout 14,400 s (4 h, the approved cap; the account has a 30 h/week GPU allowance) with a 30 min in-job reserve; embedded smoke + full run in that one job; serialized with the peer through `/home/xphi/.local/state/mpr-study/kaggle-gpu.lock`. 64 workers match `--max-num-seqs 64`. A `check_cuda()` gate fails fast if the allocated torch cannot see a GPU. |
| Software | vLLM pinned `0.32.0` (CUDA 13 wheels, matching the prior T4 session) when the node reports driver >= 580, otherwise `0.11.2` (CUDA 12 wheels) for 525 <= driver < 580, and the kernel refuses to run below 525 (`choose_vllm_pin`, tested). `bm25s==0.2.9`, `sentence-transformers==6.3.0`; actual resolved versions are recorded in the run manifest and `environment.json` |
| Expected rows | 306 unique successful rows per retriever = **612** total; any less is partial/blocked |

### Why branch calls are sampled

The two candidate calls in a round have identical prompts: `branch_id` is not
used in `StructuredReason.py:37-41`, and both use the same `question`,
`previous_thoughts`, retrieval key and references. With greedy decoding they
would be exact duplicates and MR would collapse to a beam of copies. Upstream's
hosted path samples at temperature 0.9 (`utils.py:92-96`), but its local-server
settings are not shipped. We therefore use seeded sampling for branch generation
only, as approved. This means MR-vs-NR and MR-vs-SR comparisons are **not
decoding-matched**, even though model, slice, retrievers and k match; no claim is
made that extra branches alone cause a difference.

### Pre-registered gates and endpoints

Before scoring, the following are fixed:

- **NR deployment sanity gate** on the 3 fixed smoke questions (one per hop
  group) x 2 retrievers: no empty answers, EM no more than 0.34 below the
  committed one-call s1 rows on these 6 cases, and no wild answer-length drift.
  Exact-response agreement is reported, not gated (vLLM versions may change it).
- **MR smoke gate** on the same 3 questions x 2 retrievers: exactly 6 unique
  rows, 12 calls each, no empty non-selection response, non-empty retrieval ids,
  selectors not all falling back, and not all branch candidate pairs exact
  duplicates. Smoke rows are part of the final 306 rows and are resumed, not
  rerun.
- **Full-run gate**: exactly 306 unique successful rows per retriever, all keys
  equal to the frozen slice, ground truth equal to the released QA answer, hop
  equal to the slice, and 12 calls per row.
- **Primary metrics** (existing `experiments/_shared/score.py`): EM and lenient
  `contains`, overall, hop groups 2-4 / 5-7 / 8-10, and leak-free subsets; 95%
  percentile bootstrap over questions, 2,000 resamples, fixed seeds; paired
  bootstrap on identical question keys for MR minus NR and MR minus SR.
- **Secondary/exploratory**: MR BM25 vs e5, per-hop traces, duplicate branch
  rate, selection fallback rate, selected depths, truncations. No multiple-
  comparison correction; one slice and one seed.

## (c) Files and flow

| File | Role |
|---|---|
| `mr.py` | ADAPTED reference flow, safe selector, retrieval wiring, resumable runner, smoke/sanity/coverage gates |
| `kernel.py` | ORIGINAL TO THIS REPO private Kaggle entrypoint: input verification, pinned clone, driver-aware vLLM install, CUDA gate, embedding cache, vLLM, sanity, smoke, full run, manifest/package. Also shipped inside the private dataset so a UI-started notebook can exec the same code |
| `pack_mr.py` | ORIGINAL TO THIS REPO explicit-allowlist private dataset staging (flat: Kaggle CLI default dir-mode=skip drops directories) |
| `run_remote.sh` | ORIGINAL TO THIS REPO push/monitor/download wrapper that holds the shared GPU flock (fd 9) until terminal status and output download finish |
| `monitor_remote.sh` | ORIGINAL TO THIS REPO same lock/poll/download wrapper without a push, for a version started from the Kaggle UI |
| `kernel-metadata.json` | private kernel metadata (`phinguynxun/mpr-mr-reasoning`), private dataset `phinguynxun/mpr-mr-inputs` |
| `analyze_mr.py` | ORIGINAL TO THIS REPO scoring/comparison report from raw rows |
| `../../tests/test_mr.py` | stdlib tests: upstream mock parity, buffer semantics, parsing, seeds, duplicates, no gold leak, cut-tail resume, reused grid regression |

`kernel.py` resolves the dataset root (`/kaggle/input/datasets/<owner>/<slug>`
or the legacy flat `/kaggle/input/<slug>`, then a bounded walk, 90 s retry) and
stages in this order: verify private inputs -> clone upstream at the
pinned commit -> install pinned packages -> build e5 embedding cache for all 12
users -> start one vLLM replica on GPU 0 -> NR sanity -> MR smoke -> full
306 x 2 -> coverage/manifest/package. If the T4 x2 shape really exposes two
GPUs, retrieval/query encoding uses GPU 1; otherwise it uses CPU. vLLM is always
one replica on one GPU, never tensor-parallel across two 16 GB cards.

## (d) Provenance

Labels follow [`../README.md`](../README.md).

| Script / function | Label | Reference |
|---|---|---|
| `mr.py: MRRunner.solve`, `_select`, `_generate` | ADAPTED | Upstream `reasonings/StructuredReason.py:22-53`; prompt parameter names and path formatting from lines 10, 34-44, 51-53; retrieval keys from lines 29, 41, 50-53. Deliberate differences are the safe parser, bounded empty retry, call caps and seeded branch sampling, all listed in (b). Checked by `tests/test_mr.py` against the upstream class with a mock model: identical 12-call sequence, formatted prompts and retrieval keys. |
| `mr.py: stable_seed`, `parse_selection` | ORIGINAL TO THIS REPO | No upstream reference for seeded sampling; safe parser replaces `isdigit`/`eval` at `StructuredReason.py:13-20`. |
| `mr.py: MemoryStore`, `prepare_embeddings` | ADAPTED | `baselines/SparseRAG.py:14-26`, `baselines/DenseRAG.py:22-39`; ordered ids and cached embeddings follow `experiments/s1_t4_one_call_grid/retrieve.py`. |
| `mr.py: make_row`, `cmd_run`, resume/timing/failures | ORIGINAL TO THIS REPO | Driver/result fields follow `run_qa.py:53-60` and `experiments/_shared/grid.py`; no upstream analysis code exists. |
| `kernel.py`, `pack_mr.py`, `analyze_mr.py` | ORIGINAL TO THIS REPO | No upstream reference. |
| `experiments/_shared/lib.py`, `score.py` | reused unchanged | See `../_shared/README.md`; `tests/test_mr.py` regression-checks the NR/SR call counts. |

## (e) How to run

Local, no GPU:

```bash
python -m unittest tests.test_mr -v
python experiments/s5_kaggle_mr/pack_mr.py --out /tmp/mpr-mr-kaggle/dataset --clean
```

Private Kaggle resources (only after local tests pass):

```bash
# 1. create/update the private input dataset (allowlisted files only)
~/.local/bin/kaggle datasets create -p /tmp/mpr-mr-kaggle/dataset
# or, after a change: kaggle datasets version -p ... -m "..."
# the staged folder is flat on purpose; pass -r only if you change that layout

# 2. push under the shared GPU lock and keep the lock until the run is terminal
#    and the outputs are downloaded; run_remote.sh holds fd 9 on the lock for
#    its whole lifetime, unlike `flock <lock> kaggle kernels push` which
#    releases as soon as push returns
experiments/s5_kaggle_mr/run_remote.sh results/kaggle/mr/_kaggle_out

# alternative when only the Kaggle UI can allocate a GPU: open the saved
# version, select Accelerator = GPU T4 x2 and start it, then monitor/download
experiments/s5_kaggle_mr/monitor_remote.sh phinguynxun/mpr-mr-reasoning results/kaggle/mr/_kaggle_out

# 3. inspect later if needed
~/.local/bin/kaggle kernels status phinguynxun/mpr-mr-reasoning
~/.local/bin/kaggle kernels logs phinguynxun/mpr-mr-reasoning

# 4. score locally (numpy required)
python experiments/s5_kaggle_mr/analyze_mr.py \
  --results-dir results/kaggle/mr --slice results/t4/slices/s306.json \
  --data-dir ../mpr-study/data/mpr_dataset/question_answer
```

The kernel slug, dataset slug and versions used for the actual run are recorded
in `report.md` and `results/kaggle/mr/manifest.json`.
