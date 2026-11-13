# S5 MR - report

Status: **BLOCKED at Kaggle accelerator provisioning** (2026-11-13). No
evaluation rows exist yet; nothing below is presented as a result.

## What is complete

- Reference audit and the reference-faithful `StructuredReason` runner (see
  `README.md`), including the append-only buffer semantics, 12 calls/question
  and the safe selector.
- The private Kaggle entrypoint, packager and analyzer.
- Packaged private dataset `phinguynxun/mpr-mr-inputs` v1: 24 allowlisted
  files, all hashes re-verified against `input_manifest.json` at commit
  `c1ad941`.
- Mount-layout fix: the dataset is mounted at
  `/kaggle/input/datasets/phinguynxun/mpr-mr-inputs`, not the flat
  `/kaggle/input/mpr-mr-inputs`; `kernel.py` now resolves both layouts, walks
  `/kaggle/input` as a fallback and retries for 90 s.
- `check_cuda()` fails fast with a clear message if the installed torch cannot
  use a GPU.
- 18 local stdlib tests pass.

## Why it is blocked

Kaggle accepts `enable_gpu: true` + `machine_shape: NvidiaTeslaT4` (and
`enable_tpu: true` + `TpuV5E8`) but executes every run on CPU: no
`/dev/nvidia*`, no driver, `torch 2.11.0+cpu`, no `/dev/accel*`; the GPU quota
meter never moves (`used 0.00h`) and the internal quota object reports
`hasEverRun=False`. Script and notebook kernels, three zones, with and without
the `docker_image` pin, GPU and TPU - all CPU. Details and probe IDs are in
`results/kaggle/mr/run_log.md`.

Kaggle requires one-time phone verification on the account before accelerators
are available. This account (`phinguynxun`) has never had an accelerator
session. Resolving it needs the account owner: verify at
<https://www.kaggle.com/settings> (phone verification), or open a Kaggle
support ticket if the account is already verified and the accelerator is still
not provisioned.

## What runs once the accelerator works

```bash
# 1. local tests
python -m unittest tests.test_mr -v

# 2. hold the shared GPU lock for the whole job, then push the one 3 h job
flock /home/xphi/.local/state/mpr-study/kaggle-gpu.lock \
  ~/.local/bin/kaggle kernels push -p experiments/s5_kaggle_mr \
  --accelerator NvidiaTeslaT4 --timeout 10800

# 3. monitor to terminal status, then download
~/.local/bin/kaggle kernels status phinguynxun/mpr-mr-reasoning
~/.local/bin/kaggle kernels output phinguynxun/mpr-mr-reasoning \
  -p results/kaggle/mr/_kaggle_out

# 4. score and compare with NR/SR
python experiments/s5_kaggle_mr/analyze_mr.py --results-dir results/kaggle/mr \
  --slice results/t4/slices/s306.json \
  --data-dir ../mpr-study/data/mpr_dataset/question_answer
```

The kernel runs its own NR sanity check, MR smoke across hop groups and then
the full 306 x {BM25, e5} evaluation, or stops at the failing gate. The
`report.md` will be rewritten with the real counts, scores and paired
comparisons after that run.

## Reproducibility notes

- Frozen settings: `README.md` (b). Method/deviation labels: `README.md` (d).
- Reference: MPR `a24953c8...`, `reasonings/StructuredReason.py`; no upstream
  source or prompt text is stored in this repository.
- Comparisons with NR (s1) and SR (s2) will not be decoding-matched: only the
  two branch-generation calls per middle round are sampled (T=0.7, seeded),
  selection/start/final answer are greedy, while the historical baselines were
  greedy throughout.
