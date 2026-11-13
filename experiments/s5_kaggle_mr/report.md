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
`hasEverRun=False` while the weekly allowance was untouched (30 h GPU, 20 h
TPU). Script and notebook kernels, three zones, with and without the
`docker_image` pin, GPU and TPU - all CPU. Details and probe IDs are in
`results/kaggle/mr/run_log.md` (probe kernels
`phinguynxun/mpr-mr-mount-probe` versions 1-5,
`phinguynxun/mpr-mr-gpu-notebook-probe` v1, `phinguynxun/mpr-mr-gpu-probe2`
v1).

What the evidence establishes is only that Kaggle provisioned CPU-only
containers for every accelerator request on this account (`hasEverRun=false`,
no quota charged). Kaggle documents that free accelerators require one-time
phone verification and that push-based accelerator requests can be accepted
while the session runs in the standard image (kaggle-cli #1197); an account
hold or a platform fault would look the same from inside a kernel. The cause
is therefore *not* concluded here. Confirming it needs the account owner
(https://www.kaggle.com/settings, or Kaggle support if the account is already
verified).

## What runs once the accelerator works

```bash
# 1. local tests
python -m unittest tests.test_mr -v

# 2. push under the shared lock, poll to terminal status and download outputs;
#    the wrapper holds the flock on fd 9 until the script exits.  A bare
#    `flock <lock> kaggle kernels push ...` would release the lock as soon as
#    push returns (the kernel is only queued at that point), letting the peer
#    start while this job still runs.
experiments/s5_kaggle_mr/run_remote.sh results/kaggle/mr/_kaggle_out

# 3. score and compare with NR/SR
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
