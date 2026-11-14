# NR vs SR on the s306.json slice

NR: `results/t4/s1` with `NaiveReason_{mem}_k20.jsonl`; SR: `results/t4/s2` with `SequentialReason_{mem}_k20_s5.jsonl`. Both sides use `experiments/_shared/score.py` (EM, lenient contains), 2000 bootstrap resamples (seed 0 intervals, seed 1 paired differences), hop groups 2-4/5-7/8-10 and the leak-free subset `leak_any == false` (140 of 306 questions). Paired rows drop questions missing on either side and report their `n`.

> Cross-check: these NR/SR point estimates reproduce the archived documentation exactly — SR BM25 EM 0.193,
> SR e5 0.222, paired SR-NR +0.078/+0.095 overall and +0.057/+0.093 leak-free
> (`experiments/s2_t4_chain_of_thought/README.md`), with the sign flipped here because this table is
> NR minus SR. CIs differ by at most 0.003 (different bootstrap implementation). This is the same code
> path that will score DR, run on real rows, so the tool is verified beyond synthetic fixtures.

## Coverage

| method | memory | rows | expected | missing | extra | duplicates ignored |
|---|---|---|---|---|---|---|
| NR | SparseRAG | 900 | 306 | 0 | 594 | 0 |
| NR | DenseRAG | 900 | 306 | 0 | 594 | 0 |
| SR | SparseRAG | 306 | 306 | 0 | 0 | 0 |
| SR | DenseRAG | 306 | 306 | 0 | 0 | 0 |

### NR on all slice questions

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| NR SparseRAG | 306 | 0.114 [0.082, 0.150] | 0.134 [0.098, 0.173] | 0.186 (n=102) | 0.069 (n=102) | 0.088 (n=102) |
| NR DenseRAG | 306 | 0.127 [0.092, 0.167] | 0.154 [0.114, 0.193] | 0.186 (n=102) | 0.088 (n=102) | 0.108 (n=102) |

### NR on the leak-free subset

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| NR SparseRAG | 140 | 0.136 [0.086, 0.193] | 0.150 [0.093, 0.214] | 0.207 (n=87) | 0.031 (n=32) | 0.000 (n=21) |
| NR DenseRAG | 140 | 0.143 [0.086, 0.200] | 0.171 [0.114, 0.236] | 0.195 (n=87) | 0.031 (n=32) | 0.095 (n=21) |

### SR on all slice questions

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| SR SparseRAG | 306 | 0.193 [0.150, 0.242] | 0.199 [0.154, 0.248] | 0.255 (n=102) | 0.157 (n=102) | 0.167 (n=102) |
| SR DenseRAG | 306 | 0.222 [0.173, 0.271] | 0.239 [0.190, 0.288] | 0.314 (n=102) | 0.147 (n=102) | 0.206 (n=102) |

### SR on the leak-free subset

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| SR SparseRAG | 140 | 0.193 [0.129, 0.264] | 0.193 [0.129, 0.264] | 0.253 (n=87) | 0.062 (n=32) | 0.143 (n=21) |
| SR DenseRAG | 140 | 0.236 [0.171, 0.307] | 0.264 [0.200, 0.336] | 0.322 (n=87) | 0.062 (n=32) | 0.143 (n=21) |

## Paired differences (NR minus SR)

| memory | subset | n | dEM [95% CI] | dContains [95% CI] |
|---|---|---|---|---|
| SparseRAG | overall | 306 | -0.078 [-0.124, -0.033] | -0.065 [-0.111, -0.020] |
| SparseRAG | leak_free | 140 | -0.057 [-0.121, 0.007] | -0.043 [-0.114, 0.021] |
| SparseRAG | 2-4 | 102 | -0.069 [-0.147, 0.020] | -0.059 [-0.147, 0.029] |
| SparseRAG | 5-7 | 102 | -0.088 [-0.157, -0.020] | -0.078 [-0.167, 0.000] |
| SparseRAG | 8-10 | 102 | -0.078 [-0.167, 0.000] | -0.059 [-0.137, 0.020] |
| DenseRAG | overall | 306 | -0.095 [-0.137, -0.049] | -0.085 [-0.127, -0.042] |
| DenseRAG | leak_free | 140 | -0.093 [-0.157, -0.029] | -0.093 [-0.157, -0.029] |
| DenseRAG | 2-4 | 102 | -0.127 [-0.206, -0.049] | -0.118 [-0.196, -0.039] |
| DenseRAG | 5-7 | 102 | -0.059 [-0.137, 0.020] | -0.059 [-0.137, 0.020] |
| DenseRAG | 8-10 | 102 | -0.098 [-0.176, -0.029] | -0.078 [-0.157, -0.010] |

## Caveats

- Two methods measured once each on one slice: exploratory, not a pre-registered comparison and not a reproduction of the paper's A800/full-precision tables.
- Call counts and token budgets differ between reasoning methods; a paired difference compares complete methods, not equal compute.
- Missing keys on either side shrink the paired `n`: NR missing {'SparseRAG': 0, 'DenseRAG': 0}, SR missing {'SparseRAG': 0, 'DenseRAG': 0}.
- Engine/model differences between the two runs, if any, are not controlled by this table; read them from each experiment's run manifest.

