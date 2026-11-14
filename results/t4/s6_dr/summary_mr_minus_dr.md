# MR vs DR on the s306.json slice

MR: `results/t4/s5_mr` with `StructuredReason_{mem}_k20_s5.jsonl`; DR: `results/t4/s6_dr/runs` with `DecompositionReason_{mem}_k20.jsonl`. Both sides use `experiments/_shared/score.py` (EM, lenient contains), 2000 bootstrap resamples (seed 0 intervals, seed 1 paired differences), hop groups 2-4/5-7/8-10 and the leak-free subset `leak_any == false` (140 of 306 questions). Paired rows drop questions missing on either side and report their `n`.

## Coverage

| method | memory | rows | expected | missing | extra | duplicates ignored |
|---|---|---|---|---|---|---|
| MR | SparseRAG | 306 | 306 | 0 | 0 | 0 |
| MR | DenseRAG | 306 | 306 | 0 | 0 | 0 |
| DR | SparseRAG | 306 | 306 | 0 | 0 | 0 |
| DR | DenseRAG | 306 | 306 | 0 | 0 | 0 |

### MR on all slice questions

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| MR SparseRAG | 306 | 0.203 [0.160, 0.248] | 0.209 [0.167, 0.255] | 0.275 (n=102) | 0.147 (n=102) | 0.186 (n=102) |
| MR DenseRAG | 306 | 0.216 [0.170, 0.261] | 0.225 [0.176, 0.271] | 0.314 (n=102) | 0.137 (n=102) | 0.196 (n=102) |

### MR on the leak-free subset

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| MR SparseRAG | 140 | 0.186 [0.121, 0.257] | 0.193 [0.129, 0.264] | 0.276 (n=87) | 0.031 (n=32) | 0.048 (n=21) |
| MR DenseRAG | 140 | 0.236 [0.164, 0.307] | 0.250 [0.179, 0.321] | 0.322 (n=87) | 0.094 (n=32) | 0.095 (n=21) |

### DR on all slice questions

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| DR SparseRAG | 306 | 0.131 [0.095, 0.167] | 0.131 [0.095, 0.167] | 0.216 (n=102) | 0.069 (n=102) | 0.108 (n=102) |
| DR DenseRAG | 306 | 0.118 [0.082, 0.157] | 0.121 [0.085, 0.160] | 0.176 (n=102) | 0.078 (n=102) | 0.098 (n=102) |

### DR on the leak-free subset

| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |
|---|---|---|---|---|---|---|
| DR SparseRAG | 140 | 0.143 [0.086, 0.207] | 0.143 [0.086, 0.207] | 0.195 (n=87) | 0.000 (n=32) | 0.143 (n=21) |
| DR DenseRAG | 140 | 0.129 [0.079, 0.186] | 0.136 [0.086, 0.193] | 0.172 (n=87) | 0.031 (n=32) | 0.095 (n=21) |

## Paired differences (MR minus DR)

| memory | subset | n | dEM [95% CI] | dContains [95% CI] |
|---|---|---|---|---|
| SparseRAG | overall | 306 | 0.072 [0.029, 0.118] | 0.078 [0.033, 0.124] |
| SparseRAG | leak_free | 140 | 0.043 [-0.007, 0.100] | 0.050 [-0.007, 0.107] |
| SparseRAG | 2-4 | 102 | 0.059 [-0.020, 0.137] | 0.069 [-0.020, 0.147] |
| SparseRAG | 5-7 | 102 | 0.078 [0.000, 0.157] | 0.088 [0.010, 0.167] |
| SparseRAG | 8-10 | 102 | 0.078 [0.000, 0.157] | 0.078 [0.000, 0.157] |
| DenseRAG | overall | 306 | 0.098 [0.052, 0.147] | 0.105 [0.056, 0.154] |
| DenseRAG | leak_free | 140 | 0.107 [0.029, 0.186] | 0.114 [0.036, 0.193] |
| DenseRAG | 2-4 | 102 | 0.137 [0.039, 0.235] | 0.147 [0.049, 0.255] |
| DenseRAG | 5-7 | 102 | 0.059 [-0.020, 0.137] | 0.069 [-0.010, 0.147] |
| DenseRAG | 8-10 | 102 | 0.098 [0.029, 0.176] | 0.098 [0.029, 0.176] |

## Caveats

- Two methods measured once each on one slice: exploratory, not a pre-registered comparison and not a reproduction of the paper's A800/full-precision tables.
- Call counts and token budgets differ between reasoning methods; a paired difference compares complete methods, not equal compute.
- Missing keys on either side shrink the paired `n`: MR missing {'SparseRAG': 0, 'DenseRAG': 0}, DR missing {'SparseRAG': 0, 'DenseRAG': 0}.
- Engine/model differences between the two runs, if any, are not controlled by this table; read them from each experiment's run manifest.

