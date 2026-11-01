# Farm token usage (user 0, 9 questions)

Summed from the `[Usage]` lines of the run logs, with the same awk used by `scripts/farm_run.sh`. The raw .log files are not committed (gitignored).

| Run | calls | input tokens | output tokens |
|---|---|---|---|
| Naive-DenseRAG | 9 | 4663 | 1217 |
| Naive-Ignoramus | 9 | 1889 | 6260 |
| Naive-Oracle | 9 | 2400 | 1260 |
| Naive-SparseRAG | 9 | 4580 | 1236 |
| Sequential-Ignoramus | 45 | 12031 | 7031 |
| Sequential-Oracle | 45 | 14511 | 5512 |
| Sequential-SparseRAG | 45 | 25397 | 5555 |
| total | 171 | 65471 | 28071 |

