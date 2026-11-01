# Farm run summary (user 0, 9 questions)

Scored with `scripts/score.py` over the raw `run_qa.py` output in this directory. Subset: user 0, 1 question per hop 2-10 (n = 9), full 13,100-statement memory, k = 20, farm model with reasoning effort low. One question is worth 0.111, so only large gaps mean anything.

Run: NaiveReason (NR) with Ignoramus, Oracle, SparseRAG, DenseRAG; SequentialReason (SR) with Ignoramus, Oracle, SparseRAG. Not run: SR with DenseRAG, MR, DR (see `docs/reproduction-status.md`).

## EM / lenient contains

| Reasoning | Ignoramus | Oracle | SparseRAG | DenseRAG |
|---|---|---|---|---|
| NR (1 call) | 0.11 / 0.11 | 0.89 / 1.00 | 0.78 / 0.78 | 0.67 / 0.67 |
| SR (5 calls) | 0.11 / 0.11 | 0.78 / 0.89 | 0.67 / 0.89 | not run |

## Raw `scripts/score.py` output

```
file                                                n     EM  contains | EM by hop group   2-4   5-7  8-10
NaiveReason/user0_1perhop-DenseRAG.jsonl            9  0.667     0.667 |                  0.67  0.67  0.67
NaiveReason/user0_1perhop-Ignoramus.jsonl           9  0.111     0.111 |                  0.00  0.00  0.33
NaiveReason/user0_1perhop-Oracle.jsonl              9  0.889     1.000 |                  1.00  0.67  1.00
NaiveReason/user0_1perhop-SparseRAG.jsonl           9  0.778     0.778 |                  0.67  0.67  1.00
SequentialReason/user0_1perhop-Ignoramus.jsonl      9  0.111     0.111 |                  0.00  0.00  0.33
SequentialReason/user0_1perhop-Oracle.jsonl         9  0.778     0.889 |                  0.67  0.67  1.00
SequentialReason/user0_1perhop-SparseRAG.jsonl      9  0.667     0.889 |                  0.67  0.33  1.00
```

Command: `.venv/bin/python scripts/score.py results/farm/*/*.jsonl`
