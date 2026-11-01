# Retrieval recall probe (user 0)

Output of `scripts/retrieval_recall.py` (defaults: user 0, k = 20, e5-base-v2 without prefixes as upstream does), re-run on 2026-11-01 for this commit. BM25 uses bm25s with English stopwords. rec = mean fraction of a question's gold statements found in the top 20; all = share of questions whose gold statements were all found in one retrieval; gold = mean number of unique gold statements. Timings are from a run with a CUDA GPU.

```
user 0: 13100 statements, 2700 questions, top-20
BM25 done in 1.5s
e5 done in 37.7s

         hop |            n |         gold |  BM25 rec@20 |    e5 rec@20 |  BM25 all@20 |    e5 all@20
           2 |          300 |         1.00 |        0.847 |        1.000 |        0.847 |        1.000
           3 |          300 |         2.00 |        0.433 |        0.643 |        0.137 |        0.390
           4 |          300 |         2.00 |        0.563 |        0.570 |        0.307 |        0.303
           5 |          300 |         3.56 |        0.536 |        0.481 |        0.177 |        0.100
           6 |          300 |         3.90 |        0.496 |        0.432 |        0.140 |        0.063
           7 |          300 |         4.63 |        0.406 |        0.367 |        0.057 |        0.027
           8 |          300 |         5.04 |        0.385 |        0.341 |        0.043 |        0.037
           9 |          300 |         5.99 |        0.370 |        0.310 |        0.027 |        0.010
          10 |          300 |         6.54 |        0.351 |        0.257 |        0.003 |        0.000
```

Command: `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt .venv/bin/python scripts/retrieval_recall.py`
