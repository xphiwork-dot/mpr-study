# 00 - No-LLM probes: retrieval recall and dataset leakage

Two cheap probes that need no LLM: how many gold statements one question-only retrieval finds, and how many questions restate a
number from their own gold statements. See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

The paper has **no** retrieval-recall experiment and **no** leakage check; both probes are our addition. What they relate to:

- Section 4.2 (Figure 3 discussion, last paragraph): the paper says RAG baselines "still exhibit rapid degradation" on long hops and
  explains it by "partial retrieval". Our recall probe measures that partial retrieval directly, per hop count.
- Section 4.3 (Figure 4 discussion): the paper explains the effect of the retrieval count k by "high recall" versus "low precision".
  The probe gives recall@k per hop.
- Section 3.4.2: the retrieval setup the probe copies (BM25s for sparse, e5-base-v2 for dense, k = 20).
- Section 3.3.1 argues that datasets built on public knowledge risk "reasoning shortcuts and information leakage", and that MPR avoids
  this by using personalised statements. Section 3.3.2 Step 5 says questions are rewritten from the gold statements by LLMs. Our leakage
  check tests a possible shortcut inside MPR itself: whether a question's text already contains a number from its gold statements.

Claims tested: (1) one retrieval from the question text cannot find all facts of a long chain, and BM25 and e5 swap places around hop 4-5
(compare the paper's remark in Section 4.2 that DenseRAG is best on short hops and SparseRAG on long hops); (2) a large share of
long-hop questions leak numbers, so part of the long-hop accuracy is arithmetic on the question.

## (b) What we ran

- Data: the released MPR dataset, user 0 for the recall probe (13,100 statements, 2,700 questions, 300 per hop for hops 2-10), all 12
  users for the leakage check (32,399 questions; one question in the release is null and skipped).
- Recall probe: top-20 per question from the user's whole memory, BM25 (bm25s, English stopwords) and e5-base-v2 (cosine, no `query:` /
  `passage:` prefixes, as upstream does). Metrics: recall@20 (mean fraction of gold statements found) and all-gold@20 (share of questions
  with every gold statement found).
- Hardware: CPU for BM25 (about 2 s); e5 on a GPU takes about 38 s after the model download (`results/probes/retrieval_recall.md`).
- Key numbers, copied from `results/probes/retrieval_recall.md` (user 0): recall@20 BM25 / e5 at hop 2 is 0.847 / 1.000, at hop 10
  0.351 / 0.257; all-gold@20 at hop 10 is 0.003 / 0.000. e5 is ahead at hops 2-3, tied at hop 4 (0.563 vs 0.570, BM25 / e5), BM25 is ahead from hop 5.
- Leakage, from `results/probes/leak_check.md` (all 12 users): a number from the gold statements reappears in the question for 0.53 of
  all questions (0.13 at hop 2, 0.82 at hop 10); counting only numbers with 3 or more digits, 0.38 overall and 0.62 at hop 10.

## (c) High-level code walkthrough

Data flow of `retrieval_recall.py`: load one user's JSON (statements + questions) -> index all statements -> use each question text as
the query -> take top-k -> compare retrieved ids with the question's gold ids -> table per hop.
Data flow of `leak_check.py`: for each question extract numbers from the question and from its gold statements -> flag if they share one -> share per hop.

| File / function | What it does |
|---|---|
| `retrieval_recall.py: bm25_topk` | builds a bm25s index of all statements, retrieves top-k for all questions |
| `retrieval_recall.py: e5_topk` | encodes statements and questions with e5-base-v2 (normalised), top-k by dot product; `--e5-prefix` tests the recommended prefixes |
| `retrieval_recall.py: main` | per-question recall and all-gold flag, aggregated per hop count |
| `leak_check.py: numbers` | regex `\d+(\.\d+)?` number extraction, optional minimum digit count |
| `leak_check.py: main` | leak flag per question (any number, 3+ digit numbers), share per hop |
| `../_shared/common.py` | dataset loading (`load_user`, `hops`) |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `retrieval_recall.py: bm25_topk` | ADAPTED | upstream `baselines/SparseRAG.py:14-26` (`__make_index__`, `__retrieve_message__`): same `bm25s.tokenize(..., stopwords="en")`, `BM25().index`, `retrieve(k=topk)`. Ours batches all queries at once and returns indices. |
| `retrieval_recall.py: e5_topk` | ADAPTED | upstream `baselines/DenseRAG.py:14, 22-39`: `SentenceTransformer`, `encode(..., normalize_embeddings=True)`, score by `matmul`, top-k. Ours batches queries, runs on CPU or GPU, and adds the optional prefix flag (no upstream reference for that flag). |
| `retrieval_recall.py: main` (recall@k, all-gold@k, per-hop table) | ORIGINAL TO THIS REPO | no upstream reference: upstream ships no retrieval metric. The paper does not report recall. |
| `leak_check.py` (all of it) | ORIGINAL TO THIS REPO | no upstream reference; the leakage question is not asked in the paper (see (a)). |
| `../_shared/common.py` | ORIGINAL TO THIS REPO | no upstream reference. It reads the released dataset format (`message`, `qa`), which upstream `run_qa.py:36-37, 45` also reads. |

## (e) How to run

```bash
./setup.sh                                                       # uv venv, pinned upstream clone, dataset (needs uv, git)
.venv/bin/python experiments/00_probes_retrieval_leak/retrieval_recall.py            # BM25 ~2 s; e5 ~40 s on a GPU (+ ~3 min first model download)
.venv/bin/python experiments/00_probes_retrieval_leak/retrieval_recall.py --no-e5    # BM25 only, no model download
.venv/bin/python experiments/00_probes_retrieval_leak/leak_check.py --show-example   # stdlib only, all 12 users, a few seconds
```

Behind a corporate proxy set `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` (e5 downloads its model on first use).
No notebook: both probes take seconds and run anywhere.

## (f) Results

Files: [`results/probes/retrieval_recall.md`](../../results/probes/retrieval_recall.md), [`results/probes/leak_check.md`](../../results/probes/leak_check.md);
12-user error bars for the recall probe come from session 1 ([`../s1_t4_one_call_grid/`](../s1_t4_one_call_grid/)).

Outcome: one question-only retrieval finds fewer than half of the gold statements at long hops (BM25 0.351 and e5 0.257 recall@20 at hop 10) and almost never all of them;
BM25 overtakes e5 from hop 5 on. More than half of all questions restate a number from their own gold statements (0.82 at hop 10), so
long-hop accuracy partly measures arithmetic on the question text. This is a measurement, not a paper claim.

Verified at restructuring: `--help` of both scripts and `make_subset.py`; `leak_check.py` output for all 12 users is identical to the
committed `leak_check.md` block; the BM25 columns of `retrieval_recall.py --no-e5` are identical to the committed table. The e5 columns were
not re-run here (no model download in this task).
