"""No-LLM probe: how many gold statements does one question-only retrieval find?

For every question of one user, retrieve the top-k statements out of the user's
whole memory (~13k statements) using the question text as the query, then measure
which of the question's gold statements were found. Reports, per hop count:
  recall@k   = mean fraction of gold statements retrieved
  all-gold@k = share of questions whose gold statements were ALL retrieved
for BM25 (bm25s, English stopwords) and dense e5-base-v2 (cosine, no prefixes,
as the upstream code does).

Usage: python scripts/retrieval_recall.py [--user 0] [--k 20] [--no-e5]
"""
import argparse
import time

import numpy as np

from common import DEFAULT_DATA_DIR, hops, load_user


def bm25_topk(texts, queries, k):
    import bm25s

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    res, _ = retriever.retrieve(
        bm25s.tokenize(queries, stopwords="en", show_progress=False), k=k, show_progress=False
    )
    return res


def e5_topk(texts, queries, k, model_name, prefix):
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(model_name, device=dev)
    p, q = ("passage: ", "query: ") if prefix else ("", "")
    doc = enc.encode([p + t for t in texts], normalize_embeddings=True, batch_size=256,
                     convert_to_tensor=True, show_progress_bar=False)
    qry = enc.encode([q + t for t in queries], normalize_embeddings=True, batch_size=256,
                     convert_to_tensor=True, show_progress_bar=False)
    top = torch.topk(qry @ doc.T, k=k, dim=1).indices
    return top.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--user", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--no-e5", action="store_true", help="BM25 only (no model download)")
    ap.add_argument("--e5-model", default="intfloat/e5-base-v2")
    ap.add_argument("--e5-prefix", action="store_true",
                    help="use the recommended 'query: '/'passage: ' prefixes (upstream does not)")
    args = ap.parse_args()

    data = load_user(args.data_dir, args.user)
    ids = list(data["message"])
    texts = [data["message"][i] for i in ids]
    rows = [(h, rc, qa) for h in hops(data) for rc, qa in data["qa"][h].items() if qa.get("question")]
    queries = [qa["question"] for _, _, qa in rows]
    print(f"user {args.user}: {len(ids)} statements, {len(rows)} questions, top-{args.k}")

    systems = {}
    t = time.time()
    systems["BM25"] = bm25_topk(texts, queries, args.k)
    print(f"BM25 done in {time.time() - t:.1f}s")
    if not args.no_e5:
        t = time.time()
        systems["e5"] = e5_topk(texts, queries, args.k, args.e5_model, args.e5_prefix)
        print(f"e5 done in {time.time() - t:.1f}s")

    # per-question recall and all-gold flag
    stat = {name: ([], []) for name in systems}
    for qi, (_, _, qa) in enumerate(rows):
        gold = set(qa["message"])
        for name, top in systems.items():
            got = {ids[j] for j in top[qi]}
            r = len(gold & got) / len(gold)
            stat[name][0].append(r)
            stat[name][1].append(r == 1.0)

    hop_of = np.array([int(h) for h, _, _ in rows])
    ngold = np.array([len(set(qa["message"])) for _, _, qa in rows])  # message lists can repeat ids
    names = list(systems)
    head = ["hop", "n", "gold"] + [f"{n} rec@{args.k}" for n in names] + [f"{n} all@{args.k}" for n in names]
    print("\n" + " | ".join(f"{c:>12}" for c in head))
    for h in sorted(set(hop_of)):
        m = hop_of == h
        cells = [f"{h}", f"{m.sum()}", f"{ngold[m].mean():.2f}"]
        cells += [f"{np.mean(np.array(stat[n][0])[m]):.3f}" for n in names]
        cells += [f"{np.mean(np.array(stat[n][1])[m]):.3f}" for n in names]
        print(" | ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()
