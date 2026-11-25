"""Retrieval-only pass over all 12 users (no LLM). Runs on the VM, before vLLM takes the GPU.

For every question of every user: top-50 by BM25 (bm25s, English stopwords) and by dense
e5-base-v2 (cosine, no query:/passage: prefixes, as upstream; encoder in fp16). Writes
  cache/emb_user<u>.npy        statement embeddings (used later by grid.py for chain-of-thought queries)
  results/retrieval/recall.csv.gz   one row per question: per-system recall@k and all-gold@k for k in 10,20,30,50
Resumable at user level: a user whose cache file and recall rows already exist is skipped.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared")
)  # lib.py, grid.py, score.py (flat next to this file on the Colab VM)
import lib  # noqa: E402

KS = (10, 20, 30, 50)
HEADER = ["uid", "hop", "rc_id", "n_gold"] + [
    f"{s}_{m}{k}" for s in ("bm25", "e5") for k in KS for m in ("rec", "all")
]


def main():
    out = os.path.join(lib.OUT, "retrieval")
    os.makedirs(out, exist_ok=True)
    os.makedirs(lib.CACHE, exist_ok=True)
    enc = lib.get_encoder()
    import bm25s

    t_all = time.time()
    for u in lib.USERS:
        part = os.path.join(out, f"recall_user{u}.csv")
        emb_path = os.path.join(lib.CACHE, f"emb_user{u}.npy")
        if os.path.exists(part + ".done") and os.path.exists(emb_path):
            print(f"user {u}: done, skip", flush=True)
            continue
        t = time.time()
        d = lib.load_user(u)
        ids = list(d["message"])
        texts = [d["message"][i] for i in ids]
        rows = lib.all_questions(d)
        queries = [qa["question"] for _, _, qa in rows]
        ret = bm25s.BM25()
        ret.index(
            bm25s.tokenize(texts, stopwords="en", show_progress=False),
            show_progress=False,
        )
        bm, _ = ret.retrieve(
            bm25s.tokenize(queries, stopwords="en", show_progress=False),
            k=50,
            show_progress=False,
        )
        emb = enc.encode(
            texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False
        )
        qe = enc.encode(
            queries, normalize_embeddings=True, batch_size=128, show_progress_bar=False
        )
        np.save(emb_path, emb.astype("float16"))
        import torch

        dev = "cuda"
        et = torch.from_numpy(emb).to(dev).float()
        top = []
        for s in range(0, len(qe), 512):
            sc = torch.from_numpy(qe[s : s + 512]).to(dev).float() @ et.T
            top.append(torch.topk(sc, 50, dim=1).indices.cpu().numpy())
        top = np.concatenate(top)
        with open(part, "w") as f:
            for qi, (h, rc, qa) in enumerate(rows):
                gold = set(qa["message"])
                cells = [u, h, rc, len(gold)]
                for sysname, tp in (("bm25", bm), ("e5", top)):
                    for k in KS:
                        got = {ids[j] for j in tp[qi][:k]}
                        r = len(gold & got) / len(gold)
                        cells += [f"{r:.4f}", int(r == 1.0)]
                f.write(",".join(str(c) for c in cells) + "\n")
        open(part + ".done", "w").close()
        print(
            f"user {u}: {len(rows)} questions, {len(ids)} statements, {time.time() - t:.1f}s",
            flush=True,
        )
    # merge into one gz csv
    rows = []
    for u in lib.USERS:
        with open(os.path.join(out, f"recall_user{u}.csv")) as f:
            rows += [line.rstrip("\n").split(",") for line in f]
    lib.write_gz_csv(os.path.join(out, "recall.csv.gz"), HEADER, rows)
    print(
        f"retrieval total {time.time() - t_all:.1f}s, {len(rows)} questions", flush=True
    )


if __name__ == "__main__":
    main()
