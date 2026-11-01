"""Stratified question slice over all 12 users, fixed seed, nested prefixes.

For every hop (2-10) all valid questions of all users are shuffled once with Random(seed)
and the first N are kept. Slices are nested: the --per-hop 34 slice (306 questions, used for
chain-of-thought runs) is a subset of the --per-hop 100 slice (900 questions), so paired
comparisons across sessions use the same questions. Each entry also carries the leakage flags
from scripts/leak_check.py (the question restates a number from its own gold statements).

Usage: python scripts/t4/make_slice.py --per-hop 100 --out results/t4/slices/s900.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from leak_check import numbers  # noqa: E402  (the rule is defined there)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join("data", "unz", "mpr_dataset", "question_answer"))
    ap.add_argument("--per-hop", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lib.DATA_DIR = args.data_dir

    by_hop = {h: [] for h in range(2, 11)}
    for u in lib.USERS:
        d = lib.load_user(u)
        for h, rc, qa in lib.all_questions(d):
            gold = " ".join(d["message"][m] for m in qa["message"])
            by_hop[h].append({
                "uid": u, "hop": h, "rc_id": rc, "n_gold": len(set(qa["message"])),
                "leak_any": bool(numbers(qa["question"]) & numbers(gold)),
                "leak_big": bool(numbers(qa["question"], 3) & numbers(gold, 3)),
            })
    items = []
    for h in sorted(by_hop):
        cand = sorted(by_hop[h], key=lambda x: (x["uid"], x["rc_id"]))
        random.Random(f"{args.seed}-{h}").shuffle(cand)
        items.extend(cand[: args.per_hop])
    meta = {"seed": args.seed, "per_hop": args.per_hop, "n": len(items),
            "rule": "per hop: sort by (uid, rc_id), shuffle with random.Random(f'{seed}-{hop}'), keep first per_hop",
            "per_user": dict(sorted(Counter(i["uid"] for i in items).items())),
            "leak_any": sum(i["leak_any"] for i in items), "leak_big": sum(i["leak_big"] for i in items)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "items": items}, f, indent=0)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
