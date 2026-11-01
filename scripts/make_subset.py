"""Write a small stratified question subset in the upstream data format.

Keeps the user's full memory (all statements) and samples N questions per hop,
so retrieval stays as hard as in the full benchmark while an LLM run stays cheap.
Output: data/subsets/user<U>_<N>perhop.json (the title upstream run_qa.py reads
through MPR_DATA_TITLE / MPR_DATA_DIR; defaults point at user0_1perhop).

Usage: python scripts/make_subset.py [--user 0] [--per-hop 1] [--seed 0]
"""
import argparse
import json
import os
import random

from common import DEFAULT_DATA_DIR, hops, load_user


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--user", type=int, default=0)
    ap.add_argument("--per-hop", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join("data", "subsets"))
    args = ap.parse_args()

    data = load_user(args.data_dir, args.user)
    rng = random.Random(args.seed)
    qa = {}
    for h in hops(data):
        ok = sorted(k for k, v in data["qa"][h].items() if v.get("question") and v.get("answer"))
        qa[h] = {k: data["qa"][h][k] for k in sorted(rng.sample(ok, args.per_hop))}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"user{args.user}_{args.per_hop}perhop.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"message": data["message"], "qa": qa}, f, ensure_ascii=False)
    print(f"wrote {out}: {len(data['message'])} statements, {sum(len(v) for v in qa.values())} questions")


if __name__ == "__main__":
    main()
