"""Score upstream run_qa.py output files (JSONL).

Upstream ships no scorer. Each line of a result file has k_hop, rc_id, response,
ground_truth, fold_tag. This script reports, per file, overall and per hop group:
  EM       SQuAD-normalised exact match (lowercase, strip punctuation and
           articles a/an/the, collapse whitespace) against ground_truth
  contains lenient: normalised gold appears in the normalised response as a whole
           word sequence (forgives "800 millimeters" vs "800 (millimeters)")

Usage: python scripts/score.py upstream/MPR/results/*/*.jsonl
Stdlib only.
"""
import argparse
import json
import re
import string
from collections import defaultdict

GROUPS = {"2-4": range(2, 5), "5-7": range(5, 8), "8-10": range(8, 11)}


def normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return normalize(pred) == normalize(gold)


def contains(pred, gold):
    g, p = normalize(gold), normalize(pred)
    return bool(g) and re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p) is not None


def score_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    by = defaultdict(lambda: [0, 0, 0])  # n, em, contains
    for r in rows:
        hop = int(r["k_hop"])
        keys = ["all"] + [g for g, rng in GROUPS.items() if hop in rng]
        for k in keys:
            by[k][0] += 1
            by[k][1] += em(r["response"], r["ground_truth"])
            by[k][2] += contains(r["response"], r["ground_truth"])
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    print(f"{'file':<48} {'n':>4} {'EM':>6} {'contains':>9} | EM by hop group " + " ".join(f"{g:>5}" for g in GROUPS))
    for path in args.files:
        by = score_file(path)
        if not by:
            print(f"{path}: empty")
            continue
        n, e, c = by["all"]
        name = "/".join(path.split("/")[-2:])
        groups = " ".join(f"{by[g][1] / by[g][0]:5.2f}" if by[g][0] else "    -" for g in GROUPS)
        print(f"{name:<48} {n:>4} {e / n:6.3f} {c / n:9.3f} | {'':<15} {groups}")


if __name__ == "__main__":
    main()
