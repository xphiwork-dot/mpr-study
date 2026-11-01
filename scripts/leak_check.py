"""Leakage probe: does the question text restate a number from its own gold statements?

For each question, extract numbers from the question and from the gold statements
(the statements the question was built from). A question "leaks" if the two sets
share at least one number. A leaking question can be partly answered with
arithmetic on the question text alone, without any memory. Reports the share of
leaking questions per hop count, for any number and for numbers with 3+ digits
(which are unlikely to be chance matches like "2" or "10").

Runs over all 12 users (32,400 questions) by default. Stdlib only.

Usage: python scripts/leak_check.py [--users 0 1 2 ...]
"""
import argparse
import re
from collections import defaultdict

from common import DEFAULT_DATA_DIR, hops, load_user

NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text, min_digits=1):
    return {n for n in NUM.findall(text) if len(n.replace(".", "")) >= min_digits}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--users", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--show-example", action="store_true", help="print one leaking hop-7 question")
    args = ap.parse_args()

    tot = defaultdict(int)
    any_n = defaultdict(int)
    big_n = defaultdict(int)
    example = None
    for u in args.users:
        data = load_user(args.data_dir, u)
        for h in hops(data):
            for rc, qa in data["qa"][h].items():
                q = qa.get("question")
                if not q:  # upstream has one null question
                    continue
                gold_text = " ".join(data["message"][m] for m in qa["message"])
                tot[h] += 1
                a = numbers(q) & numbers(gold_text)
                b = numbers(q, 3) & numbers(gold_text, 3)
                any_n[h] += bool(a)
                big_n[h] += bool(b)
                if example is None and h == "7" and a:
                    example = (u, rc, q, sorted(a))

    hs = sorted(tot, key=int)
    n_all = sum(tot.values())
    print(f"users {args.users}: {n_all} questions")
    print("hop            " + " ".join(f"{h:>6}" for h in hs) + "    all")
    print("any number     " + " ".join(f"{any_n[h] / tot[h]:6.2f}" for h in hs) + f" {sum(any_n.values()) / n_all:6.2f}")
    print("3+ digit nums  " + " ".join(f"{big_n[h] / tot[h]:6.2f}" for h in hs) + f" {sum(big_n.values()) / n_all:6.2f}")
    if args.show_example and example:
        print(f"\nexample (user {example[0]}, {example[1]}), shared numbers {example[3]}:\n{example[2]}")


if __name__ == "__main__":
    main()
