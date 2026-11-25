"""Paired comparison of two reasoning methods on the same slice (used for DR vs MR).

    python3 experiments/s6_t4_dr/compare_mr_dr.py \
        --a-dir results/t4/s6_dr/runs --a-pattern 'DecompositionReason_{mem}_k20.jsonl' \
        --b-dir /path/to/mr/runs     --b-pattern 'MRReasoning_{mem}_k20.jsonl' \
        --slice results/t4/slices/s306.json --out results/t4/s6_dr/summary_mr_vs_dr.md

Both sides are read with the same rules as summarize_dr.py: jsonl rows keyed `uid:rc_id`, last row wins,
duplicates counted, EM/contains from `experiments/_shared/score.py`, 95% percentile bootstrap over
questions (2000 resamples, seed 0 for intervals, seed 1 for paired differences), hop groups 2-4/5-7/8-10 and
the leak-free subset (`leak_any == false` in the slice). Questions present on only one side are dropped from
that comparison and reported as missing/extra, never silently replaced; every paired row states its `n`.

The method labels are display-only: nothing here assumes a call flow, so the same tool compares NR/SR/DR/MR
as long as both sides follow the shared row schema (`uid`, `rc_id`, `k_hop`, `response`, `ground_truth`).

Provenance: original to this repo (no upstream reference; upstream ships no analysis code).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import summarize_dr as S  # noqa: E402


def load_side(directory, pattern, memories, expected):
    """Return {label: rows}, {label: meta} for each memory, with coverage against the slice keys."""
    runs, meta = {}, {}
    for mem in memories:
        name = pattern.format(mem=mem)
        path = os.path.join(directory, name)
        rows, dup = S.read_rows(path)
        label = f"{mem}"
        runs[label] = rows
        meta[label] = {
            "path": path,
            "exists": os.path.exists(path),
            "rows": len(rows),
            "missing": len([k for k in expected if k not in rows]),
            "extra": len([k for k in rows if k not in expected]),
            "duplicates_ignored": dup,
            "missing_keys": [k for k in expected if k not in rows][:10],
        }
    return runs, meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--a-pattern", default="DecompositionReason_{mem}_k20.jsonl")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--b-pattern", default="DecompositionReason_{mem}_k20.jsonl")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--memories", default="SparseRAG,DenseRAG")
    ap.add_argument(
        "--slice", default=os.path.join(REPO, "results", "t4", "slices", "s306.json")
    )
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "results", "t4", "s6_dr", "summary_mr_vs_dr.md"),
    )
    a = ap.parse_args()

    memories = [m.strip() for m in a.memories.split(",") if m.strip()]
    sl = json.load(open(a.slice, encoding="utf-8"))
    flags = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    expected = list(flags)
    leak_free = [k for k in expected if not flags[k]["leak_any"]]

    side_a, meta_a = load_side(a.a_dir, a.a_pattern, memories, expected)
    side_b, meta_b = load_side(a.b_dir, a.b_pattern, memories, expected)

    lines = [
        f"# {a.a_label} vs {a.b_label} on the {os.path.basename(a.slice)} slice",
        "",
        f"{a.a_label}: `{a.a_dir}` with `{a.a_pattern}`; {a.b_label}: `{a.b_dir}` with "
        f"`{a.b_pattern}`. Both sides use `experiments/_shared/score.py` (EM, lenient contains), "
        f"{S.B} bootstrap resamples (seed 0 intervals, seed 1 paired differences), hop groups "
        f"2-4/5-7/8-10 and the leak-free subset `leak_any == false` ({len(leak_free)} of "
        f"{len(expected)} questions). Paired rows drop questions missing on either side and report "
        f"their `n`.",
        "",
    ]

    lines += [
        "## Coverage",
        "",
        f"| method | memory | rows | expected | missing | extra | duplicates ignored |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, meta in ((a.a_label, meta_a), (a.b_label, meta_b)):
        for mem, m in meta.items():
            lines.append(
                f"| {label} | {mem} | {m['rows']} | {len(expected)} | {m['missing']} | "
                f"{m['extra']} | {m['duplicates_ignored']} |"
            )
    lines.append("")

    for label, runs in ((a.a_label, side_a), (a.b_label, side_b)):
        S.table_for(
            lines,
            {f"{label} {mem}": rows for mem, rows in runs.items()},
            expected,
            f"{label} on all slice questions",
        )
        S.table_for(
            lines,
            {f"{label} {mem}": rows for mem, rows in runs.items()},
            leak_free,
            f"{label} on the leak-free subset",
        )

    subsets = {"overall": expected, "leak_free": leak_free}
    for g, rng in S.GROUPS.items():
        subsets[g] = [k for k in expected if int(flags[k]["hop"]) in rng]

    lines += [
        f"## Paired differences ({a.a_label} minus {a.b_label})",
        "",
        "| memory | subset | n | dEM [95% CI] | dContains [95% CI] |",
        "|---|---|---|---|---|",
    ]
    payload = {
        "slice": os.path.basename(a.slice),
        "paired": {},
        "coverage": {a.a_label: meta_a, a.b_label: meta_b},
        "leak_free_n": len(leak_free),
    }
    for mem in memories:
        ra, rb = side_a.get(mem, {}), side_b.get(mem, {})
        payload["paired"][mem] = {}
        for sub, keys in subsets.items():
            n, dm, ci_e = S.paired(ra, rb, keys, "em")
            _, dc, ci_c = S.paired(ra, rb, keys, "contains")
            if n == 0:
                continue
            lines.append(
                f"| {mem} | {sub} | {n} | {S.fmt(dm, ci_e)} | {S.fmt(dc, ci_c)} |"
            )
            payload["paired"][mem][sub] = {
                "n": n,
                "dEM": dm,
                "dEM_ci": ci_e,
                "dContains": dc,
                "dContains_ci": ci_c,
            }
    lines.append("")

    lines += [
        "## Caveats",
        "",
        "- Two methods measured once each on one slice: exploratory, not a pre-registered comparison "
        "and not a reproduction of the paper's A800/full-precision tables.",
        "- Call counts and token budgets differ between reasoning methods; a paired difference "
        "compares complete methods, not equal compute.",
        f"- Missing keys on either side shrink the paired `n`: {a.a_label} missing "
        f"{ {m: v['missing'] for m, v in meta_a.items()} }, {a.b_label} missing "
        f"{ {m: v['missing'] for m, v in meta_b.items()} }.",
        "- Engine/model differences between the two runs, if any, are not controlled by this table; "
        "read them from each experiment's run manifest.",
        "",
    ]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {a.out} and {os.path.splitext(a.out)[0]}.json")
    for mem in memories:
        m = payload["paired"].get(mem, {}).get("overall")
        print(
            f"  {mem}: "
            + (
                "no shared questions"
                if not m
                else f"n={m['n']} dEM={m['dEM']:+.3f} dContains={m['dContains']:+.3f}"
            )
        )


if __name__ == "__main__":
    main()
