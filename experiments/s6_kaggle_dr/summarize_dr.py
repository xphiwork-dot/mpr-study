"""Score and summarise the S6 DR Kaggle results, and pair them with the archived NR/SR runs.

    python3 experiments/s6_kaggle_dr/summarize_dr.py \
        --dr results/kaggle/dr/runs --slice results/t4/slices/s306.json \
        --nr results/t4/s1 --sr results/t4/s2 --out results/kaggle/dr/summary_dr.md

Conventions are the same as experiments/_shared/summarize.py: EM and lenient `contains` from
experiments/_shared/score.py, 95% percentile bootstrap over questions with 2000 resamples (seed 0 for
intervals, seed 1 for paired differences), hop groups 2-4 / 5-7 / 8-10, leak-free = slice flag
leak_any == false. Run files are deduplicated by uid:rc_id (last row wins); questions missing from either
side of a comparison are dropped from that comparison and reported, never silently replaced.

Writes the markdown summary to --out and the same numbers as JSON to <out without .md>.json.
Provenance: original to this repo (no upstream reference; upstream ships no analysis code).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "experiments", "_shared"))
from score import contains, em  # noqa: E402

GROUPS = {"2-4": range(2, 5), "5-7": range(5, 8), "8-10": range(8, 11)}
B = 2000
MEM_FILE = {"SparseRAG": "SparseRAG", "DenseRAG": "DenseRAG"}


def read_rows(path):
    """Last row wins per uid:rc_id; returns (dict, duplicate_count)."""
    by, dup = {}, 0
    if not os.path.exists(path):
        return by, dup
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        k = f"{r['uid']}:{r['rc_id']}"
        if k in by:
            dup += 1
        by[k] = r
    return by, dup


def boot_ci(vals, seed=0, b=B):
    import numpy as np
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.sort(np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(b)]))
    return float(means[int(0.025 * b)]), float(means[int(0.975 * b) - 1])


def fmt(m, ci=None):
    return f"{m:.3f}" if ci is None else f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def metrics(rows, keys, metric):
    fn = em if metric == "em" else contains
    return [float(fn(rows[k]["response"], rows[k]["ground_truth"])) for k in keys]


def subset(rows, keys, pred):
    return [k for k in keys if pred(rows[k])]


def paired(a, b, keys, metric):
    usable = [k for k in keys if k in a and k in b]
    va, vb = metrics(a, usable, metric), metrics(b, usable, metric)
    d = [x - y for x, y in zip(va, vb)]
    return len(usable), sum(d) / len(d) if d else None, boot_ci(d, seed=1) if d else None


def load_run(path):
    rows, dup = read_rows(path)
    return rows, dup


def table_for(lines, runs, keys, title, flags=None):
    lines += [f"### {title}", "",
              "| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |", "|---|---|---|---|---|---|---|"]
    for name, rows in runs.items():
        ks = [k for k in keys if k in rows]
        if not ks:
            continue
        e, c = metrics(rows, ks, "em"), metrics(rows, ks, "contains")
        g = []
        for gname, rng in GROUPS.items():
            gk = [k for k in ks if int(rows[k]["k_hop"]) in rng]
            g.append(f"{sum(metrics(rows, gk, 'em')) / len(gk):.3f} (n={len(gk)})" if gk else "-")
        lines.append(f"| {name} | {len(ks)} | {fmt(sum(e) / len(e), boot_ci(e))} | "
                     f"{fmt(sum(c) / len(c), boot_ci(c))} | " + " | ".join(g) + " |")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dr", default=os.path.join(REPO, "results", "kaggle", "dr", "runs"))
    ap.add_argument("--slice", default=os.path.join(REPO, "results", "t4", "slices", "s306.json"))
    ap.add_argument("--nr", default=os.path.join(REPO, "results", "t4", "s1"))
    ap.add_argument("--sr", default=os.path.join(REPO, "results", "t4", "s2"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "kaggle", "dr", "summary_dr.md"))
    a = ap.parse_args()

    sl = json.load(open(a.slice, encoding="utf-8"))
    flags = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    expected = list(flags)
    leak_free = [k for k in expected if not flags[k]["leak_any"]]

    dr_runs, dr_meta = {}, {}
    for mem in ("SparseRAG", "DenseRAG"):
        path = os.path.join(a.dr, f"DecompositionReason_{mem}_k20.jsonl")
        rows, dup = load_run(path)
        dr_runs[f"DR {mem}"] = rows
        fail_path = os.path.join(a.dr, f"failures_{mem}.jsonl")
        failures = read_rows(fail_path)[0] if os.path.exists(fail_path) else {}
        dr_meta[mem] = {"path": path, "rows": len(rows), "expected": len(expected),
                        "missing": len([k for k in expected if k not in rows]),
                        "extra": len([k for k in rows if k not in flags]),
                        "duplicates_ignored": dup, "failures": len(failures),
                        "missing_keys": [k for k in expected if k not in rows][:10]}

    nr_runs, sr_runs = {}, {}
    for mem, f in MEM_FILE.items():
        nr_runs[f"NR {mem}"] = load_run(os.path.join(a.nr, f"NaiveReason_{f}_k20.jsonl"))[0]
        sr_runs[f"SR {mem}"] = load_run(os.path.join(a.sr, f"SequentialReason_{f}_k20_s5.jsonl"))[0]

    lines = ["# S6 DR results (generated by experiments/s6_kaggle_dr/summarize_dr.py)", "",
             f"Method: DecompositionReason (upstream a24953c), `decompose_num` = 5, greedy, "
             f"divide/solve/merge caps 200/200/64 tokens. Slice `{os.path.basename(a.slice)}`: "
             f"{sl['meta']['n']} questions, seed {sl['meta']['seed']}, {sl['meta']['per_hop']} per hop "
             f"(2-10), all 12 users; leak-free subset {len(leak_free)} (any-number rule). "
             f"Intervals: 95% percentile bootstrap over questions ({B} resamples); paired differences use "
             f"the same questions on both sides and are reported with the shared-question n.", ""]

    rows_md = ["## Coverage", "", "| run | rows | expected | missing | extra | duplicates ignored | failures |",
               "|---|---|---|---|---|---|---|"]
    for mem, m in dr_meta.items():
        rows_md.append(f"| DR {mem} | {m['rows']} | {m['expected']} | {m['missing']} | {m['extra']} | "
                       f"{m['duplicates_ignored']} | {m['failures']} |")
    lines += rows_md + [""]
    complete = all(m["rows"] == 306 and m["missing"] == 0 and m["failures"] == 0 for m in dr_meta.values())

    table_for(lines, dr_runs, expected, "DR on all slice questions")
    table_for(lines, dr_runs, leak_free, "DR on the leak-free subset")
    lines += ["### DR EM by hop (all questions)", "",
              "| run | " + " | ".join(str(h) for h in range(2, 11)) + " |",
              "|---|" + "---|" * 9]
    for name, rows in dr_runs.items():
        cells = []
        for h in range(2, 11):
            ks = [k for k in rows if int(rows[k]["k_hop"]) == h]
            cells.append(f"{sum(metrics(rows, ks, 'em')) / len(ks):.2f}" if ks else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    # parser / call accounting
    lines += ["## DR call accounting and parser behaviour", "",
              "| run | rows | mean sub-questions | calls | 3/4/5/6/7-call rows | fallback rows | rows with dropped blank lines | prompt tokens | completion tokens | engine seconds |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    callhist = {}
    for mem, rows in dr_runs.items():
        vals = list(rows.values())
        if not vals:
            continue
        hist = {c: sum(1 for r in vals if r.get("calls") == c) for c in range(3, 8)}
        callhist[mem] = hist
        lines.append(f"| {mem} | {len(vals)} | "
                     f"{sum(len(r.get('sub_questions') or []) for r in vals) / len(vals):.2f} | "
                     f"{sum(r.get('calls', 0) for r in vals)} | "
                     + "/".join(str(hist[c]) for c in range(3, 8)) + " | "
                     f"{sum(1 for r in vals if (r.get('parser') or {}).get('fallback'))} | "
                     f"{sum(1 for r in vals if (r.get('parser') or {}).get('dropped_blank'))} | "
                     f"{sum(r.get('prompt_tokens', 0) for r in vals)} | "
                     f"{sum(r.get('completion_tokens', 0) for r in vals)} | "
                     f"{sum(r.get('secs', 0.0) for r in vals):.0f} |")
    lines.append("")

    # paired against NR and SR
    diffs = {"overall": expected, "leak_free": leak_free}
    for g, rng in GROUPS.items():
        diffs[g] = [k for k in expected if int(flags[k]["hop"]) in rng]
    lines += ["## Paired differences against the archived runs", "",
              "DR minus baseline, same questions on both sides. NR is the one-call NaiveReason of "
              "`results/t4/s1`; SR is the five-step SequentialReason of `results/t4/s2`. Both used the same "
              "model revision, slice, retrievers, k=20 and greedy decoding; the Kaggle engine version is "
              "recorded in the run manifest and is not guaranteed to equal the Colab engine version.", "",
              "| comparison | subset | n | dEM [95% CI] | dContains [95% CI] |", "|---|---|---|---|---|"]
    diff_json = {}
    for mem in ("SparseRAG", "DenseRAG"):
        for base_name, base in (("NR", nr_runs[f"NR {mem}"]), ("SR", sr_runs[f"SR {mem}"])):
            for sub, keys in diffs.items():
                n, dm, ci_e = paired(dr_runs[f"DR {mem}"], base, keys, "em")
                _, dc, ci_c = paired(dr_runs[f"DR {mem}"], base, keys, "contains")
                if n == 0:
                    continue
                lines.append(f"| DR {mem} - {base_name} {mem} | {sub} | {n} | {fmt(dm, ci_e)} | {fmt(dc, ci_c)} |")
                diff_json.setdefault(f"DR {mem}", {})[f"vs {base_name} {sub}"] = {
                    "n": n, "dEM": dm, "dEM_ci": ci_e, "dContains": dc, "dContains_ci": ci_c}
    lines.append("")

    # wall time / call totals from the DR rows and the archived timing files
    lines += ["## Calls and wall time", ""]
    for mem, rows in dr_runs.items():
        vals = list(rows.values())
        lines.append(f"- {mem}: {sum(r.get('calls', 0) for r in vals)} calls, "
                     f"{sum(r.get('prompt_tokens', 0) for r in vals)} prompt + "
                     f"{sum(r.get('completion_tokens', 0) for r in vals)} completion tokens, "
                     f"{sum(r.get('secs', 0.0) for r in vals):.0f}s summed request time, "
                     f"{len(vals)} rows.")
    timing = os.path.join(os.path.dirname(a.dr), "run_summary.json")
    if os.path.exists(timing):
        s = json.load(open(timing, encoding="utf-8"))
        for name, m in s.get("memories", {}).items():
            lines.append(f"- {name}: wall {m.get('wall_s')}s, rows {m.get('rows')}, failures {m.get('failures')}, "
                         f"stopped={m.get('stopped')}.")
    lines.append("")

    lines += ["## Caveats", "",
              "- One slice (306 questions), one seed, one 4-bit 7B model on a T4: exploratory, not a "
              "reproduction of the paper's A800/full-precision tables.",
              "- DR makes 3-7 calls per question versus 1 for NR and 5 for SR, so the comparison is of "
              "complete methods, not of equal compute.",
              "- The Kaggle vLLM/torch versions are recorded in `manifest_out.json`; the historical Colab "
              "versions were not pinned, so small decoding differences cannot be excluded. Greedy decoding "
              "is used on both sides.",
              "- Intervals are percentile bootstraps over questions; no multiple-comparison correction is "
              "applied to the exploratory subset table.",
              ""]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    payload = {"complete": complete, "meta": dr_meta, "paired": diff_json,
               "call_histogram": callhist, "slice": os.path.basename(a.slice),
               "leak_free_n": len(leak_free)}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {a.out} and {os.path.splitext(a.out)[0]}.json; complete={complete}")
    for mem, m in dr_meta.items():
        print(f"  {mem}: {m['rows']}/{m['expected']} rows, missing={m['missing']}, failures={m['failures']}")


if __name__ == "__main__":
    main()
