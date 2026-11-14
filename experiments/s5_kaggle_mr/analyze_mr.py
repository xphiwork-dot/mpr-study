#!/usr/bin/env python3
"""Score and compare the MR Kaggle run.  Needs numpy (analysis only).

Produces results/kaggle/mr/summary_mr.md from the raw jsonl rows, with the same
EM / contains rules as experiments/_shared/score.py and 2,000-resample
percentile bootstrap intervals over questions.  Paired differences use a fixed
seed and only the questions present in both runs.

Usage:
  python experiments/s5_kaggle_mr/analyze_mr.py \
    --results-dir results/kaggle/mr --slice results/t4/slices/s306.json \
    --data-dir ../mpr-study/data/mpr_dataset/question_answer
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHARED = os.path.join(ROOT, "experiments", "_shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, HERE)
from score import contains, em  # noqa: E402
import lib  # noqa: E402
import mr  # noqa: E402

GROUPS = {"2-4": range(2, 5), "5-7": range(5, 8), "8-10": range(8, 11)}
B = 2000
RUN_NAMES = {"SparseRAG": "StructuredReason_SparseRAG_k20_s5",
             "DenseRAG": "StructuredReason_DenseRAG_k20_s5"}


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


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_runs(results_dir):
    runs = {}
    for mem, name in RUN_NAMES.items():
        path = os.path.join(results_dir, name + ".jsonl")
        if os.path.exists(path):
            raw = load_jsonl(path)
            runs[mem] = {mr.row_key(r): r for r in raw if r.get("response")}
            runs[mem + "_raw_n"] = len(raw)
    return runs


def load_baseline(tag, prefix):
    path = os.path.join(ROOT, "results", "t4", tag, prefix + ".jsonl")
    if not os.path.exists(path):
        return None
    return {mr.row_key(r): r for r in load_jsonl(path) if r.get("response")}


def scores(rows, keys):
    e = [float(em(rows[k]["response"], rows[k]["ground_truth"])) for k in keys]
    c = [float(contains(rows[k]["response"], rows[k]["ground_truth"])) for k in keys]
    return e, c


def subset_keys(items, name):
    if name == "all":
        return [f"{i['uid']}:{i['rc_id']}" for i in items]
    if name == "leak-free":
        return [f"{i['uid']}:{i['rc_id']}" for i in items if not i["leak_any"]]
    if name == "leak-free-3digit":
        return [f"{i['uid']}:{i['rc_id']}" for i in items if not i["leak_big"]]
    if name in GROUPS:
        return [f"{i['uid']}:{i['rc_id']}" for i in items if int(i["hop"]) in GROUPS[name]]
    raise ValueError(name)


def score_line(rows, keys, label):
    e, c = scores(rows, keys)
    n = len(keys)
    return (f"| {label} | {n} | {fmt(sum(e) / n, boot_ci(e))} | "
            f"{fmt(sum(c) / n, boot_ci(c))} |")


def paired(rows_a, rows_b, keys, metric):
    kk = [k for k in keys if k in rows_a and k in rows_b]
    if not kk:
        return None
    ea = scores(rows_a, kk)[metric]
    eb = scores(rows_b, kk)[metric]
    d = [x - y for x, y in zip(ea, eb)]
    return len(kk), sum(d) / len(d), boot_ci(d, seed=1)


def validate(results_dir, items, data_dir):
    rep = {"expected": len(items), "memories": {}, "raw_lines": {}, "failures": 0}
    fail_path = os.path.join(results_dir, "failures.jsonl")
    if os.path.exists(fail_path):
        rep["failures"] = len(load_jsonl(fail_path))
    cache = {}
    expected = {f"{i['uid']}:{i['rc_id']}" for i in items}
    hop = {f"{i['uid']}:{i['rc_id']}": int(i["hop"]) for i in items}
    raw_counts = {}
    for mem, name in RUN_NAMES.items():
        path = os.path.join(results_dir, name + ".jsonl")
        raw = load_jsonl(path) if os.path.exists(path) else []
        raw_counts[mem] = len(raw)
        rows = {mr.row_key(r): r for r in raw if r.get("response")}
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        bad_gt, bad_hop, bad_calls = [], [], []
        for k, r in rows.items():
            uid = int(r["uid"])
            if uid not in cache:
                cache[uid] = lib.load_user(uid)
            qa = cache[uid]["qa"].get(str(r.get("k_hop")), {}).get(r["rc_id"])
            if qa is None or str(r.get("ground_truth")) != str(qa.get("answer")):
                bad_gt.append(k)
            if int(r.get("k_hop", -1)) != hop.get(k):
                bad_hop.append(k)
            if int(r.get("calls") or 0) != mr.MRConfig().expected_calls:
                bad_calls.append(k)
        rep["memories"][mem] = {
            "unique_rows": len(rows), "expected": len(expected),
            "missing": missing, "missing_n": len(missing), "extra": extra[:20],
            "extra_n": len(extra), "bad_ground_truth": bad_gt[:20],
            "bad_ground_truth_n": len(bad_gt), "bad_hop_n": len(bad_hop),
            "bad_calls_n": len(bad_calls),
            "complete": not missing and not extra and not bad_gt and not bad_hop and not bad_calls,
        }
    rep["raw_lines"] = raw_counts
    rep["complete"] = all(v["complete"] for v in rep["memories"].values())
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "kaggle", "mr"))
    ap.add_argument("--slice", default=os.path.join(ROOT, "results", "t4", "slices", "s306.json"))
    ap.add_argument("--data-dir", default=os.environ.get("MPR_QA_DIR"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--hardware", default="one Colab free-tier Tesla T4 session via colab_mr.sh")
    a = ap.parse_args()
    if not a.data_dir:
        for cand in (os.path.join(ROOT, "data", "mpr_dataset", "question_answer"),
                     os.path.join(os.path.dirname(ROOT), "mpr-study", "data", "mpr_dataset",
                                  "question_answer")):
            if os.path.isdir(cand):
                a.data_dir = cand
                break
    if not a.data_dir:
        raise SystemExit("--data-dir not found")
    lib.DATA_DIR = a.data_dir
    items = lib.load_slice(a.slice)["items"]
    runs = load_runs(a.results_dir)
    validation = validate(a.results_dir, items, a.data_dir)
    if not validation["complete"] and not a.allow_partial:
        print(json.dumps(validation, indent=2))
        raise SystemExit("MR rows are incomplete; pass --allow-partial to write a partial summary")

    nr = {m: load_baseline("s1", f"NaiveReason_{m}_k20") for m in RUN_NAMES}
    sr = {m: load_baseline("s2", f"SequentialReason_{m}_k20_s5") for m in RUN_NAMES}
    manifest = {}
    mpath = os.path.join(a.results_dir, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
    served_vllm = (manifest.get("model") or {}).get("vllm") or "unknown"
    timing = []
    tpath = os.path.join(a.results_dir, "timing.jsonl")
    if os.path.exists(tpath):
        timing = load_jsonl(tpath)

    lines = [
        "# MR (StructuredReason) on Kaggle: results",
        "",
        "Generated by `experiments/s5_kaggle_mr/analyze_mr.py` from the raw jsonl rows. "
        f"Qwen2.5-7B-Instruct-AWQ at the pinned revision, vLLM {served_vllm} on {a.hardware}; "
        "BM25 (SparseRAG) and e5-base-v2 (DenseRAG), k = 20, max_step = 5, branch_size = 2. "
        "MR uses seeded sampling only for the two branch-generation calls per middle round "
        "(T=0.7, top_p=1.0); start, selections and final answer are greedy. "
        "The historical NR/SR rows were greedy, so MR-vs-NR/SR comparisons are NOT decoding-matched. "
        f"Intervals are 95% percentile bootstrap over questions ({B} resamples, fixed seeds); "
        "paired differences are paired bootstrap on the same questions. "
        "This is one slice and one seed, an exploratory comparison with no multiple-comparison correction.",
        "",
        "## Validation",
        "",
        f"- Slice: `{os.path.basename(a.slice)}`, {len(items)} unique questions; "
        f"leak-free any-number {sum(1 for i in items if not i['leak_any'])}, "
        f"3+ digit {sum(1 for i in items if not i['leak_big'])}.",
        f"- Raw rows: " + ", ".join(f"{m} {validation['raw_lines'].get(m, 0)}" for m in RUN_NAMES) +
        f"; failure records {validation['failures']}.",
    ]
    for mem, v in validation["memories"].items():
        lines.append(
            f"- {mem}: {v['unique_rows']}/{v['expected']} unique successful rows, missing {v['missing_n']}, "
            f"extra {v['extra_n']}, bad ground truth {v['bad_ground_truth_n']}, "
            f"bad hop {v['bad_hop_n']}, bad call count {v['bad_calls_n']}."
        )
    lines += ["", "## Scores on the frozen 306-question slice", "",
              "| run | n | EM [95% CI] | contains [95% CI] |", "|---|---|---|---|"]
    for mem, rows in runs.items():
        if mem.endswith("_raw_n") or not isinstance(rows, dict):
            continue
        keys = [k for k in subset_keys(items, "all") if k in rows]
        if keys:
            lines.append(score_line(rows, keys, f"MR {mem}"))
            if nr.get(mem):
                kk = [k for k in keys if k in nr[mem]]
                lines.append(score_line(nr[mem], kk, f"NR {mem} (s1, matched)"))
            if sr.get(mem):
                kk = [k for k in keys if k in sr[mem]]
                lines.append(score_line(sr[mem], kk, f"SR {mem} (s2, matched)"))

    lines += ["", "## Subsets", "",
              "| run | subset | n | EM [95% CI] | contains [95% CI] |", "|---|---|---|---|---|"]
    for mem, rows in runs.items():
        if mem.endswith("_raw_n") or not isinstance(rows, dict):
            continue
        for sname in ("all", "leak-free", "leak-free-3digit", "2-4", "5-7", "8-10"):
            keys = [k for k in subset_keys(items, sname) if k in rows]
            if keys:
                lines.append(score_line(rows, keys, f"MR {mem} | {sname}"))

    lines += ["", "## Paired comparisons (MR minus baseline, same questions)", "",
              "| retriever | baseline | subset | n | dEM [95% CI] | dContains [95% CI] |",
              "|---|---|---|---|---|---|"]
    for mem, rows in runs.items():
        if mem.endswith("_raw_n") or not isinstance(rows, dict):
            continue
        for bname, base in (("NR (s1)", nr.get(mem)), ("SR (s2)", sr.get(mem))):
            if not base:
                continue
            for sname in ("all", "leak-free", "2-4", "5-7", "8-10"):
                keys = subset_keys(items, sname)
                re = paired(rows, base, keys, 0)
                rc = paired(rows, base, keys, 1)
                if re:
                    lines.append(f"| {mem} | {bname} | {sname} | {re[0]} | "
                                 f"{re[1]:+.3f} [{re[2][0]:+.3f}, {re[2][1]:+.3f}] | "
                                 f"{rc[1]:+.3f} [{rc[2][0]:+.3f}, {rc[2][1]:+.3f}] |")
    lines += ["", "## BM25 (SparseRAG) minus e5 (DenseRAG), same questions", "",
              "| subset | n | dEM [95% CI] | dContains [95% CI] |", "|---|---|---|---|"]
    if "SparseRAG" in runs and "DenseRAG" in runs:
        for sname in ("all", "leak-free", "2-4", "5-7", "8-10"):
            keys = subset_keys(items, sname)
            re = paired(runs["SparseRAG"], runs["DenseRAG"], keys, 0)
            rc = paired(runs["SparseRAG"], runs["DenseRAG"], keys, 1)
            if re:
                lines.append(f"| {sname} | {re[0]} | {re[1]:+.3f} [{re[2][0]:+.3f}, {re[2][1]:+.3f}] | "
                             f"{rc[1]:+.3f} [{rc[2][0]:+.3f}, {rc[2][1]:+.3f}] |")

    lines += ["", "## MR trace diagnostics", "",
              "| retriever | rows | calls | prompt tok | completion tok | duplicate branch pairs | "
              "selection fallbacks | truncated thought calls | final-path depth distribution |",
              "|---|---|---|---|---|---|---|---|---|"]
    for mem, rows in runs.items():
        if mem.endswith("_raw_n") or not isinstance(rows, dict) or not rows:
            continue
        calls = sum(int(r.get("calls") or 0) for r in rows.values())
        pt = sum(int(r.get("prompt_tokens") or 0) for r in rows.values())
        ct = sum(int(r.get("completion_tokens") or 0) for r in rows.values())
        dup = sum(int((r.get("trace") or {}).get("duplicate_candidate_pairs") or 0) for r in rows.values())
        pairs = sum(int((r.get("trace") or {}).get("candidate_pairs_total") or 0) for r in rows.values())
        fall = sum(int(((r.get("trace") or {}).get("stats") or {}).get("selection_fallbacks") or 0)
                   for r in rows.values())
        trunc = sum(sum((((r.get("trace") or {}).get("stats") or {}).get("truncations") or {}).values())
                    for r in rows.values())
        depths = Counter((r.get("trace") or {}).get("final_path_depth") for r in rows.values())
        dist = " ".join(f"{k}:{v}" for k, v in sorted(depths.items(), key=lambda x: (x[0] is None, x[0])))
        lines.append(f"| MR {mem} | {len(rows)} | {calls} | {pt} | {ct} | {dup}/{pairs} | {fall} | "
                     f"{trunc} | {dist} |")
    lines += ["", "## Timing", ""]
    if timing:
        lines += ["| stage | file | questions | calls | prompt tok | completion tok | wall s | workers | failures | deadline hit |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for t in timing:
            lines.append(f"| {t.get('stage')} | {t.get('file')} | {t.get('questions_done')} | "
                         f"{t.get('calls')} | {t.get('prompt_tokens')} | {t.get('completion_tokens')} | "
                         f"{t.get('wall_s')} | {t.get('workers')} | {t.get('failures')} | {t.get('deadline_hit')} |")
        total_calls = sum(int(t.get("calls") or 0) for t in timing if t.get("stage") == "full")
        total_wall = sum(float(t.get("wall_s") or 0) for t in timing if t.get("stage") == "full")
        lines += ["", f"Full-stage generation: {total_calls} calls in {total_wall / 60:.1f} minutes of job wall time."]
    else:
        lines.append("No timing.jsonl found.")

    lines += ["", "## Caveats", "",
              f"- {a.hardware[0].upper() + a.hardware[1:]}, one slice, one experiment seed. Not a reproduction of the paper's A800/full-precision numbers.",
              "- MR branch calls are sampled; NR/SR baselines are historical greedy runs. Any MR-vs-NR/SR difference mixes structure with decoding.",
              "- The MR reference keeps every old path and does not force depth growth; selected paths can stay shallow. Depths above are observed, not assumed.",
              "- Intervals are over questions only; no correction for the many subset/retriever comparisons in this report.",
              ""]
    out = a.out or os.path.join(a.results_dir, "summary_mr.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write = {"validation": validation, "generated_from": os.path.abspath(a.results_dir)}
    with open(os.path.join(a.results_dir, "analysis_validation.json"), "w", encoding="utf-8") as f:
        json.dump(write, f, indent=2, sort_keys=True)
        f.write("\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
