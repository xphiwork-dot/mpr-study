"""Turn the raw T4 outputs into markdown tables. Needs numpy for the bootstrap; every number printed comes from a result file.

  python experiments/_shared/summarize.py grid      --tag s1 --slice results/t4/slices/s900.json   -> results/t4/summary_<tag>.md
  python experiments/_shared/summarize.py retrieval                                                -> results/t4/summary_retrieval.md
  python experiments/_shared/summarize.py throughput                                               -> results/t4/summary_throughput.md

grid: for each results/t4/<tag>/*.jsonl run file: EM and lenient "contains" (experiments/_shared/score.py rules), overall
  and by hop group, with a 95% bootstrap interval over questions; then the same on the leak-free subset (questions
  whose text shares no number with their own gold statements, the rule of experiments/00_probes_retrieval_leak/leak_check.py; flags are stored in
  the slice file), then paired differences between memory types with a paired bootstrap over the same questions.
compare: one-call (NaiveReason, tag s1) vs chain-of-thought (SequentialReason, tag s2) on the questions of the s2 slice,
  per memory type: both scores with 95% intervals and the paired SR-minus-NR difference (paired bootstrap), overall,
  leak-free and by hop group.   python experiments/_shared/summarize.py compare --tag s2 --nr-tag s1 --slice results/t4/slices/s306.json
  Provenance: summarize.py is written by us, no upstream reference (the compare mode too).
topk: top-k sweep, NaiveReason, BM25 and e5, k = 10/20/30/50 (k = 20 from tag s1, the others from tag s3_topk) on the s900 slice:
  EM and contains per k, overall / leak-free / hop group, with 95% bootstrap intervals, the retrieval recall@k and all-gold@k of the
  same questions (from results/t4/retrieval), paired differences between k values and BM25 vs e5.
  python3 experiments/_shared/summarize.py topk                      -> results/t4/summary_s3_topk_sweep.md
steps: reasoning-step sweep, SequentialReason with 2/3/5/8 steps (5 from tag s2, the others from tag s3_steps) and the one-call
  baseline (tag s1, k = 20) on the s306 slice: EM and contains per setting, paired differences between step counts and vs one-call.
  python3 experiments/_shared/summarize.py steps                     -> results/t4/summary_s3_steps_sweep.md
  Provenance: written by us, no upstream reference.
retrieval: recall@k and all-gold@k per hop over all 12 users: mean over questions, std of the 12 per-user means,
  and a 95% bootstrap interval over questions.
"""

import argparse
import glob
import gzip
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # score.py sits next to this file
from score import contains, em  # noqa: E402

T4 = os.path.join(HERE, "..", "..", "results", "t4")
GROUPS = {"2-4": range(2, 5), "5-7": range(5, 8), "8-10": range(8, 11)}
B = 2000


def boot_ci(vals, seed=0, b=B):
    """95% percentile bootstrap interval of the mean of vals (numpy)."""
    import numpy as np

    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.sort(
        np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(b)])
    )
    return float(means[int(0.025 * b)]), float(means[int(0.975 * b) - 1])


def fmt(m, ci=None):
    return f"{m:.3f}" if ci is None else f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def load_runs(tag):
    runs = {}
    for p in sorted(glob.glob(os.path.join(T4, tag, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        if name == "timing":
            continue
        rows = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[f"{r['uid']}:{r['rc_id']}"] = r
        runs[name] = rows
    return runs


def score(rows, keys):
    em_v = [float(em(rows[k]["response"], rows[k]["ground_truth"])) for k in keys]
    ct_v = [float(contains(rows[k]["response"], rows[k]["ground_truth"])) for k in keys]
    return em_v, ct_v


def table_for(runs, meta, keys_filter, title, lines):
    lines += [
        f"### {title}",
        "",
        "| run | n | EM [95% CI] | contains [95% CI] | EM 2-4 | EM 5-7 | EM 8-10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in runs.items():
        keys = [k for k in rows if keys_filter(k)]
        if not keys:
            continue
        e, c = score(rows, keys)
        gcells = []
        for g, rng in GROUPS.items():
            ks = [k for k in keys if int(rows[k]["k_hop"]) in rng]
            gcells.append(
                f"{sum(score(rows, ks)[0]) / len(ks):.3f} (n={len(ks)})" if ks else "-"
            )
        lines.append(
            f"| {name} | {len(keys)} | {fmt(sum(e) / len(e), boot_ci(e))} | {fmt(sum(c) / len(c), boot_ci(c))} | "
            + " | ".join(gcells)
            + " |"
        )
    lines.append("")


def paired(runs, a, b, keys, metric):
    ka = [k for k in keys if k in runs[a] and k in runs[b]]
    if not ka:
        return None
    va = score(runs[a], ka)[metric]
    vb = score(runs[b], ka)[metric]
    d = [x - y for x, y in zip(va, vb)]
    return len(ka), sum(d) / len(d), boot_ci(d, seed=1)


def grid(tag, slice_path):
    sl = json.load(open(slice_path))
    flag = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    runs = load_runs(tag)
    if not runs:
        sys.exit(f"no run files in results/t4/{tag}")
    lines = [
        f"# Results `{tag}` (generated by experiments/_shared/summarize.py)",
        "",
        f"Slice `{os.path.basename(slice_path)}`: {sl['meta']['n']} questions, seed {sl['meta']['seed']}, "
        f"{sl['meta']['per_hop']} per hop (hops 2-10, all 12 users). Intervals: 95% bootstrap over questions ({B} resamples). "
        f"Leaking questions in the slice: any-number rule {sl['meta']['leak_any']}, 3+ digit rule {sl['meta']['leak_big']}.",
        "",
    ]
    table_for(runs, sl["meta"], lambda k: True, "All questions", lines)
    table_for(
        runs,
        sl["meta"],
        lambda k: not flag[k]["leak_any"],
        "Leak-free: questions sharing no number with their gold statements (any-number rule)",
        lines,
    )
    table_for(
        runs,
        sl["meta"],
        lambda k: not flag[k]["leak_big"],
        "Leak-free: no shared number with 3+ digits",
        lines,
    )
    table_for(
        runs,
        sl["meta"],
        lambda k: flag[k]["leak_any"],
        "Leaking questions only (any-number rule), for contrast",
        lines,
    )
    # per-hop EM
    lines += [
        "### EM by hop (all questions)",
        "",
        "| run | " + " | ".join(str(h) for h in range(2, 11)) + " |",
        "|---|" + "---|" * 9,
    ]
    for name, rows in runs.items():
        cells = []
        for h in range(2, 11):
            ks = [k for k in rows if int(rows[k]["k_hop"]) == h]
            cells.append(f"{sum(score(rows, ks)[0]) / len(ks):.2f}" if ks else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    # paired differences between memory types within the same reasoning/topk
    names = list(runs)
    pairs = []
    for a in names:
        for b in names:
            if (
                a < b
                and a.rsplit("_", 1)[0].split("_")[0]
                == b.rsplit("_", 1)[0].split("_")[0]
            ):
                pairs.append((a, b))
    if pairs:
        lines += [
            "### Paired differences (A minus B, same questions; paired bootstrap)",
            "",
            "Only runs of the same reasoning structure are paired. n is the number of questions present in both runs.",
            "",
            "| A | B | subset | n | dEM [95% CI] | dContains [95% CI] |",
            "|---|---|---|---|---|---|",
        ]
        for a, b in pairs:
            for sname, flt in (
                ("all", lambda k: True),
                ("leak-free", lambda k: not flag[k]["leak_any"]),
            ):
                keys = [k for k in flag if flt(k)]
                r1, r2 = paired(runs, a, b, keys, 0), paired(runs, a, b, keys, 1)
                if r1:
                    lines.append(
                        f"| {a} | {b} | {sname} | {r1[0]} | {r1[1]:+.3f} [{r1[2][0]:+.3f}, {r1[2][1]:+.3f}] | {r2[1]:+.3f} [{r2[2][0]:+.3f}, {r2[2][1]:+.3f}] |"
                    )
        lines.append("")
    # token use / timing
    tpath = os.path.join(T4, tag, "timing.jsonl")
    if os.path.exists(tpath):
        lines += [
            "### Wall time and tokens per run file (from the VM)",
            "",
            "| file | questions | calls | prompt tokens | completion tokens | wall s | in tok/s | out tok/s |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for line in open(tpath):
            t = json.loads(line)
            if (
                t["calls"] == 0
            ):  # a failed attempt (all questions errored) is not a timing
                continue
            lines.append(
                f"| {t['file']} | {t['questions']} | {t['calls']} | {t['prompt_tokens']} | {t['completion_tokens']} | {t['wall_s']} | "
                f"{t['prompt_tokens'] / t['wall_s']:.0f} | {t['completion_tokens'] / t['wall_s']:.0f} |"
            )
        lines.append("")
    out = os.path.join(T4, f"summary_{tag}.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def compare(sr_tag, nr_tag, slice_path):
    sl = json.load(open(slice_path))
    flag = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    sr, nr = load_runs(sr_tag), load_runs(nr_tag)
    lines = [
        f"# One-call (NR, `{nr_tag}`) vs chain-of-thought (SR, `{sr_tag}`) on the same questions (generated by experiments/_shared/summarize.py)",
        "",
        f"Slice `{os.path.basename(slice_path)}`: {sl['meta']['n']} questions. Same model, same k = 20, same retrieval; SR re-retrieves after each of up to 5 thoughts. "
        f"Intervals: 95% bootstrap over questions ({B} resamples); the difference is SR minus NR on the same questions (paired bootstrap).",
        "",
    ]
    subsets = (
        ("all", lambda k: True),
        ("leak-free (any-number rule)", lambda k: not flag[k]["leak_any"]),
        ("hops 2-4", lambda k: int(flag[k]["hop"]) in GROUPS["2-4"]),
        ("hops 5-7", lambda k: int(flag[k]["hop"]) in GROUPS["5-7"]),
        ("hops 8-10", lambda k: int(flag[k]["hop"]) in GROUPS["8-10"]),
    )
    lines += [
        "| memory | subset | n | NR EM [95% CI] | SR EM [95% CI] | SR-NR dEM [95% CI] | NR contains | SR contains | SR-NR dContains [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mem in ("Ignoramus", "Oracle", "SparseRAG", "DenseRAG"):
        a = next((n for n in sr if n.startswith(f"SequentialReason_{mem}_")), None)
        b = next((n for n in nr if n.startswith(f"NaiveReason_{mem}_")), None)
        if not a or not b:
            continue
        for sname, flt in subsets:
            keys = [k for k in flag if flt(k) and k in sr[a] and k in nr[b]]
            if not keys:
                continue
            ea, ca = score(sr[a], keys)
            eb, cb = score(nr[b], keys)
            de = [x - y for x, y in zip(ea, eb)]
            dc = [x - y for x, y in zip(ca, cb)]
            n = len(keys)
            lo, hi = boot_ci(de, seed=1)
            clo, chi = boot_ci(dc, seed=1)
            lines.append(
                f"| {mem} | {sname} | {n} | {fmt(sum(eb) / n, boot_ci(eb))} | {fmt(sum(ea) / n, boot_ci(ea))} | {sum(de) / n:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                f"{sum(cb) / n:.3f} | {sum(ca) / n:.3f} | {sum(dc) / n:+.3f} [{clo:+.3f}, {chi:+.3f}] |"
            )
    lines.append("")
    out = os.path.join(T4, f"summary_{sr_tag}_vs_{nr_tag}.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


SUBSETS = (
    ("all", lambda f: True),
    ("leak-free", lambda f: not f["leak_any"]),
    ("hops 2-4", lambda f: int(f["hop"]) in GROUPS["2-4"]),
    ("hops 5-7", lambda f: int(f["hop"]) in GROUPS["5-7"]),
    ("hops 8-10", lambda f: int(f["hop"]) in GROUPS["8-10"]),
)


def pick(tag, prefix):
    """Run file of results/t4/<tag> whose name starts with prefix (exact name up to the suffix)."""
    runs = load_runs(tag)
    return runs.get(prefix)


def sweep_report(title, intro, settings, flag, pair_list, out, recall=None):
    """settings: ordered {label: {memory: rows}}; pair_list: (label_a, label_b) paired A minus B on the same questions."""
    mems = ("SparseRAG", "DenseRAG")
    lines = [
        f"# {title} (generated by experiments/_shared/summarize.py)",
        "",
        intro,
        "",
    ]
    # 1. scores per setting and subset
    hdr = "| memory | setting | subset | n | EM [95% CI] | contains [95% CI] |" + (
        " recall@k | all-gold@k |" if recall else ""
    )
    lines += [
        "## Scores",
        "",
        hdr,
        "|---|---|---|---|---|---|" + ("---|---|" if recall else ""),
    ]
    for mem in mems:
        for lab, byMem in settings.items():
            rows = byMem.get(mem)
            if rows is None:
                continue
            for sname, flt in SUBSETS:
                keys = [k for k in flag if flt(flag[k]) and k in rows]
                if not keys:
                    continue
                e, c = score(rows, keys)
                cell = f"| {mem} | {lab} | {sname} | {len(keys)} | {fmt(sum(e) / len(e), boot_ci(e))} | {fmt(sum(c) / len(c), boot_ci(c))} |"
                if recall:
                    rr = [recall[mem][lab][k] for k in keys]
                    cell += f" {sum(r[0] for r in rr) / len(rr):.3f} | {sum(r[1] for r in rr) / len(rr):.3f} |"
                lines.append(cell)
    lines.append("")
    # 2. EM by single hop
    lines += [
        "## EM by hop (all questions in the setting)",
        "",
        "| memory | setting | " + " | ".join(str(h) for h in range(2, 11)) + " |",
        "|---|---|" + "---|" * 9,
    ]
    for mem in mems:
        for lab, byMem in settings.items():
            rows = byMem.get(mem)
            if rows is None:
                continue
            cells = []
            for h in range(2, 11):
                ks = [k for k in flag if int(flag[k]["hop"]) == h and k in rows]
                cells.append(f"{sum(score(rows, ks)[0]) / len(ks):.2f}" if ks else "-")
            lines.append(f"| {mem} | {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    # 3. paired differences between settings
    lines += [
        "## Paired differences between settings (A minus B on the same questions; paired bootstrap, 95%)",
        "",
        "| memory | A | B | subset | n | dEM [95% CI] | dContains [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for mem in mems:
        for la, lb in pair_list:
            ra, rb = settings.get(la, {}).get(mem), settings.get(lb, {}).get(mem)
            if ra is None or rb is None:
                continue
            for sname, flt in SUBSETS:
                keys = [k for k in flag if flt(flag[k]) and k in ra and k in rb]
                if not keys:
                    continue
                ea, ca = score(ra, keys)
                eb, cb = score(rb, keys)
                de = [x - y for x, y in zip(ea, eb)]
                dc = [x - y for x, y in zip(ca, cb)]
                n = len(keys)
                lo, hi = boot_ci(de, seed=1)
                clo, chi = boot_ci(dc, seed=1)
                lines.append(
                    f"| {mem} | {la} | {lb} | {sname} | {n} | {sum(de) / n:+.3f} [{lo:+.3f}, {hi:+.3f}] | {sum(dc) / n:+.3f} [{clo:+.3f}, {chi:+.3f}] |"
                )
    lines.append("")
    # 4. BM25 vs e5 per setting
    lines += [
        "## BM25 (SparseRAG) minus e5 (DenseRAG), same questions",
        "",
        "| setting | subset | n | dEM [95% CI] | dContains [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for lab, byMem in settings.items():
        ra, rb = byMem.get("SparseRAG"), byMem.get("DenseRAG")
        if ra is None or rb is None:
            continue
        for sname, flt in SUBSETS:
            keys = [k for k in flag if flt(flag[k]) and k in ra and k in rb]
            if not keys:
                continue
            ea, ca = score(ra, keys)
            eb, cb = score(rb, keys)
            de = [x - y for x, y in zip(ea, eb)]
            dc = [x - y for x, y in zip(ca, cb)]
            n = len(keys)
            lo, hi = boot_ci(de, seed=1)
            clo, chi = boot_ci(dc, seed=1)
            lines.append(
                f"| {lab} | {sname} | {n} | {sum(de) / n:+.3f} [{lo:+.3f}, {hi:+.3f}] | {sum(dc) / n:+.3f} [{clo:+.3f}, {chi:+.3f}] |"
            )
    lines.append("")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def topk_sweep():
    sl = json.load(open(os.path.join(T4, "slices", "s900.json")))
    flag = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    s1, s3 = load_runs("s1"), load_runs("s3_topk")
    settings = {}
    for k in (10, 20, 30, 50):
        src = s1 if k == 20 else s3
        settings[f"k={k}"] = {
            m: src.get(f"NaiveReason_{m}_k{k}") for m in ("SparseRAG", "DenseRAG")
        }
    rec = {}
    with gzip.open(os.path.join(T4, "retrieval", "recall.csv.gz"), "rt") as f:
        head = f.readline().strip().split(",")
        for l in f:
            r = dict(zip(head, l.strip().split(",")))
            rec[f"{r['uid']}:{r['rc_id']}"] = r
    recall = {
        m: {
            f"k={k}": {
                q: (float(rec[q][f"{sysn}_rec{k}"]), float(rec[q][f"{sysn}_all{k}"]))
                for q in flag
            }
            for k in (10, 20, 30, 50)
        }
        for m, sysn in (("SparseRAG", "bm25"), ("DenseRAG", "e5"))
    }
    pairs = [
        ("k=20", "k=10"),
        ("k=30", "k=20"),
        ("k=50", "k=30"),
        ("k=30", "k=10"),
        ("k=50", "k=10"),
        ("k=50", "k=20"),
    ]
    intro = (
        f"NaiveReason (one call per question), Qwen2.5-7B-Instruct-AWQ on a Colab T4, greedy, slice `s900.json` ({sl['meta']['n']} questions, "
        f"{sl['meta']['per_hop']} per hop, all 12 users). k = 20 is the committed session-1 run (tag `s1`); k = 10, 30, 50 are tag `s3_topk`. "
        "recall@k / all-gold@k are the retrieval-only numbers of the same questions (no LLM), mean over the questions of the subset. "
        f"Intervals: 95% bootstrap over questions ({B} resamples); paired differences use a paired bootstrap on the same questions. "
        "Leak-free = questions sharing no number with their gold statements (any-number rule)."
    )
    sweep_report(
        "Top-k sweep (NaiveReason, BM25 and e5)",
        intro,
        settings,
        flag,
        pairs,
        os.path.join(T4, "summary_s3_topk_sweep.md"),
        recall,
    )


def steps_sweep():
    sl = json.load(open(os.path.join(T4, "slices", "s306.json")))
    flag = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    s1, s2, s3 = load_runs("s1"), load_runs("s2"), load_runs("s3_steps")
    settings = {
        "one call (NR)": {
            m: s1.get(f"NaiveReason_{m}_k20") for m in ("SparseRAG", "DenseRAG")
        }
    }
    for st in (2, 3, 5, 8):
        src = s2 if st == 5 else s3
        runs = {
            m: src.get(f"SequentialReason_{m}_k20_s{st}")
            for m in ("SparseRAG", "DenseRAG")
        }
        if any(v is not None for v in runs.values()):
            settings[f"{st} steps"] = runs
    labs = list(settings)
    pairs = [(a, b) for a, b in zip(labs[2:], labs[1:-1])] + [
        (a, "one call (NR)") for a in labs[1:]
    ]
    pairs += [(labs[-1], labs[1])] if len(labs) > 3 else []
    pairs += (
        [(labs[-1], "3 steps")] if "3 steps" in labs and labs[-1] != "3 steps" else []
    )
    intro = (
        f"SequentialReason with 2, 3, 5 and 8 steps and the one-call NaiveReason baseline, k = 20, BM25 and e5, Qwen2.5-7B-Instruct-AWQ on a Colab T4, greedy, "
        f"slice `s306.json` ({sl['meta']['n']} questions, {sl['meta']['per_hop']} per hop, all 12 users; a subset of the 900-question slice). "
        "One call = tag `s1` (k = 20) restricted to these questions; 5 steps = committed session-2 run (tag `s2`); 2, 3 and 8 steps = tag `s3_steps`. "
        f"Intervals: 95% bootstrap over questions ({B} resamples); paired differences use a paired bootstrap on the same questions. "
        "Leak-free = questions sharing no number with their gold statements (any-number rule)."
    )
    sweep_report(
        "Reasoning-step sweep (SequentialReason, BM25 and e5)",
        intro,
        settings,
        flag,
        pairs,
        os.path.join(T4, "summary_s3_steps_sweep.md"),
    )


def retrieval():
    path = os.path.join(T4, "retrieval", "recall.csv.gz")
    with gzip.open(path, "rt") as f:
        head = f.readline().strip().split(",")
        rows = [dict(zip(head, l.strip().split(","))) for l in f]
    lines = [
        "# Retrieval-only error bars (generated by experiments/_shared/summarize.py)",
        "",
        f"{len(rows)} questions, all 12 users, query = the question text, no LLM. BM25 = bm25s with English stopwords; e5 = e5-base-v2 "
        "(fp16, no query:/passage: prefixes, as upstream). 'std users' = standard deviation of the 12 per-user means; "
        f"interval = 95% bootstrap over questions ({B} resamples).",
        "",
    ]
    for k in (20, 10, 30, 50):
        for metric, mname in (("rec", "recall"), ("all", "all-gold")):
            lines += [
                f"### {mname}@{k} per hop",
                "",
                "| hop | n | BM25 mean | BM25 std users | BM25 95% CI | e5 mean | e5 std users | e5 95% CI |",
                "|---|---|---|---|---|---|---|---|",
            ]
            hops = sorted({int(r["hop"]) for r in rows})
            for h in hops + ["all"]:
                sel = [r for r in rows if h == "all" or int(r["hop"]) == h]
                cells = [str(h), str(len(sel))]
                for sysn in ("bm25", "e5"):
                    vals = [float(r[f"{sysn}_{metric}{k}"]) for r in sel]
                    per_user = {}
                    for r, v in zip(sel, vals):
                        per_user.setdefault(r["uid"], []).append(v)
                    ums = [sum(v) / len(v) for v in per_user.values()]
                    ci = boot_ci(vals, b=500 if h == "all" else B)
                    cells += [
                        f"{sum(vals) / len(vals):.3f}",
                        f"{statistics.pstdev(ums):.3f}",
                        f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                    ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    out = os.path.join(T4, "summary_retrieval.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


def throughput():
    path = os.path.join(T4, "throughput", "throughput.jsonl")
    lines = [
        "# vLLM throughput on the T4 (generated by experiments/_shared/summarize.py)",
        "",
        "Qwen2.5-7B-Instruct-AWQ, fp16, vLLM `--max-num-seqs 64`, real grid prompts (BM25 memory, top-20).",
        "",
        "| phase | concurrent | wall s | questions | calls | calls/s | questions/min | prompt tok/s | completion tok/s | median call s | avg prompt tok |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for line in open(path):
        r = json.loads(line)
        if r["phase"] == "single_request_decode":
            lines.append(
                f"| {r['phase']} | 1 | {r['wall_s']} | - | 1 | - | - | - | {r['completion_tok_per_s']} | - | - |"
            )
        else:
            lines.append(
                f"| {r['phase']} | {r['workers']} | {r['wall_s']} | {r['questions']} | {r['calls']} | {r['calls_per_s']} | {r['questions_per_min']} | "
                f"{r['prompt_tok_per_s']} | {r['completion_tok_per_s']} | {r['median_call_latency_s']} | {r['avg_prompt_tokens_per_call']} |"
            )
    out = os.path.join(T4, "summary_throughput.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "what", choices=["grid", "compare", "topk", "steps", "retrieval", "throughput"]
    )
    ap.add_argument("--nr-tag", default="s1")
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--slice", default=os.path.join(T4, "slices", "s900.json"))
    a = ap.parse_args()
    {
        "grid": lambda: grid(a.tag, a.slice),
        "compare": lambda: compare(a.tag, a.nr_tag, a.slice),
        "topk": topk_sweep,
        "steps": steps_sweep,
        "retrieval": retrieval,
        "throughput": throughput,
    }[a.what]()
