"""Independent local validation of the downloaded S6 DR results.

    python3 experiments/s6_kaggle_dr/validate_dr.py \
        --runs results/kaggle/dr/runs --slice results/t4/slices/s306.json \
        --qa-dir data/mpr_dataset/question_answer --settings experiments/s6_kaggle_dr/settings.json

Checks coverage (exactly the 306 slice keys per memory, no extras), ground-truth consistency with the
released data and with the archived S1 rows, DR row invariants (call count 3-7, state/retrieval structure,
token caps, parser metadata, settings hash), duplicate/cut jsonl lines, and failure records. Writes
`validation_local.json` next to the runs directory and exits non-zero if any check fails.

Provenance: original to this repo.
"""
import argparse
import json
import os
import sys


def read_jsonl(path):
    rows, cut = [], 0
    if not os.path.exists(path):
        return rows, cut
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                cut += 1
    return rows, cut


def unique(rows):
    by, dup, order = {}, 0, []
    for r in rows:
        k = f"{r['uid']}:{r['rc_id']}"
        if k in by:
            dup += 1
        else:
            order.append(k)
        by[k] = r
    return [by[k] for k in order], dup


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(repo, "results", "kaggle", "dr", "runs"))
    ap.add_argument("--slice", default=os.path.join(repo, "results", "t4", "slices", "s306.json"))
    ap.add_argument("--qa-dir", default=os.path.join(repo, "data", "mpr_dataset", "question_answer"))
    ap.add_argument("--settings", default=os.path.join(here, "settings.json"))
    ap.add_argument("--archived-nr", default=os.path.join(repo, "results", "t4", "s1"))
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    settings = json.load(open(a.settings, encoding="utf-8"))
    sys.path.insert(0, here)
    import dr

    sl = json.load(open(a.slice, encoding="utf-8"))
    wanted = {f"{i['uid']}:{i['rc_id']}" for i in sl["items"]}
    flags = {f"{i['uid']}:{i['rc_id']}": i for i in sl["items"]}
    qa_cache = {}

    def qa(uid):
        if uid not in qa_cache:
            with open(os.path.join(a.qa_dir, f"question_answer_{uid}.json"), encoding="utf-8") as f:
                qa_cache[uid] = json.load(f)
        return qa_cache[uid]

    report = {"settings_sha256": dr.config_hash(settings), "slice_keys": len(wanted), "memories": {},
              "ok": True}
    for mem in ("SparseRAG", "DenseRAG"):
        path = os.path.join(a.runs, f"DecompositionReason_{mem}_k20.jsonl")
        raw, cut = read_jsonl(path)
        rows, dup = unique(raw)
        keys = {f"{r['uid']}:{r['rc_id']}" for r in rows}
        failures, fcut = read_jsonl(os.path.join(a.runs, f"failures_{mem}.jsonl"))
        bad = []
        calls_hist = {c: 0 for c in range(3, 8)}

        def fail(r, msg):
            bad.append({"key": f"{r['uid']}:{r['rc_id']}", "error": msg})

        for r in rows:
            k = f"{r['uid']}:{r['rc_id']}"
            q = qa(r["uid"])["qa"][str(r["k_hop"])][r["rc_id"]]
            n = len(r.get("sub_questions") or [])
            checks = [
                (r.get("ground_truth") == q["answer"], "ground_truth != dataset answer"),
                (r.get("calls") == 1 + n + 1, f"calls {r.get('calls')} != 1+{n}+1"),
                (3 <= (r.get("calls") or 0) <= 7, "calls outside 3..7"),
                (len(r.get("solve_queries") or []) == n, "solve_queries length != sub_questions"),
                (len(r.get("solve_retrieved") or []) == n, "solve_retrieved length != sub_questions"),
                (len(r.get("current_states") or []) == n, "current_states length != sub_questions"),
                (all(len(x or []) == 20 for x in r.get("solve_retrieved", [])), "a solve retrieval is not top-20"),
                (len(r.get("merge_retrieved") or []) == 20, "merge retrieval is not top-20"),
                (bool(str(r.get("response") or "").strip()), "empty final response"),
                (len(r.get("call_trace") or []) == r.get("calls"), "call_trace length != calls"),
                ([c["kind"] for c in r.get("call_trace", [])] ==
                 ["Decomposition_Divide"] + ["Decomposition_Solve"] * n + ["Decomposition_Merge"],
                 "call_trace kinds are not Divide, Solve*n, Merge"),
                (all(c["prompt_tokens"] + c["max_tokens"] <= a.max_model_len for c in r.get("call_trace", [])),
                 "prompt + max_tokens exceeds max_model_len"),
                (r.get("settings_sha256") == report["settings_sha256"], "settings hash differs from the frozen settings"),
                ((r.get("parser") or {}).get("policy") == settings["parser"]["policy"], "parser policy differs"),
                (r.get("memory") == mem, "memory name differs"),
                (r.get("reason") == "DecompositionReason", "reason name differs"),
            ]
            for ok, msg in checks:
                if not ok:
                    fail(r, msg)
            calls_hist[r.get("calls", 0)] = calls_hist.get(r.get("calls", 0), 0) + 1
            # merge query must equal the answered state reconstructed from the stored parts
            if r.get("sub_questions") and r.get("sub_answers") is not None:
                expected_state = dr.build_answered_state(r["sub_questions"], r["sub_answers"])
                if r.get("merge_query") != expected_state:
                    fail(r, "merge_query != answered current_state")
                for i, sq in enumerate(r["sub_questions"]):
                    if r["solve_queries"][i] != dr.solve_retrieval_key(q["question"], sq):
                        fail(r, f"solve query {i} != exact upstream key")
                    if r["current_states"][i] != dr.build_current_state(r["sub_questions"], r["sub_answers"][:i], i):
                        fail(r, f"current_state {i} not reconstructible")
            if r.get("first_retrieved") and r.get("solve_retrieved") and r["first_retrieved"] != r["solve_retrieved"][0]:
                fail(r, "first_retrieved != first solve retrieval")

        # archived S1 ground truth must agree where the key exists (guards against a key mismatch)
        gt_mismatch = 0
        archived = os.path.join(a.archived_nr, f"NaiveReason_{mem}_k20.jsonl")
        if os.path.exists(archived):
            base, _ = read_jsonl(archived)
            base = {f"{r['uid']}:{r['rc_id']}": r for r in base}
            for r in rows:
                k = f"{r['uid']}:{r['rc_id']}"
                if k in base and base[k]["ground_truth"] != r["ground_truth"]:
                    gt_mismatch += 1

        mem_r = {"rows_raw": len(raw), "rows_unique": len(rows), "duplicates": dup, "cut_tail_lines": cut,
                 "missing": sorted(wanted - keys)[:10], "missing_count": len(wanted - keys),
                 "extra_count": len(keys - wanted), "failures": len(failures), "failure_cut_lines": fcut,
                 "failure_stages": sorted({f.get("stage") for f in failures}),
                 "invalid_rows": len(bad), "invalid_examples": bad[:10],
                 "call_histogram": calls_hist,
                 "parser_fallback_rows": sum(1 for r in rows if (r.get("parser") or {}).get("fallback")),
                 "rows_with_dropped_blank": sum(1 for r in rows if (r.get("parser") or {}).get("dropped_blank")),
                 "rows_with_over_limit": sum(1 for r in rows if (r.get("parser") or {}).get("over_limit")),
                 "empty_sub_answers": sum(1 for r in rows for x in r.get("sub_answers", []) if not str(x).strip()),
                 "archived_ground_truth_mismatches": gt_mismatch,
                 "leak_free_rows": sum(1 for r in rows if not flags[f"{r['uid']}:{r['rc_id']}"]["leak_any"])}
        mem_r["ok"] = (mem_r["missing_count"] == 0 and mem_r["extra_count"] == 0 and dup == 0
                       and len(failures) == 0 and cut == 0 and not bad and gt_mismatch == 0)
        report["memories"][mem] = mem_r
        report["ok"] &= mem_r["ok"]

    report["complete_612"] = all(m["rows_unique"] == 306 for m in report["memories"].values())
    out = os.path.join(os.path.dirname(a.runs.rstrip("/")), "validation_local.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for mem, m in report["memories"].items():
        print(f"{mem}: {m['rows_unique']}/306 rows, missing={m['missing_count']} extra={m['extra_count']} "
              f"dups={m['duplicates']} failures={m['failures']} invalid={m['invalid_rows']} "
              f"cut_lines={m['cut_tail_lines']} ok={m['ok']}")
    print(f"complete_612={report['complete_612']} ok={report['ok']} -> {out}")
    sys.exit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
