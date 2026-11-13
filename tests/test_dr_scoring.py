"""GPU-free tests of the two result tools: validate_dr.py and summarize_dr.py.

These are the tools that turn downloaded Kaggle rows into the numbers we would report, so they are tested
on synthetic fixtures for the properties that matter: every DR row invariant the validator enforces,
coverage/missing/extra accounting, deduplication, and the paired-comparison denominators. The bootstrap
part of summarize_dr.py needs numpy; those tests skip on a numpy-less interpreter and the suite stays
green (run `python3 -m unittest discover -s tests` with the repo venv to exercise them).

No network, no GPU, no model.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP = os.path.join(REPO, "experiments", "s6_kaggle_dr")
SETTINGS_PATH = os.path.join(EXP, "settings.json")

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except Exception:  # noqa: BLE001
    HAVE_NUMPY = False


def load_dr():
    sys.path.insert(0, os.path.join(REPO, "experiments", "_shared"))
    sys.path.insert(0, EXP)
    spec = importlib.util.spec_from_file_location("dr_tools_under_test", os.path.join(EXP, "dr.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_summary():
    spec = importlib.util.spec_from_file_location("summarize_under_test", os.path.join(EXP, "summarize_dr.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SLICE_ITEMS = [
    {"uid": 0, "hop": 2, "rc_id": "RC000001#02", "n_gold": 1, "leak_any": False, "leak_big": False},
    {"uid": 0, "hop": 5, "rc_id": "RC000002#05", "n_gold": 2, "leak_any": True, "leak_big": True},
    {"uid": 1, "hop": 9, "rc_id": "RC000003#09", "n_gold": 1, "leak_any": False, "leak_big": False},
]
ANSWERS = {("0", "RC000001#02"): "41", ("0", "RC000002#05"): "7", ("1", "RC000003#09"): "12"}
RESPONSES = {("0", "RC000001#02"): "41", ("0", "RC000002#05"): "eight", ("1", "RC000003#09"): "12"}


def make_row(dr, settings, mem, item, n_sub=2, response=None):
    uid, rc, hop = item["uid"], item["rc_id"], item["hop"]
    question = f"question {rc}"
    answer = ANSWERS[(str(uid), rc)]
    subs = [f"sub {i} of {rc}" for i in range(n_sub)]
    answers = [f"partial {i}" for i in range(n_sub)]
    ids = [[f"MG{u:03d}" for u in range(20)] for _ in range(n_sub)]
    trace = [{"kind": "Decomposition_Divide", "prompt_sha256": "0" * 64, "prompt_tokens": 120,
              "completion_tokens": 40, "secs": 1.5, "max_tokens": 200, "empty_response": False}]
    trace += [{"kind": "Decomposition_Solve", "prompt_sha256": "1" * 64, "prompt_tokens": 900,
               "completion_tokens": 40, "secs": 2.0, "max_tokens": 200, "empty_response": False}
              for _ in range(n_sub)]
    trace += [{"kind": "Decomposition_Merge", "prompt_sha256": "2" * 64, "prompt_tokens": 900,
               "completion_tokens": 10, "secs": 1.0, "max_tokens": 64, "empty_response": False}]
    return {
        "uid": uid, "rc_id": rc, "k_hop": hop, "memory": mem, "reason": "DecompositionReason",
        "response": RESPONSES[(str(uid), rc)] if response is None else response,
        "ground_truth": answer,
        "divide_raw": "\n".join(subs),
        "parser": {"policy": settings["parser"]["policy"], "raw_lines": n_sub, "dropped_blank": 0,
                   "over_limit": 0, "fallback": False},
        "sub_questions": subs, "sub_answers": answers,
        "current_states": [dr.build_current_state(subs, answers[:i], i) for i in range(n_sub)],
        "solve_queries": [dr.solve_retrieval_key(question, s) for s in subs],
        "solve_retrieved": ids,
        "merge_query": dr.build_answered_state(subs, answers),
        "merge_retrieved": [f"MG9{u:02d}" for u in range(20)],
        "first_retrieved": ids[0],
        "calls": n_sub + 2,
        "prompt_tokens": sum(t["prompt_tokens"] for t in trace),
        "completion_tokens": sum(t["completion_tokens"] for t in trace),
        "secs": sum(t["secs"] for t in trace),
        "call_trace": trace,
        "settings_sha256": dr.config_hash(settings),
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_tree(tmp):
    """slice + question_answer data + valid DR/NR/SR run directories."""
    dr = load_dr()
    settings = json.load(open(SETTINGS_PATH, encoding="utf-8"))
    paths = {k: os.path.join(tmp, *v) for k, v in {
        "slice": ("slice.json",), "qa": ("data",), "dr": ("runs",), "nr": ("nr",), "sr": ("sr",)}.items()}
    with open(paths["slice"], "w", encoding="utf-8") as f:
        json.dump({"meta": {"seed": 0, "per_hop": 1, "n": len(SLICE_ITEMS)}, "items": SLICE_ITEMS}, f)
    os.makedirs(paths["qa"], exist_ok=True)
    for uid in {it["uid"] for it in SLICE_ITEMS}:
        qa = {"message": {}, "qa": {}}
        for it in SLICE_ITEMS:
            if it["uid"] != uid:
                continue
            qa["qa"].setdefault(str(it["hop"]), {})[it["rc_id"]] = {
                "question": f"question {it['rc_id']}", "answer": ANSWERS[(str(uid), it["rc_id"])],
                "message": [f"MG{uid:03d}"], "fold_tag": it["hop"]}
        with open(os.path.join(paths["qa"], f"question_answer_{uid}.json"), "w", encoding="utf-8") as f:
            json.dump(qa, f)
    dr_rows = {mem: [make_row(dr, settings, mem, it) for it in SLICE_ITEMS]
               for mem in ("SparseRAG", "DenseRAG")}
    for mem, rows in dr_rows.items():
        write_jsonl(os.path.join(paths["dr"], f"DecompositionReason_{mem}_k20.jsonl"), rows)
    for mem in ("SparseRAG", "DenseRAG"):
        base = [{"uid": it["uid"], "rc_id": it["rc_id"], "k_hop": it["hop"],
                 "response": RESPONSES[(str(it["uid"]), it["rc_id"])],
                 "ground_truth": ANSWERS[(str(it["uid"]), it["rc_id"])],
                 "first_retrieved": [f"MG{u:03d}" for u in range(20)]} for it in SLICE_ITEMS]
        write_jsonl(os.path.join(paths["nr"], f"NaiveReason_{mem}_k20.jsonl"), base)
        write_jsonl(os.path.join(paths["sr"], f"SequentialReason_{mem}_k20_s5.jsonl"), base)
    return paths, dr_rows


def run_script(script, args):
    cmd = [sys.executable, os.path.join(EXP, script), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=600)


class ReadRowsTests(unittest.TestCase):
    def test_last_row_wins_and_duplicates_are_counted(self):
        mod = load_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rows.jsonl")
            write_jsonl(path, [{"uid": 0, "rc_id": "RC1", "response": "old"},
                               {"uid": 0, "rc_id": "RC2", "response": "keep"},
                               {"uid": 0, "rc_id": "RC1", "response": "new"}])
            rows, dup = mod.read_rows(path)
            self.assertEqual(dup, 1)
            self.assertEqual(rows["0:RC1"]["response"], "new")
            self.assertEqual(sorted(rows), ["0:RC1", "0:RC2"])

    def test_missing_file_is_empty_not_an_error(self):
        mod = load_summary()
        rows, dup = mod.read_rows("/nonexistent/none.jsonl")
        self.assertEqual((rows, dup), ({}, 0))


@unittest.skipUnless(HAVE_NUMPY, "summarize_dr.py bootstrap needs numpy (use the repo venv)")
class PairedComparisonTests(unittest.TestCase):
    def test_paired_uses_shared_keys_and_reports_its_denominator(self):
        mod = load_summary()
        a = {k: {"response": "41", "ground_truth": "41"} for k in ("0:A", "0:B", "0:C")}
        b = {k: {"response": "41", "ground_truth": "41"} for k in ("0:A", "0:B")}
        n, d, ci = mod.paired(a, b, ["0:A", "0:B", "0:C"], "em")
        self.assertEqual(n, 2)                    # 0:C is dropped: absent on the b side
        self.assertEqual(d, 0.0)
        self.assertEqual(ci, (0.0, 0.0))

    def test_paired_difference_sign_and_ci_cover_it(self):
        mod = load_summary()
        a = {"0:A": {"response": "41", "ground_truth": "41"},
             "0:B": {"response": "no", "ground_truth": "7"}}
        b = {"0:A": {"response": "no", "ground_truth": "41"},
             "0:B": {"response": "no", "ground_truth": "7"}}
        n, d, ci = mod.paired(a, b, ["0:A", "0:B"], "em")
        self.assertEqual(n, 2)
        self.assertAlmostEqual(d, 0.5)
        self.assertLessEqual(ci[0], d)
        self.assertGreaterEqual(ci[1], d)

    def test_paired_without_overlap_is_none_not_zero(self):
        mod = load_summary()
        a = {"0:A": {"response": "41", "ground_truth": "41"}}
        b = {"0:B": {"response": "41", "ground_truth": "41"}}
        self.assertEqual(mod.paired(a, b, ["0:A", "0:B"], "em"), (0, None, None))


class ValidatorTests(unittest.TestCase):
    def _run(self, paths):
        return run_script("validate_dr.py", [
            "--runs", paths["dr"], "--slice", paths["slice"], "--qa-dir", paths["qa"],
            "--settings", SETTINGS_PATH, "--archived-nr", paths["nr"]])

    def test_accepts_a_complete_synthetic_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = make_tree(tmp)
            p = self._run(paths)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            report = json.load(open(os.path.join(tmp, "validation_local.json"), encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertFalse(report["complete_612"])          # 3 of 306 rows: partial, never "complete"
            self.assertEqual(report["memories"]["SparseRAG"]["rows_unique"], 3)
            self.assertEqual(report["memories"]["DenseRAG"]["missing_count"], 0)

    def test_flags_a_row_whose_merge_query_was_not_reconstructed(self):
        dr = load_dr()
        settings = json.load(open(SETTINGS_PATH, encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            paths, rows = make_tree(tmp)
            rows["SparseRAG"][1]["merge_query"] = "tampered"
            write_jsonl(os.path.join(paths["dr"], "DecompositionReason_SparseRAG_k20.jsonl"),
                        rows["SparseRAG"])
            p = self._run(paths)
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            report = json.load(open(os.path.join(tmp, "validation_local.json"), encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertEqual(report["memories"]["SparseRAG"]["invalid_rows"], 1)
            self.assertIn("merge_query != answered current_state",
                          [e["error"] for e in report["memories"]["SparseRAG"]["invalid_examples"]])
            self.assertEqual(rows["SparseRAG"][1]["calls"], len(rows["SparseRAG"][1]["sub_questions"]) + 2)

    def test_flags_a_call_count_that_is_not_divide_solves_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, rows = make_tree(tmp)
            rows["DenseRAG"][0]["call_trace"] = rows["DenseRAG"][0]["call_trace"][:-1]
            write_jsonl(os.path.join(paths["dr"], "DecompositionReason_DenseRAG_k20.jsonl"),
                        rows["DenseRAG"])
            p = self._run(paths)
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            report = json.load(open(os.path.join(tmp, "validation_local.json"), encoding="utf-8"))
            errors = [e["error"] for e in report["memories"]["DenseRAG"]["invalid_examples"]]
            self.assertIn("call_trace length != calls", errors)
            self.assertIn("call_trace kinds are not Divide, Solve*n, Merge", errors)

    def test_reports_missing_and_extra_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, rows = make_tree(tmp)
            rows["SparseRAG"] = rows["SparseRAG"][:2]
            rows["SparseRAG"].append(dict(rows["SparseRAG"][0], rc_id="RC999999#02"))
            write_jsonl(os.path.join(paths["dr"], "DecompositionReason_SparseRAG_k20.jsonl"),
                        rows["SparseRAG"])
            p = self._run(paths)
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            m = json.load(open(os.path.join(tmp, "validation_local.json"), encoding="utf-8"))["memories"]["SparseRAG"]
            self.assertEqual(m["missing_count"], 1)
            self.assertEqual(m["extra_count"], 1)


@unittest.skipUnless(HAVE_NUMPY, "summarize_dr.py bootstrap needs numpy (use the repo venv)")
class SummarizeTests(unittest.TestCase):
    def _run(self, paths, out):
        return run_script("summarize_dr.py", [
            "--dr", paths["dr"], "--slice", paths["slice"], "--nr", paths["nr"], "--sr", paths["sr"],
            "--out", out])

    def test_writes_coverage_and_paired_tables_for_all_three_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = make_tree(tmp)
            out = os.path.join(tmp, "summary_dr.md")
            p = self._run(paths, out)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            payload = json.load(open(os.path.join(tmp, "summary_dr.json"), encoding="utf-8"))
            self.assertFalse(payload["complete"])              # 3 rows, not 306
            self.assertEqual(payload["meta"]["SparseRAG"]["rows"], 3)
            self.assertEqual(payload["meta"]["DenseRAG"]["missing"], 0)
            self.assertEqual(payload["leak_free_n"], 2)
            paired = payload["paired"]["DR SparseRAG"]
            # identical responses to the baselines: paired differences must be exactly zero, n shared keys
            for vs in ("vs NR overall", "vs SR overall"):
                self.assertEqual(paired[vs]["n"], 3)
                self.assertEqual(paired[vs]["dEM"], 0.0)
                self.assertEqual(paired[vs]["dEM_ci"], [0.0, 0.0])
            self.assertEqual(paired["vs NR 5-7"]["n"], 1)
            self.assertEqual(paired["vs NR leak_free"]["n"], 2)
            md = open(out, encoding="utf-8").read()
            self.assertIn("| DR SparseRAG | 3 |", md)
            self.assertIn("## Paired differences against the archived runs", md)
            self.assertIn("## Calls and wall time", md)

    def test_missing_dr_rows_shrink_the_paired_denominator_and_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, rows = make_tree(tmp)
            write_jsonl(os.path.join(paths["dr"], "DecompositionReason_DenseRAG_k20.jsonl"),
                        rows["DenseRAG"][:2])
            out = os.path.join(tmp, "summary_dr.md")
            p = self._run(paths, out)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            payload = json.load(open(os.path.join(tmp, "summary_dr.json"), encoding="utf-8"))
            meta = payload["meta"]["DenseRAG"]
            self.assertEqual(meta["rows"], 2)
            self.assertEqual(meta["missing"], 1)
            self.assertEqual(payload["paired"]["DR DenseRAG"]["vs NR overall"]["n"], 2)
            self.assertFalse(payload["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
