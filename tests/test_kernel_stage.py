"""GPU-free tests of the Kaggle kernel's staging/orchestration helpers.

Run with the rest: `python3 -m unittest discover -s tests -v`.

These tests never contact Kaggle, never import torch/vLLM and never run the model: they build a tiny
synthetic input dataset (same layout as the private Kaggle dataset), point the kernel module at it, and
check that staging verifies every sha256, reproduces the expected layout, rejects a tampered file,
selects the pre-registered smoke keys from the real slice, and always writes status/DONE on finish.
"""
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP = os.path.join(REPO, "experiments", "s6_kaggle_dr")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_kernel():
    spec = importlib.util.spec_from_file_location("s6_kernel_under_test", os.path.join(EXP, "kernel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_input(tmp):
    """Write a synthetic private-dataset directory with a valid manifest; return (input_dir, manifest)."""
    inp = os.path.join(tmp, "input")
    os.makedirs(inp)
    files = {}

    def put(name, content):
        path = os.path.join(inp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        files[name] = {"sha256": sha256(path), "bytes": os.path.getsize(path)}

    for u in range(12):
        put(f"question_answer_{u}.json",
            json.dumps({"message": {f"MG{i}": f"statement {i}" for i in range(25)},
                        "qa": {"2": {"RC000001#02": {"question": "q?", "answer": "a",
                                                     "message": ["MG001"], "fold_tag": 2}}}}))
    put("s306.json", json.dumps({"meta": {"n": 306}, "items": []}))
    for name, src in (("dr.py", os.path.join(EXP, "dr.py")),
                      ("lib.py", os.path.join(REPO, "experiments", "_shared", "lib.py")),
                      ("score.py", os.path.join(REPO, "experiments", "_shared", "score.py")),
                      ("settings.json", os.path.join(EXP, "settings.json"))):
        put(name, open(src, encoding="utf-8").read())
    for mem in ("SparseRAG", "DenseRAG"):
        put(f"baseline_nr_{mem}_k20.jsonl", "")
    manifest = {"code_commit": "test-commit", "files": files,
                "upstream": {"prompts_sha256": "0" * 64}}
    put("manifest.json", json.dumps(manifest))
    return inp, manifest


def point_at(km, tmp, inp):
    km.INPUT_ROOT = inp
    km.WORK = os.path.join(tmp, "work")
    km.ROOT = os.path.join(tmp, "root")
    km.SCRIPTS = os.path.join(km.ROOT, "scripts")
    km.CACHE = os.path.join(km.ROOT, "cache")
    km.QA_DIR = os.path.join(km.ROOT, "data", "mpr_dataset", "question_answer")
    km.SLICE = os.path.join(km.ROOT, "results", "slices", "s306.json")
    km.OUT = os.path.join(km.WORK, "results", "kaggle", "dr")
    km.RUNS = os.path.join(km.OUT, "runs")
    km.STATUS_PATH = os.path.join(km.WORK, "status.json")
    os.makedirs(km.WORK, exist_ok=True)


class KernelStagingTests(unittest.TestCase):
    def test_stage_inputs_verifies_hashes_and_layout(self):
        km = load_kernel()
        with tempfile.TemporaryDirectory() as tmp:
            inp, manifest = make_input(tmp)
            point_at(km, tmp, inp)
            got_manifest, verified = km.stage_inputs()
            self.assertEqual(got_manifest["code_commit"], "test-commit")
            self.assertTrue(all(v["matches_manifest"] for v in verified.values()), verified)
            for name in ("dr.py", "lib.py", "score.py", "settings.json"):
                self.assertTrue(os.path.isfile(os.path.join(km.SCRIPTS, name)), name)
            for u in range(12):
                self.assertTrue(os.path.isfile(os.path.join(km.QA_DIR, f"question_answer_{u}.json")))
            self.assertEqual(sha256(km.SLICE), manifest["files"]["s306.json"]["sha256"])
            self.assertTrue(os.path.isfile(os.path.join(km.ROOT, "baseline_nr_SparseRAG_k20.jsonl")))
            with open(km.STATUS_PATH) as f:
                status = json.load(f)
            self.assertEqual(status["stage"], "stage_inputs")

    def test_stage_inputs_discovers_nested_datasets_mount_layout(self):
        """The 2026-11-13 layout: /kaggle/input/datasets/<owner>/<slug>, not /kaggle/input/<slug>."""
        km = load_kernel()
        with tempfile.TemporaryDirectory() as tmp:
            inp, _manifest = make_input(tmp)
            base = os.path.join(tmp, "kaggle_input")
            nested = os.path.join(base, "datasets", "phinguynxun", "mpr-dr-inputs")
            os.makedirs(os.path.dirname(nested))
            os.rename(inp, nested)
            point_at(km, tmp, os.path.join(base, "mpr-dr-inputs"))  # the Pi-era assumption: absent
            km.INPUT_BASE = base
            got, verified = km.stage_inputs()
            self.assertEqual(got["code_commit"], "test-commit")
            self.assertTrue(all(v["matches_manifest"] for v in verified.values()))
            with open(os.path.join(km.WORK, "input_probe.json")) as f:
                probe = json.load(f)
            self.assertEqual(probe["resolved"], nested)
            self.assertEqual(probe["strategy"], "explicit")
            self.assertEqual(len(probe["files"]), 20)
            with open(km.STATUS_PATH) as f:
                status = json.load(f)
            self.assertEqual(status["input_root"], nested)

    def test_discover_input_root_failure_reports_candidates_and_tree(self):
        km = load_kernel()
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "kaggle_input")
            os.makedirs(os.path.join(base, "unrelated_dir"))
            km.INPUT_BASE = base
            km.INPUT_ROOT = os.path.join(tmp, "absent_mpr_dr_inputs")
            with self.assertRaises(FileNotFoundError) as cm:
                km.discover_input_root(timeout_s=0)
            msg = str(cm.exception)
            self.assertIn("manifest.json not found in any known input layout", msg)
            self.assertIn(os.path.join(tmp, "absent_mpr_dr_inputs"), msg)
            self.assertIn("unrelated_dir", msg)

    def test_stage_inputs_rejects_manifest_of_the_other_dataset(self):
        km = load_kernel()
        with tempfile.TemporaryDirectory() as tmp:
            inp, manifest = make_input(tmp)
            manifest["kaggle"] = {"dataset": "phinguynxun/mpr-mr-inputs"}
            with open(os.path.join(inp, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            point_at(km, tmp, inp)
            with self.assertRaises(RuntimeError) as cm:
                km.stage_inputs()
            self.assertIn("belongs to dataset", str(cm.exception))

    def test_stage_inputs_rejects_tampering(self):
        km = load_kernel()
        with tempfile.TemporaryDirectory() as tmp:
            inp, _manifest = make_input(tmp)
            point_at(km, tmp, inp)
            with open(os.path.join(inp, "question_answer_3.json"), "a") as f:
                f.write(" ")
            with self.assertRaises(RuntimeError) as cm:
                km.stage_inputs()
            self.assertIn("hash mismatch", str(cm.exception))

    def test_runtime_projection_uses_per_call_latency_and_worker_slots(self):
        """The gate must not divide the 612-row token total by the 12-row smoke wall clock."""
        km = load_kernel()
        rows = [{"calls": 7, "secs": 35.0, "prompt_tokens": 5000, "completion_tokens": 200}
                for _ in range(12)]
        est = km.project_full_run(rows, 42, remaining_secs=6000)
        self.assertEqual(est["projected_full_calls"], 7 * 612)
        self.assertAlmostEqual(est["estimated_full_secs"], round(7 * 612 * 5.0 / 42 * 1.3, 1), places=1)
        self.assertEqual(est["decision"], "start_full_run")
        # doubling the worker slots halves the projection; a tight budget refuses the full run
        self.assertAlmostEqual(km.project_full_run(rows, 84, 6000)["estimated_full_secs"],
                               round(est["estimated_full_secs"] / 2, 1), places=1)
        self.assertEqual(km.project_full_run(rows, 42, remaining_secs=100)["decision"],
                         "do_not_start_full_run")

    def test_sanity_gate_thresholds_follow_engine_parity(self):
        """A CUDA-12 vLLM pin must not fail the deployment gate purely on a token-level text difference."""
        km = load_kernel()
        gates = json.load(open(os.path.join(EXP, "settings.json")))["run"]["baseline_sanity"]["gates"]
        same = km.sanity_gates("0.32.0", gates)
        self.assertTrue(same["engine_matches_reference"])
        self.assertEqual(same["min_exact_response_matches_of_6"], 3)
        self.assertEqual(same["min_questions_with_retrieval_overlap_18_of_20"], 5)
        other = km.sanity_gates("0.11.2", gates)
        self.assertFalse(other["engine_matches_reference"])
        self.assertEqual(other["min_exact_response_matches_of_6"], 2)
        self.assertEqual(other["min_questions_with_retrieval_overlap_18_of_20"], 5)
        self.assertEqual(other["installed_vllm"], "0.11.2")

    def test_vllm_pins_follow_the_gpu_driver_cuda_generation(self):
        """vLLM >= 0.12 ships CUDA 13 wheels (needs driver >= 580); older drivers need a CUDA 12 pin."""
        km = load_kernel()
        self.assertEqual(km.vllm_specs(580)[0], "vllm==0.32.0")   # historical T4 version, CUDA 13 ok
        self.assertEqual(km.vllm_specs(590)[0], "vllm==0.32.0")
        for driver in (550, 570, None):
            specs = km.vllm_specs(driver)
            self.assertNotIn("vllm==0.32.0", specs, (driver, specs))
            self.assertEqual(specs[0], "vllm==0.11.2")
        self.assertTrue(all(s.startswith("vllm==") for s in km.vllm_specs(550)))

    def test_smoke_keys_cover_all_hop_groups_on_the_real_slice(self):
        km = load_kernel()
        km._settings = json.load(open(os.path.join(EXP, "settings.json"), encoding="utf-8"))
        sl = json.load(open(os.path.join(REPO, "results", "t4", "slices", "s306.json"), encoding="utf-8"))
        groups, flat = km.smoke_keys(sl["items"])
        self.assertEqual(len(flat), 6)
        self.assertEqual([g["hop"] for g in groups], [2, 5, 9])
        self.assertEqual(sorted(it["hop"] for it in flat), [2, 2, 5, 5, 9, 9])
        self.assertEqual(len({(it["uid"], it["rc_id"]) for it in flat}), 6)

    def test_finish_always_writes_status_and_done(self):
        km = load_kernel()
        with tempfile.TemporaryDirectory() as tmp:
            inp, manifest = make_input(tmp)
            point_at(km, tmp, inp)
            km._settings = json.load(open(os.path.join(EXP, "settings.json"), encoding="utf-8"))
            km.stage_finish(manifest, {}, validation=None, error="UnitTestError: boom")
            with open(km.STATUS_PATH) as f:
                status = json.load(f)
            self.assertEqual(status["status"], "failed")
            self.assertIn("boom", status["error"])
            self.assertTrue(os.path.isfile(os.path.join(km.WORK, "DONE")))
            self.assertTrue(os.path.isfile(os.path.join(km.WORK, "manifest_out.json")))
            with open(os.path.join(km.WORK, "manifest_out.json")) as f:
                out = json.load(f)
            self.assertEqual(out["code_commit"], "test-commit")
            self.assertEqual(out["kaggle"]["dataset"], "phinguynxun/mpr-dr-inputs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
