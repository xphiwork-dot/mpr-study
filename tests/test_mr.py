"""Local, stdlib-only tests for the MR experiment.

Run from the repo root:  python -m unittest tests.test_mr -v

The upstream-reference test needs the read-only pinned clone at
../mpr-study/upstream/MPR (or MPR_TEST_UPSTREAM); it is skipped if absent.
No GPU, model, network, numpy or vLLM is required.
"""

import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
UPSTREAM = os.environ.get("MPR_TEST_UPSTREAM", os.path.join(PARENT, "mpr-study", "upstream", "MPR"))
PROMPTS = os.path.join(UPSTREAM, "baselines", "prompts.py")
if os.path.exists(PROMPTS):
    os.environ.setdefault("MPR_PROMPTS", PROMPTS)

sys.path.insert(0, os.path.join(ROOT, "experiments", "s5_kaggle_mr"))
import mr  # noqa: E402
import lib  # noqa: E402


class FakeChat:
    def __init__(self, selections, middle_text=None):
        self.selections = list(selections)
        self.calls = []
        self.n = 0
        self.middle_text = middle_text

    def __call__(self, prompt, max_tokens, *, temperature, top_p=None, seed=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens,
                           "temperature": temperature, "top_p": top_p, "seed": seed})
        is_selection = prompt.startswith("SELECT") or "Here are multiple reasoning paths" in prompt
        is_middle = prompt.startswith("MIDDLE") or (
            "Previous Thoughts:" in prompt and "generate your thoughts for the current step" in prompt
        )
        if is_selection:
            text = self.selections.pop(0)
        elif is_middle:
            self.n += 1
            text = self.middle_text(self.n) if self.middle_text else f"T{self.n}"
        else:
            self.n += 1
            text = f"T{self.n}"
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 5,
                "finish_reason": "stop", "seconds": 0.001, "temperature": temperature,
                "top_p": top_p, "seed": seed}


def fake_retrieve(uid, query):
    return ["M0", "M1", "M2"], "REF::" + query


def fake_prompts():
    return types.SimpleNamespace(
        Structured_Start_RAG_Prompt_Template="START\n{references}\n{question}",
        Structured_Middle_RAG_Prompt_Template=(
            "MIDDLE\ngenerate your thoughts for the current step\n{references}\n"
            "Previous Thoughts:\n{previous_thoughts}\n{question}"),
        Structured_Final_RAG_Prompt_Template="FINAL\n{references}\nPrevious Thoughts:\n{previous_thoughts}\n{question}",
        Structured_Selection_RAG_Prompt_Template="SELECT\nReasoning Paths:\n{reasoning_paths}\n{question}",
        Naive_RAG_Prompt_Template="NAIVE\n{references}\n{question}",
    )


class UpstreamReferenceTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(PROMPTS):
            self.skipTest("pinned upstream clone not available")
        for p in (os.path.join(UPSTREAM, "reasonings"), os.path.join(UPSTREAM, "baselines")):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.P = lib.load_prompts()

    def test_exact_call_flow_prompts_and_retrieval_keys(self):
        from StructuredReason import StructuredReason

        selections = ["0", "2", "4", "6"]

        class UpstreamModel:
            def __init__(self):
                self.calls = []
                self.sels = list(selections)
                self.n = 0

            def response(self, call_type, param_dict, ref, knowledge_key):
                ids, refs = fake_retrieve(0, knowledge_key)
                params = dict(param_dict)
                params["references"] = refs
                prompt = getattr(self._P, f"{call_type}_RAG_Prompt_Template").format(**params)
                self.calls.append(("response", call_type, prompt, knowledge_key))
                self.n += 1
                return f"T{self.n}"

            def inference(self, call_type, param_dict):
                prompt = getattr(self._P, f"{call_type}_RAG_Prompt_Template").format(**param_dict)
                self.calls.append(("inference", call_type, prompt, None))
                return self.sels.pop(0)

        up = UpstreamModel()
        up._P = self.P
        StructuredReason({"max_step": 5, "branch_size": 2}, up).solve_qa("QUESTION")

        chat = FakeChat(selections)
        cfg = mr.MRConfig()
        runner = mr.MRRunner(cfg, self.P, chat, fake_retrieve, memory="sparse")
        result = runner.solve(uid=0, rc_id="RC1", question="QUESTION")

        ours = []
        for ev in result["trace"]["events"]:
            kind = "inference" if ev["kind"] == "Structured_Selection" else "response"
            ours.append((kind, ev["kind"], ev["prompt_sha256"], ev["retrieval_query"]))
        self.assertEqual(len(up.calls), 12)
        self.assertEqual(len(ours), 12)
        for (uk, uk_name, uprompt, ukey), (ok, ok_name, osha, okey) in zip(up.calls, ours):
            self.assertEqual(uk, ok)
            self.assertEqual(uk_name, ok_name)
            self.assertEqual(mr.sha256_text(uprompt), osha)
            self.assertEqual(ukey, okey)
        self.assertEqual(result["trace"]["stats"]["calls"], 12)
        self.assertEqual(result["trace"]["selected_path_depths"], [1, 2, 3, 4])
        self.assertEqual(result["trace"]["path_buffer_sizes"], [3, 5, 7])
        self.assertEqual(result["trace"]["final_path_depth"], 4)

    def test_upstream_selection_semantics_in_reference_source(self):
        # Selection is inference-only (no retrieval) and the buffer update retains
        # the selected path.  This is a source-level regression check with line refs.
        path = os.path.join(UPSTREAM, "reasonings", "StructuredReason.py")
        if not os.path.exists(path):
            self.skipTest("upstream source missing")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("self.knowledge_model.inference('Structured_Selection'", src)
        self.assertIn("path_buffer[:current_path_id] + path_buffer[current_path_id:] + new_path_list", src)
        self.assertIn("for current_step in range(1, max_step-1):", src)
        self.assertIn("for branch_id in range(self.config['branch_size']):", src)


class BufferSemanticsTest(unittest.TestCase):
    def test_retained_shallow_path_can_be_reselected(self):
        cfg = mr.MRConfig()
        chat = FakeChat(["0", "0", "0", "0"])
        runner = mr.MRRunner(cfg, fake_prompts(), chat, fake_retrieve, memory="sparse")
        tr = runner.solve(0, "RC", "Q")["trace"]
        self.assertEqual(tr["selected_path_depths"], [1, 1, 1, 1])
        self.assertEqual(tr["path_buffer_sizes"], [3, 5, 7])
        self.assertEqual(tr["final_path_depth"], 1)
        self.assertEqual(tr["expected_calls"], 12)

    def test_advancing_path_grows_one_step_per_round(self):
        cfg = mr.MRConfig()
        chat = FakeChat(["0", "2", "4", "6"])
        runner = mr.MRRunner(cfg, fake_prompts(), chat, fake_retrieve, memory="sparse")
        tr = runner.solve(0, "RC", "Q")["trace"]
        self.assertEqual(tr["selected_path_ids"], [0, 2, 4, 6])
        self.assertEqual(tr["selected_path_depths"], [1, 2, 3, 4])
        self.assertEqual(tr["final_path_depth"], 4)

    def test_max_step_and_branch_settings(self):
        self.assertEqual(mr.MRConfig(max_step=5, branch_size=2).expected_calls, 12)
        self.assertEqual(mr.MRConfig(max_step=3, branch_size=2).expected_calls, 6)
        self.assertEqual(mr.MRConfig(max_step=5, branch_size=3).expected_calls, 15)


class ParsingTest(unittest.TestCase):
    def test_safe_selection_parser(self):
        cases = {
            "2": (2, "ok"),
            " 2\n": (2, "ok"),
            "02": (2, "ok"),
            "7": (0, "fallback_out_of_range"),
            "-1": (0, "fallback_non_digit"),
            "2.0": (0, "fallback_non_digit"),
            "": (0, "fallback_non_digit"),
            "Path 2": (0, "fallback_non_digit"),
            "\u0662": (0, "fallback_non_digit"),  # upstream isdigit/eval would accept this
        }
        for text, expected in cases.items():
            self.assertEqual(mr.parse_selection(text, 3), expected, repr(text))

    def test_selection_fallback_is_recorded(self):
        cfg = mr.MRConfig()
        chat = FakeChat(["not a number", "0", "0", "0"])
        runner = mr.MRRunner(cfg, fake_prompts(), chat, fake_retrieve, memory="sparse")
        tr = runner.solve(0, "RC", "Q")["trace"]
        self.assertEqual(tr["selections"][0]["selected"], 0)
        self.assertEqual(tr["selections"][0]["status"], "fallback_non_digit")
        self.assertEqual(tr["stats"]["selection_fallbacks"], 1)


class SeedTest(unittest.TestCase):
    def test_stable_and_distinct(self):
        a = mr.stable_seed(0, 7, "RC1", 2, 0)
        b = mr.stable_seed(0, 7, "RC1", 2, 0)
        c = mr.stable_seed(0, 7, "RC1", 2, 1)
        d = mr.stable_seed(0, 7, "RC1", 3, 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertTrue(0 <= a <= 0x7FFFFFFF)

    def test_only_middle_calls_are_sampled(self):
        cfg = mr.MRConfig()
        chat = FakeChat(["0", "2", "4", "6"])
        runner = mr.MRRunner(cfg, fake_prompts(), chat, fake_retrieve, memory="sparse")
        tr = runner.solve(0, "RC4", "Q")["trace"]
        for ev in tr["events"]:
            if ev["kind"] == "Structured_Middle":
                self.assertEqual(ev["temperature"], 0.7)
                self.assertEqual(ev["top_p"], 1.0)
                self.assertIsNotNone(ev["seed"])
            else:
                self.assertEqual(ev["temperature"], 0.0)
                self.assertIsNone(ev["top_p"])
                self.assertIsNone(ev["seed"])


class DuplicateAndLeakTest(unittest.TestCase):
    def test_duplicate_candidates_are_measured(self):
        cfg = mr.MRConfig()
        chat = FakeChat(["0", "2", "4", "6"], middle_text=lambda n: "same")
        runner = mr.MRRunner(cfg, fake_prompts(), chat, fake_retrieve, memory="sparse")
        tr = runner.solve(0, "RC", "Q")["trace"]
        self.assertEqual(tr["candidate_pairs_total"], 3)
        self.assertEqual(tr["duplicate_candidate_pairs"], 3)

        chat2 = FakeChat(["0", "2", "4", "6"], middle_text=lambda n: f"unique-{n}")
        runner2 = mr.MRRunner(cfg, fake_prompts(), chat2, fake_retrieve, memory="sparse")
        tr2 = runner2.solve(0, "RC", "Q")["trace"]
        self.assertEqual(tr2["duplicate_candidate_pairs"], 0)

    def test_gold_answer_never_enters_prompts(self):
        cfg = mr.MRConfig()
        chat = FakeChat(["0", "2", "4", "6"])
        runner = mr.MRRunner(cfg, fake_prompts(), chat, fake_retrieve, memory="sparse")
        runner.solve(0, "RC", "QUESTION_WITHOUT_GOLD")
        for c in chat.calls:
            self.assertNotIn("SECRET_GOLD_42", c["prompt"])


class ResumeAndCoverageTest(unittest.TestCase):
    def test_cut_tail_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rows.jsonl")
            mr._append_jsonl(path, {"uid": 0, "rc_id": "A", "response": "x", "calls": 12})
            mr._append_jsonl(path, {"uid": 0, "rc_id": "A", "response": "y", "calls": 12})
            mr._append_jsonl(path, {"uid": 1, "rc_id": "B", "response": "", "calls": 12})
            mr._append_jsonl(path, {"uid": 3, "rc_id": "D", "response": "d", "calls": 12})
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"uid": 2, "rc_id": "C", "response": "cut')
            rows = mr.load_valid_rows(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows["0:A"]["response"], "y")

    def test_smoke_items_span_all_hop_groups(self):
        items = lib.load_slice(os.path.join(ROOT, "results", "t4", "slices", "s306.json"))["items"]
        smoke = mr.smoke_items(items)
        groups = {"2-4": range(2, 5), "5-7": range(5, 8), "8-10": range(8, 11)}
        for rng in groups.values():
            self.assertTrue(any(int(i["hop"]) in rng for i in smoke))
        self.assertEqual(len({i["uid"] for i in smoke}), 3)


class CommandIntegrationTest(unittest.TestCase):
    def test_cmd_run_resume_and_coverage(self):
        items = [{"uid": 0, "hop": 2, "rc_id": "RC1"},
                 {"uid": 0, "hop": 3, "rc_id": "RC2"}]
        data = {"qa": {"2": {"RC1": {"question": "Q1", "answer": "A1"}},
                        "3": {"RC2": {"question": "Q2", "answer": "A2"}}}}

        class Store:
            def __init__(self, *a, **k):
                pass

            def user(self, uid):
                return data, types.SimpleNamespace(messages={})

            def refs(self, uid, query):
                return ["m1"], "refs"

        chat = FakeChat(["0", "2", "4", "6"] * 10)

        def fake_vllm(prompt, max_tokens, **kwargs):
            return chat(prompt, max_tokens, temperature=kwargs.get("temperature"),
                        top_p=kwargs.get("top_p"), seed=kwargs.get("seed"))

        old = (mr.MemoryStore, mr.vllm_chat, lib.load_prompts, lib.load_slice, lib.load_user)
        mr.MemoryStore = Store
        mr.vllm_chat = fake_vllm
        lib.load_prompts = fake_prompts
        lib.load_slice = lambda path: {"items": items}
        lib.load_user = lambda uid: data
        try:
            with tempfile.TemporaryDirectory() as d:
                args = types.SimpleNamespace(
                    stage="full", slice="unused", data_dir="unused", cache_dir="unused",
                    output_dir=d, memories="sparse,dense", topk=20, max_step=5,
                    branch_size=2, expected_calls=12, seed=0, workers=2, max_attempts=2,
                    limit=0, deadline=0, vllm_url="unused", model="qwen", device=None,
                    request_timeout=10)
                self.assertEqual(mr.cmd_run(args), 0)
                for label in ("SparseRAG", "DenseRAG"):
                    path = os.path.join(d, f"StructuredReason_{label}_k20_s5.jsonl")
                    rows = mr.load_valid_rows(path)
                    self.assertEqual(len(rows), 2)
                    self.assertEqual({r["calls"] for r in rows.values()}, {12})
                # second run is a no-op (resume)
                before = os.path.getsize(os.path.join(d, "StructuredReason_SparseRAG_k20_s5.jsonl"))
                self.assertEqual(mr.cmd_run(args), 0)
                after = os.path.getsize(os.path.join(d, "StructuredReason_SparseRAG_k20_s5.jsonl"))
                self.assertEqual(before, after)
                cov = mr.coverage_report(d, items, "unused", ["sparse", "dense"])
                self.assertTrue(all(v["complete"] for v in cov["memories"].values()))
        finally:
            (mr.MemoryStore, mr.vllm_chat, lib.load_prompts, lib.load_slice, lib.load_user) = old


class SharedGridRegressionTest(unittest.TestCase):
    """The reused NR/SR runner still makes one / max_step calls with same prompts."""

    def test_naive_and_sequential_call_counts(self):
        sys.path.insert(0, os.path.join(ROOT, "experiments", "_shared"))
        import grid

        prompts = fake_prompts()
        prompts.Sequential_Start_RAG_Prompt_Template = "SSTART {max_step} {references} {question}"
        prompts.Sequential_Middle_RAG_Prompt_Template = (
            "SMID {current_step} {max_step} {references} {previous_thoughts} {question}")
        prompts.Sequential_Final_RAG_Prompt_Template = (
            "SFINAL {references} {previous_thoughts} {question}")
        data = {"qa": {"2": {"RC1": {"question": "Q", "answer": "A"}}}}
        index = types.SimpleNamespace()

        def fake_chat(prompt, max_tokens, *args, **kwargs):
            return "x", 1, 1, 0.0

        old = lib.chat
        lib.chat = fake_chat
        try:
            for reason, steps, expected in (("naive", 5, 1), ("seq", 5, 5), ("seq", 3, 3)):
                args = types.SimpleNamespace(reason=reason, max_step=steps, topk=20)
                runner = grid.Runner(args, prompts)
                runner.user = lambda uid, need_dense: (data, index)
                runner.refs = lambda mem, ix, query, qa: (["m1"], "refs")
                row = runner.solve("sparse", {"uid": 0, "rc_id": "RC1", "hop": 2})
                self.assertEqual(row["calls"], expected, (reason, steps))
        finally:
            lib.chat = old


if __name__ == "__main__":
    unittest.main()

class VLLMClientTest(unittest.TestCase):
    def test_request_body_and_http_error(self):
        import http.server
        import threading

        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                seen.append(body)
                if body.get("model") == "bad":
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"bad request"}')
                    return
                payload = json.dumps({
                    "choices": [{"message": {"content": " answer "}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            url = f"http://127.0.0.1:{srv.server_port}"
            out = mr.vllm_chat("hello", 7, temperature=0.7, top_p=1.0, seed=123,
                               url=url, model="qwen", retries=1)
            self.assertEqual(out["text"], " answer ")
            self.assertEqual(seen[-1]["seed"], 123)
            self.assertEqual(seen[-1]["top_p"], 1.0)
            self.assertEqual(seen[-1]["max_tokens"], 7)
            with self.assertRaises(mr.MRFailure):
                mr.vllm_chat("hello", 7, temperature=0.0, url=url, model="bad", retries=1)
        finally:
            srv.shutdown()
            srv.server_close()


class KernelMountTest(unittest.TestCase):
    """The Kaggle dataset mount layout has changed; the kernel must find it either way."""

    def setUp(self):
        import importlib.util

        path = os.path.join(ROOT, "experiments", "s5_kaggle_mr", "kernel.py")
        spec = importlib.util.spec_from_file_location("mr_kaggle_kernel", path)
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        td = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, td, True)
        self.tmp = td
        self.kernel.LOGS = os.path.join(td, "logs")
        self.kernel.LOG_PATH = os.path.join(td, "logs", "kernel.log")
        os.makedirs(self.kernel.LOGS, exist_ok=True)
        self._saved_root = os.environ.pop("MPR_INPUT_ROOT", None)
        if self._saved_root is not None:
            self.addCleanup(os.environ.__setitem__, "MPR_INPUT_ROOT", self._saved_root)

    def _layout(self, *parts):
        root = os.path.join(self.tmp, *parts)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "s306.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        return root

    def test_both_known_mount_layouts_resolve(self):
        legacy = self._layout("input", "mpr-mr-inputs")
        nested = self._layout("input", "datasets", "phinguynxun", "mpr-mr-inputs")
        for root in (legacy, nested):
            self.kernel.INPUT_CANDIDATES = [root]
            got, slice_path = self.kernel.find_input(wait_s=0)
            self.assertEqual(got, os.path.abspath(root))
            self.assertEqual(slice_path, os.path.join(os.path.abspath(root), "s306.json"))

    def test_bounded_walk_fallback_and_clear_failure(self):
        actual = self._layout("input", "datasets", "phinguynxun", "mpr-mr-inputs")
        self.kernel.INPUT_CANDIDATES = [os.path.join(self.tmp, "does-not-exist")]
        real_walk = os.walk

        def found_walk(top):
            yield os.path.join(self.tmp, "input"), ["datasets"], []
            yield os.path.join(self.tmp, "input", "datasets"), ["phinguynxun"], []
            yield os.path.dirname(actual), ["mpr-mr-inputs"], []
            yield actual, [], ["s306.json"]

        def empty_walk(top):
            yield os.path.join(self.tmp, "input"), ["datasets"], []
            yield os.path.join(self.tmp, "input", "datasets"), [], []

        try:
            os.walk = found_walk
            got, _ = self.kernel.find_input(wait_s=0)
            self.assertEqual(got, os.path.abspath(actual))
            os.walk = empty_walk
            with self.assertRaisesRegex(RuntimeError, "does not contain s306.json"):
                self.kernel.find_input(wait_s=0)
        finally:
            os.walk = real_walk
