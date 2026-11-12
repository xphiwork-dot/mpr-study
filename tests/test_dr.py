"""Local, GPU-free tests for the S6 DecompositionReason (DR) adaptation.

Run from the repo root:
    python3 -m unittest discover -s tests -v

The parity tests import the pinned upstream clone at upstream/MPR (gitignored, commit a24953c) and
execute its real DecompositionReason class against stub knowledge models that use the same fake memory
index as our runner. That pins the call order, the exact prompt strings, the retrieval keys and the
final answer text. Everything else is stdlib: no vLLM, no torch, no network.
"""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP = os.path.join(REPO, "experiments", "s6_kaggle_dr")
sys.path.insert(0, EXP)

import dr  # noqa: E402
import lib  # noqa: E402

UPSTREAM = os.environ.get("MPR_UPSTREAM", os.path.join(REPO, "upstream", "MPR"))
lib.UPSTREAM_PROMPTS = os.path.join(UPSTREAM, "baselines", "prompts.py")  # tests never read /content
UPSTREAM_COMMIT = "a24953c8f342c9f3b2064e562ead49d477d49138"
SETTINGS = dr.load_settings()
QUESTION = "What is the attachment count of the Lunar Initiative Memo after subtracting 1?"
QUESTION2 = "How many pages does the book have after subtracting 195?"
DIVISION = "What is the attachment count of the Lunar Initiative Memo?\n" "What number results from subtracting 1 from it?"


def make_data(question=QUESTION, answer="ZQX-1847", gold=("MG001", "MG002")):
    return {
        "message": {f"MG{i:03d}": f"statement {i} about the memo" for i in range(40)},
        "qa": {"2": {"RC000001#02": {"question": question, "answer": answer,
                                     "message": list(gold), "fold_tag": 2}},
               "5": {"RC000002#05": {"question": QUESTION2, "answer": "125",
                                     "message": ["MG003"], "fold_tag": 5}}},
    }


class FakeIndex:
    """Deterministic memory index: top-k = first k ids; records every retrieval query."""

    def __init__(self, data, k=20):
        self.messages = data["message"]
        self.ids = list(self.messages)
        self.k = k
        self.emb = None
        self.queries = []

    def sparse(self, query, k):
        self.queries.append(query)
        return self.ids[:k]

    def dense(self, query, k):
        self.queries.append(query)
        return self.ids[:k]

    def add_dense(self, encoder):
        self.emb = encoder


class FakeChat:
    """Consumes canned outputs in order, records prompts/temperatures/token caps."""

    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.prompts = []
        self.temps = []
        self.caps = []

    def __call__(self, prompt, max_tokens, retries=5, temperature=0):
        self.prompts.append(prompt)
        self.temps.append(temperature)
        self.caps.append(max_tokens)
        text = self.outputs.pop(0) if self.outputs else ""
        return text.strip(), 10, 5, 0.001


def scripted_chat(division=DIVISION):
    """Stateless chat: answers differ by prompt kind, so reruns and concurrency are deterministic."""

    def chat(prompt, max_tokens, retries=5, temperature=0):
        if "break it down into several sub-questions" in prompt:
            text = division
        elif "have decomposed them into several sub-questions and answered them separately" in prompt:
            text = "final answer"
        else:
            text = "sub answer"
        return text.strip(), 100, 10, 0.002

    return chat


def data_for_items(items):
    """make_data() plus a unique, well-formed qa row for every (uid, hop, rc_id) in items."""
    data = make_data()
    for it in items:
        h, rc = str(it["hop"]), it["rc_id"]
        data["qa"].setdefault(h, {})
        data["qa"][h].setdefault(rc, {"question": f"question for {rc}?", "answer": "17",
                                      "message": ["MG003"], "fold_tag": int(it["hop"])})
    return data


def make_runner(chat, data=None, mem="sparse", prompts=None):
    data = data or make_data()
    index = FakeIndex(data)
    runner = dr.Runner(
        SETTINGS,
        prompts=prompts,
        chat=chat,
        user_loader=lambda uid: data,
        index_factory=lambda uid, d, need_dense, enc: index,
    )
    return runner, index


def item(uid=0, rc_id="RC000001#02", hop=2):
    return {"uid": uid, "rc_id": rc_id, "hop": hop}


# ---------- upstream import (guarded) ----------

def _stub(name, **attrs):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def load_upstream():
    if not os.path.isdir(UPSTREAM):
        return None
    _stub("torch")
    _stub("requests")
    _stub("openai", OpenAI=object)
    _stub("peft", PeftModel=object)
    _stub("transformers", AutoModelForCausalLM=object, AutoTokenizer=object)
    for sub in ("baselines", "reasonings", ""):
        p = os.path.join(UPSTREAM, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import BaseMethod  # noqa: PLC0415  (upstream file)
    import Ignoramus  # noqa: PLC0415  (upstream file)
    spec = importlib.util.spec_from_file_location(
        "upstream_decomposition_reason", os.path.join(UPSTREAM, "reasonings", "DecompositionReason.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"RAGMethod": BaseMethod.RAGMethod, "Ignoramus": Ignoramus.Ignoramus,
            "DecompositionReason": mod.DecompositionReason}


UP = load_upstream()


class StubLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def fast_run(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0) if self.outputs else ""


class StubRetriever:
    def __init__(self, index):
        self.index = index
        self.queries = []

    def retrieve(self, key):
        self.queries.append(key)
        ids = self.index.sparse(key, 20)
        return "\n".join(self.index.messages[i] for i in ids)


class ParserTests(unittest.TestCase):
    def test_upstream_rule_with_blank_lines(self):
        text = "one?\n\ntwo?\n"
        subs, meta = dr.parse_division(text, 5, QUESTION, policy="upstream_strict")
        self.assertEqual(subs, ["one?", "", "two?", ""])  # upstream keeps blank lines and the tail
        self.assertEqual(meta["raw_lines"], 4)

    def test_primary_policy_drops_blank_lines_only(self):
        subs, meta = dr.parse_division("1. one?\n\n2. two?\n\n\n", 5, QUESTION)
        self.assertEqual(subs, ["1. one?", "2. two?"])  # numbering kept verbatim
        self.assertEqual(meta["dropped_blank"], 4)
        self.assertFalse(meta["fallback"])
        self.assertEqual(meta["policy"], "ref_faithful_drop_blank")

    def test_first_n_and_over_limit(self):
        text = "\n".join(f"q{i}?" for i in range(8))
        subs, meta = dr.parse_division(text, 5, QUESTION)
        self.assertEqual(subs, [f"q{i}?" for i in range(5)])
        self.assertEqual(meta["over_limit"], 3)

    def test_empty_division_falls_back_to_the_question(self):
        for text in ("", "   ", "\n\n \n"):
            subs, meta = dr.parse_division(text, 5, QUESTION)
            self.assertEqual(subs, [QUESTION])
            self.assertTrue(meta["fallback"])

    def test_single_line_and_crlf(self):
        subs, _ = dr.parse_division("only one?\r\n", 5, QUESTION)
        self.assertEqual(subs, ["only one?"])

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            dr.parse_division("q?", 5, QUESTION, policy="repair-everything")


class StateFormattingTests(unittest.TestCase):
    def test_current_state_exact_lines(self):
        subs, answers = ["s1", "s2", "s3"], ["a1"]
        self.assertEqual(dr.build_current_state(subs, answers, 1),
                         "1. s1 (Answer: a1)\n2. s2 (Current Sub-question)\n3. s3")
        self.assertEqual(dr.build_current_state(subs, [], 0),
                         "1. s1 (Current Sub-question)\n2. s2\n3. s3")
        self.assertEqual(dr.build_answered_state(subs, ["a1", "a2", "a3"]),
                         "1. s1 (Answer: a1)\n2. s2 (Answer: a2)\n3. s3 (Answer: a3)")

    def test_retrieval_keys_exact(self):
        self.assertEqual(dr.solve_retrieval_key("Q?", "S?"), "(question) Q?\n (sub-question) S?")
        self.assertEqual(dr.merge_retrieval_key("state"), "state")


class FlowTests(unittest.TestCase):
    def test_calls_states_and_keys(self):
        answers = ["ans one", "ans two"]
        chat = FakeChat([DIVISION, answers[0], answers[1], "final"])
        runner, index = make_runner(chat)
        row = runner.solve("sparse", item())

        self.assertEqual(row["response"], "final")
        self.assertEqual(row["calls"], 4)  # 1 divide + 2 solve + 1 merge
        self.assertEqual([c["kind"] for c in row["call_trace"]],
                         ["Decomposition_Divide", "Decomposition_Solve", "Decomposition_Solve",
                          "Decomposition_Merge"])
        self.assertEqual([c["max_tokens"] for c in row["call_trace"]], [200, 200, 200, 64])
        self.assertTrue(all(t == 0.0 for t in chat.temps))
        self.assertEqual(row["sub_questions"], DIVISION.split("\n"))
        self.assertEqual(row["sub_answers"], answers)
        self.assertEqual(row["current_states"],
                         ["1. %s (Current Sub-question)\n2. %s" % tuple(DIVISION.split("\n")),
                          "1. %s (Answer: ans one)\n2. %s (Current Sub-question)" % tuple(DIVISION.split("\n"))])
        self.assertEqual(row["solve_queries"],
                         ["(question) %s\n (sub-question) %s" % (QUESTION, DIVISION.split("\n")[0]),
                          "(question) %s\n (sub-question) %s" % (QUESTION, DIVISION.split("\n")[1])])
        self.assertEqual(row["merge_query"],
                         "1. %s (Answer: ans one)\n2. %s (Answer: ans two)" % tuple(DIVISION.split("\n")))
        # one retrieval for the merge + one per solve, in that order, each with the exact key
        self.assertEqual(index.queries, row["solve_queries"] + [row["merge_query"]])
        self.assertEqual(len(row["solve_retrieved"]), 2)
        self.assertEqual(len(row["merge_retrieved"]), 20)
        self.assertEqual(row["ground_truth"], "ZQX-1847")
        self.assertEqual(row["settings_sha256"], dr.config_hash(SETTINGS))

    def test_divide_prompt_has_no_references_and_caps_five(self):
        chat = FakeChat([DIVISION, "a1", "a2", "final"])
        runner, _ = make_runner(chat)
        runner.solve("sparse", item())
        divide = chat.prompts[0]
        self.assertIn("break it down into several sub-questions", divide)
        self.assertIn("1 to 5 sub-questions", divide)
        self.assertNotIn("Information:", divide)
        # solve prompts do carry the retrieved statements and the numbered state
        self.assertIn("Information:", chat.prompts[1])
        self.assertIn("statement 0 about the memo", chat.prompts[1])
        self.assertIn("1. %s (Current Sub-question)" % DIVISION.split("\n")[0], chat.prompts[1])

    def test_seven_calls_with_five_sub_questions(self):
        division = "\n".join(f"sub-question {i}?" for i in range(5))
        chat = FakeChat([division] + [f"a{i}" for i in range(5)] + ["final"])
        runner, _ = make_runner(chat)
        row = runner.solve("sparse", item())
        self.assertEqual(row["calls"], 7)
        self.assertEqual(len(row["call_trace"]), 7)
        self.assertEqual(len(row["solve_queries"]), 5)

    def test_empty_division_runs_three_calls_not_naive(self):
        chat = FakeChat(["", "only answer", "final"])
        runner, _ = make_runner(chat)
        row = runner.solve("sparse", item())
        self.assertTrue(row["parser"]["fallback"])
        self.assertEqual(row["sub_questions"], [QUESTION])
        self.assertEqual(row["calls"], 3)
        self.assertNotIn("Information:", chat.prompts[0])  # divide still has no retrieval

    def test_numbering_is_not_stripped(self):
        division = "1. first sub-question?\n2. second sub-question?"
        chat = FakeChat([division, "a1", "a2", "final"])
        runner, _ = make_runner(chat)
        row = runner.solve("sparse", item())
        self.assertEqual(row["sub_questions"], ["1. first sub-question?", "2. second sub-question?"])
        self.assertIn("(sub-question) 1. first sub-question?", row["solve_queries"][0])

    def test_max_tokens_and_caps_from_settings(self):
        self.assertEqual(SETTINGS["decoding"]["max_tokens"], {"divide": 200, "solve": 200, "merge": 64})
        self.assertEqual(SETTINGS["method"]["decompose_num"], 5)
        self.assertEqual(SETTINGS["retrieval"]["topk"], 20)
        self.assertEqual(SETTINGS["run"]["expected_rows_total"], 612)

    def test_dense_uses_encoder_and_dense_index(self):
        chat = FakeChat([DIVISION, "a1", "a2", "final"])
        data = make_data()
        index = FakeIndex(data)
        index.emb = None
        calls = []

        def fake_encoder(*a, **k):
            calls.append("encoder")
            return "ENC"

        old = lib.get_encoder
        lib.get_encoder = fake_encoder
        try:
            runner = dr.Runner(SETTINGS, chat=chat, user_loader=lambda uid: data,
                               index_factory=lambda uid, d, need_dense, enc: index)
            row = runner.solve("dense", item())
        finally:
            lib.get_encoder = old
        self.assertEqual(calls, ["encoder"])
        self.assertEqual(len(index.queries), 3)
        self.assertEqual(row["memory"], "DenseRAG")

    def test_ignoramus_never_retrieves(self):
        chat = FakeChat([DIVISION, "a1", "a2", "final"])
        runner, index = make_runner(chat, mem="ignoramus")
        row = runner.solve("ignoramus", item())
        self.assertEqual(index.queries, [])
        self.assertEqual(row["calls"], 4)
        self.assertIsNone(row["merge_retrieved"])
        self.assertEqual(row["memory"], "Ignoramus")
        self.assertNotIn("Information:", "".join(chat.prompts))
        self.assertIn("Please help me generate the final answer to the question based on the sub-questions.",
                      chat.prompts[-1])

    def test_gold_answer_and_ids_never_enter_prompts(self):
        base = make_data()
        changed = copy.deepcopy(base)
        changed["qa"]["2"]["RC000001#02"]["answer"] = "OTHER-GOLD-9999"
        changed["qa"]["2"]["RC000001#02"]["message"] = ["MG039"]
        prompts = []
        for data in (base, changed):
            chat = FakeChat([DIVISION, "a1", "a2", "final"])
            runner = dr.Runner(SETTINGS, chat=chat, user_loader=lambda uid, d=data: d,
                               index_factory=lambda uid, d, need_dense, enc: FakeIndex(d))
            runner.solve("sparse", item())
            prompts.append(chat.prompts)
        self.assertEqual(prompts[0], prompts[1])  # neither the gold answer nor the gold ids are read
        joined = "\n".join(prompts[0])
        self.assertNotIn("ZQX-1847", joined)
        self.assertNotIn("MG001", joined)


@unittest.skipUnless(UP is not None, "pinned upstream clone missing (gitignored upstream/MPR)")
class UpstreamParityTests(unittest.TestCase):
    """Execute upstream DecompositionReason itself with the same fake memory and canned outputs."""

    def _run_upstream(self, knowledge_cls, outputs, question):
        llm = StubLLM(outputs)
        knowledge = knowledge_cls(llm)
        reason = UP["DecompositionReason"]({"decompose_num": 5}, knowledge)
        final = reason.solve_qa(question)
        return knowledge, llm, final

    def test_rag_flow_prompts_keys_and_answer_match_upstream_exactly(self):
        answers = ["answer one", "answer two"]
        outputs = [DIVISION] + answers + ["merged final"]
        data = make_data()
        upstream_index = FakeIndex(data)

        class StubRAG(UP["RAGMethod"]):
            def __init__(self, llm):
                self.llm = llm
                self.topk = 20
                self.messages = data["message"]

            def __retrieve_message__(self, question, ref):
                return retr.retrieve(question)

        retr = StubRetriever(upstream_index)
        knowledge, llm, final_up = self._run_upstream(StubRAG, outputs, QUESTION)

        chat = FakeChat(outputs)
        runner, index = make_runner(chat)
        row = runner.solve("sparse", item())

        self.assertEqual(row["response"], final_up)
        self.assertEqual(chat.prompts, llm.prompts)           # exact prompt strings, all 4 calls
        self.assertEqual(index.queries, retr.queries)          # exact retrieval keys, order preserved
        self.assertEqual(row["calls"], len(llm.prompts))

    def test_rag_flow_with_five_sub_questions_matches_upstream(self):
        division = "\n".join(f"sub-question {i}?" for i in range(5))
        outputs = [division] + [f"answer {i}" for i in range(5)] + ["merged final"]
        data = make_data()
        index = FakeIndex(data)
        retr = StubRetriever(index)

        class StubRAG(UP["RAGMethod"]):
            def __init__(self, llm):
                self.llm = llm
                self.topk = 20
                self.messages = data["message"]

            def __retrieve_message__(self, question, ref):
                return retr.retrieve(question)

        _, llm, final_up = self._run_upstream(StubRAG, outputs, QUESTION)
        chat = FakeChat(outputs)
        runner, ours = make_runner(chat)
        row = runner.solve("sparse", item())
        self.assertEqual(chat.prompts, llm.prompts)
        self.assertEqual(ours.queries, retr.queries)
        self.assertEqual(row["response"], final_up)
        self.assertEqual(row["calls"], 7)

    def test_ignoramus_flow_matches_upstream(self):
        outputs = [DIVISION, "answer one", "answer two", "merged final"]

        class StubIgn(UP["Ignoramus"]):
            def __init__(self, llm):
                self.llm = llm

        _, llm, final_up = self._run_upstream(StubIgn, outputs, QUESTION)
        chat = FakeChat(outputs)
        runner, index = make_runner(chat, mem="ignoramus")
        row = runner.solve("ignoramus", item())
        self.assertEqual(chat.prompts, llm.prompts)
        self.assertEqual(index.queries, [])  # upstream Ignoramus.response never retrieves
        self.assertEqual(row["response"], final_up)

    def test_upstream_commit_is_pinned(self):
        head = os.popen(f"git -C {UPSTREAM} rev-parse HEAD").read().strip()
        self.assertEqual(head, UPSTREAM_COMMIT)


    def test_naive_flow_matches_upstream(self):
        data = make_data()

        class StubRAG(UP["RAGMethod"]):
            def __init__(self, llm):
                self.llm = llm
                self.topk = 20
                self.messages = data["message"]

            def __retrieve_message__(self, question, ref):
                return retr.retrieve(question)

        retr = StubRetriever(FakeIndex(data))
        llm = StubLLM(["one-call answer"])
        knowledge = StubRAG(llm)
        spec = importlib.util.spec_from_file_location(
            "upstream_naive_reason", os.path.join(UPSTREAM, "reasonings", "NaiveReason.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        final_up = mod.NaiveReason({"reason_method": "NaiveReason"}, knowledge).solve_qa(QUESTION)

        chat = FakeChat(["one-call answer"])
        runner, ours = make_runner(chat)
        row = runner.solve_naive("sparse", item())
        self.assertEqual(row["response"], final_up)
        self.assertEqual(chat.prompts, llm.prompts)
        self.assertEqual(ours.queries, retr.queries)
        self.assertEqual(ours.queries, [QUESTION])  # NaiveReason.py:8 keys retrieval on the question
        self.assertEqual(row["calls"], 1)
        self.assertEqual(chat.caps, [64])


class RunTests(unittest.TestCase):
    def _run(self, items, out_dir, chat=None, workers=2, deadline=None, **kw):
        data = data_for_items(items)
        runner = dr.Runner(SETTINGS, chat=chat or scripted_chat(), user_loader=lambda uid: data,
                           index_factory=lambda uid, d, need_dense, enc: FakeIndex(d))
        return runner.run(items, ["sparse"], out_dir, workers=workers, deadline=deadline, **kw)

    def test_resume_skips_finished_and_survives_a_cut_tail(self):
        items = [item(), item(rc_id="RC000002#05", hop=5)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(items, tmp)
            self.assertEqual(summary["memories"]["DecompositionReason_SparseRAG_k20"]["rows"], 2)
            path = os.path.join(tmp, "DecompositionReason_SparseRAG_k20.jsonl")
            with open(path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual({dr.lib.qkey(r) for r in rows}, {dr.lib.qkey(i) for i in items})
            # a jsonl line cut in half by a dying kernel is ignored and that question is redone
            with open(path, "a") as f:
                f.write('{"uid": 0, "rc_id"')
            summary2 = self._run(items, tmp)
            ms = summary2["memories"]["DecompositionReason_SparseRAG_k20"]
            self.assertEqual(ms["todo"], 0)  # both keys are already complete
            with open(path) as f:
                self.assertTrue(f.read().endswith('{"uid": 0, "rc_id"'))

    def test_only_keys_restricts_the_slice_subset(self):
        slice_path = os.path.join(REPO, "results", "t4", "slices", "s306.json")
        sl = json.load(open(slice_path, encoding="utf-8"))
        chosen = sl["items"][:3]
        keys = [f"{i['uid']}:{i['rc_id']}" for i in chosen]
        data = make_data()
        for it in chosen:
            data["qa"].setdefault(str(it["hop"]), {}).setdefault(
                it["rc_id"], {"question": f"question {it['rc_id']}?", "answer": "17",
                              "message": ["MG003"], "fold_tag": it["hop"]})
        old_slice, old_user = dr.lib.load_slice, dr.lib.load_user
        old_factory, old_enc, old_chat = dr.default_index_factory, lib.get_encoder, lib.chat
        dr.lib.load_slice = lambda path: {"items": chosen}
        dr.lib.load_user = lambda uid: data
        dr.default_index_factory = lambda uid, d, need_dense, enc: FakeIndex(d)
        lib.get_encoder = lambda *a, **k: "ENC"
        lib.chat = scripted_chat()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                summary = dr.run("unused", tmp, SETTINGS, memories=("sparse",),
                                 workers=2, only_keys=keys[:2])
                rows, dup = dr.unique_rows(lib.read_jsonl(
                    os.path.join(tmp, "DecompositionReason_SparseRAG_k20.jsonl")))
        finally:
            dr.lib.load_slice, dr.lib.load_user = old_slice, old_user
            dr.default_index_factory, lib.get_encoder, lib.chat = old_factory, old_enc, old_chat
        self.assertEqual(summary["memories"]["DecompositionReason_SparseRAG_k20"]["rows"], 2)
        self.assertEqual({lib.qkey(r) for r in rows}, set(keys[:2]))
        self.assertEqual(dup, 0)

    def test_cut_line_question_is_redone_and_deduped(self):
        items = [item()]
        with tempfile.TemporaryDirectory() as tmp:
            self._run(items, tmp)
            path = os.path.join(tmp, "DecompositionReason_SparseRAG_k20.jsonl")
            # simulate a cut in the middle of the only complete row: remove its last 30 characters
            rows = [json.loads(line) for line in open(path) if line.strip()]
            with open(path, "w") as f:
                f.write(json.dumps(rows[0])[:-30])
            summary = self._run(items, tmp)
            self.assertEqual(summary["memories"]["DecompositionReason_SparseRAG_k20"]["rows"], 1)
            rows, dup = dr.unique_rows(lib.read_jsonl(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(dup, 0)

    def test_duplicate_keys_are_reported_by_unique_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = [item()]
            self._run(items, tmp)
            path = os.path.join(tmp, "DecompositionReason_SparseRAG_k20.jsonl")
            row = lib.read_jsonl(path)[0]
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows, dup = dr.unique_rows(lib.read_jsonl(path))
            self.assertEqual((len(rows), dup), (1, 1))

    def test_deadline_stops_before_submitting(self):
        items = [item(rc_id=f"RC{i:06d}#02", hop=2) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(items, tmp, deadline=time.time() - 1000)
            ms = summary["memories"]["DecompositionReason_SparseRAG_k20"]
            self.assertEqual(ms["rows"], 0)
            self.assertEqual(ms["stopped"], "deadline")

    def test_circuit_breaker_on_repeated_failures(self):
        def broken(prompt, max_tokens, retries=5, temperature=0):
            raise RuntimeError("connection refused")

        items = [item(rc_id=f"RC{i:06d}#02", hop=2) for i in range(40)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(items, tmp, chat=broken, workers=2)
            ms = summary["memories"]["DecompositionReason_SparseRAG_k20"]
            self.assertEqual(ms["rows"], 0)
            self.assertTrue(ms["stopped"] and ms["stopped"].startswith("circuit_breaker"))
            failures = lib.read_jsonl(os.path.join(tmp, "failures_SparseRAG.jsonl"))
            self.assertGreaterEqual(len(failures), 20)
            self.assertTrue(all(f["stage"].startswith(("divide", "solve", "merge")) for f in failures))

    def test_failure_recorded_without_row_and_without_naive_fallback(self):
        calls = {"n": 0}

        def flaky(prompt, max_tokens, retries=5, temperature=0):
            calls["n"] += 1
            if "Current Sub-question" in prompt:
                raise RuntimeError("engine died")
            return scripted_chat()(prompt, max_tokens, temperature)

        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run([item()], tmp, chat=flaky)
            ms = summary["memories"]["DecompositionReason_SparseRAG_k20"]
            self.assertEqual(ms["rows"], 0)
            self.assertEqual(ms["failures"], 1)
            failures = lib.read_jsonl(os.path.join(tmp, "failures_SparseRAG.jsonl"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["stage"], "solve[0]")
            self.assertEqual(lib.read_jsonl(os.path.join(tmp, "DecompositionReason_SparseRAG_k20.jsonl")), [])
            self.assertNotIn("Naive", open(os.path.join(tmp, "failures_SparseRAG.jsonl")).read())

    def test_concurrent_greedy_no_duplicate_keys(self):
        temps = []

        def chat(prompt, max_tokens, retries=5, temperature=0):
            temps.append(temperature)
            time.sleep(0.005)
            return scripted_chat()(prompt, max_tokens, temperature)

        items = [item(rc_id=f"RC{i:06d}#02", hop=2) for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._run(items, tmp, chat=chat, workers=8)
            ms = summary["memories"]["DecompositionReason_SparseRAG_k20"]
            self.assertEqual(ms["rows"], 8)
            rows, dup = dr.unique_rows(lib.read_jsonl(ms["path"]))
            self.assertEqual((len(rows), dup), (8, 0))
            self.assertEqual(ms["calls"], 8 * 4)
        self.assertTrue(temps and all(t == 0.0 for t in temps))


class AnalysisToolTests(unittest.TestCase):
    def _fixture(self, tmp):
        """Two-question slice + released-data-shaped qa files + DR rows produced by the real runner."""
        exp = os.path.join(tmp, "exp")
        runs = os.path.join(tmp, "runs")
        qa_dir = os.path.join(tmp, "qa")
        os.makedirs(qa_dir)
        slice_path = os.path.join(exp, "s306.json")
        items = [{"uid": 0, "hop": 2, "rc_id": "RC000001#02", "n_gold": 1, "leak_any": False},
                 {"uid": 0, "hop": 5, "rc_id": "RC000002#05", "n_gold": 1, "leak_any": True}]
        os.makedirs(exp)
        with open(slice_path, "w") as f:
            json.dump({"meta": {"n": 2, "seed": 0, "per_hop": 1}, "items": items}, f)
        data = make_data()
        data["qa"]["5"]["RC000002#05"]["answer"] = "125"
        with open(os.path.join(qa_dir, "question_answer_0.json"), "w") as f:
            json.dump(data, f)
        runner = dr.Runner(SETTINGS, chat=scripted_chat(), user_loader=lambda uid: data,
                           index_factory=lambda uid, d, need_dense, enc: FakeIndex(d))
        old_enc = lib.get_encoder
        lib.get_encoder = lambda *a, **k: "ENC"
        try:
            for mem in ("sparse", "dense"):
                runner.run(items, [mem], runs, workers=2)
        finally:
            lib.get_encoder = old_enc
        return slice_path, runs, qa_dir

    def test_validator_accepts_a_clean_small_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            slice_path, runs, qa_dir = self._fixture(tmp)
            out = subprocess_run([sys.executable, os.path.join(EXP, "validate_dr.py"), "--runs", runs,
                                  "--slice", slice_path, "--qa-dir", qa_dir,
                                  "--settings", os.path.join(EXP, "settings.json"),
                                  "--archived-nr", os.path.join(tmp, "none")])
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            rep = json.load(open(os.path.join(tmp, "validation_local.json")))
            self.assertTrue(rep["ok"])
            self.assertFalse(rep["complete_612"])  # 2-row fixture, not the 306-question run

    def test_summarizer_writes_a_summary(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed in this interpreter (use the repo venv)")
        with tempfile.TemporaryDirectory() as tmp:
            slice_path, runs, _qa_dir = self._fixture(tmp)
            out_md = os.path.join(tmp, "summary_dr.md")
            out = subprocess_run([sys.executable, os.path.join(EXP, "summarize_dr.py"),
                                  "--dr", runs, "--slice", slice_path,
                                  "--nr", os.path.join(tmp, "none"), "--sr", os.path.join(tmp, "none"),
                                  "--out", out_md])
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            text = open(out_md).read()
            self.assertIn("DR SparseRAG", text)
            self.assertIn("DR on the leak-free subset", text)
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary_dr.json")))


def subprocess_run(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
