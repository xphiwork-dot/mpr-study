"""DecompositionReason (DR) over the T4 helpers, for the S6 Kaggle session.

Adapts upstream `reasonings/DecompositionReason.py` at commit a24953c (read-only audit in README.md (d)):

  1. Divide: one inference call, NO retrieval, prompt Decomposition_Divide_{RAG,Ignoramus}_Prompt_Template.
  2. Solve: for each sub-question, IN ORDER, one retrieval + inference; the numbered current_state lists all
     sub-questions, earlier answers as '(Answer: ...)' and the current line as '(Current Sub-question)'. The
     retrieval key is exactly f'(question) {question}\\n (sub-question) {sub_question}'.
  3. Merge: one retrieval keyed on the answered current_state + inference, prompt Decomposition_Merge_*.

Calls per question = 1 + N + 1, N <= decompose_num (5), so 3..7. The prompt templates themselves are
upstream's (loaded from the pinned clone at run time), never copied into this repo.

Deliberate adaptations, all recorded in settings.json and README.md (d):
  - questions come from a slice (all 12 users) instead of one user's file, like grid.py;
  - the engine is a local vLLM OpenAI endpoint with greedy decoding, 200/200/64-token caps;
  - parser 'ref_faithful_drop_blank': upstream's `text.split('\\n')[:n]` but whitespace-only lines are
    dropped (counted per row); if nothing remains the original question is the single sub-question and the
    row is flagged fallback=true. Numbering and all other text are kept verbatim;
  - transport failures never fall back to NR: the question is written to failures_*.jsonl and retried by a
    later rerun (the jsonl is resumable).

Everything the caller needs is injectable (chat, user loader, index factory, prompts) so
tests/test_dr.py can compare the full call flow with the upstream class and with stub engines.

CLI (on the Kaggle kernel or a machine with the input cache, dataset and pinned clone):
  python dr.py --slice results/slices/s306.json --memories sparse,dense --tag results/runs \
               --workers 32 --time-budget-s 5400 --status-file results/status.json
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
# `lib.py` sits next to this file in the Kaggle dataset; in the repo it is in ../_shared
for _cand in (
    os.environ.get("MPR_SHARED_DIR"),
    os.path.join(HERE, "..", "_shared"),
    HERE,
):
    if _cand and os.path.isfile(os.path.join(_cand, "lib.py")):
        sys.path.insert(0, _cand)
        break
import lib  # noqa: E402

MEMS = {
    "ignoramus": "Ignoramus",
    "oracle": "Oracle",
    "sparse": "SparseRAG",
    "dense": "DenseRAG",
}
SETTINGS_PATH = os.path.join(HERE, "settings.json")


class QuestionFailure(RuntimeError):
    """A question failed after the trace was started; carries the stage and the partial trace."""

    def __init__(self, stage, error, trace):
        super().__init__(f"{stage}: {error!r}")
        self.stage = stage
        self.error = error
        self.trace = trace


def load_settings(path=SETTINGS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_division(text, n, question, policy="ref_faithful_drop_blank"):
    """Return (sub_questions, meta).

    Upstream: `returned_text.split('\\n')[:decompose_num]`, no validation, no fallback.
      policy="upstream_strict": exactly upstream (kept only for the parity mock).
      policy="ref_faithful_drop_blank" (primary): upstream rule with whitespace-only lines dropped
        (they cannot be sub-questions; counted per row), and a single documented fallback when nothing
        non-blank is left: the question is the one sub-question, flagged fallback=true. Numbering and
        all other text are kept verbatim. See settings.json 'parser' and README (d).
    """
    raw_lines = str(text).split("\n")
    if policy == "upstream_strict":
        kept = raw_lines
    elif policy == "ref_faithful_drop_blank":
        kept = [ln.strip() for ln in raw_lines if ln.strip()]
    else:
        raise ValueError(f"unknown parser policy {policy!r}")
    over = max(0, len(kept) - n)
    subs = kept[:n]
    meta = {
        "policy": policy,
        "raw_lines": len(raw_lines),
        "dropped_blank": len(raw_lines) - len(kept),
        "over_limit": over,
        "fallback": False,
    }
    if not subs:
        if policy == "upstream_strict":
            subs = [
                ""
            ]  # upstream would use the empty string here; not used in the experiment
        else:
            subs = [question]
            meta["fallback"] = True
    return subs, meta


def build_current_state(sub_questions, answers, qid):
    """Exact upstream line formats (DecompositionReason.py:18-26)."""
    lines = []
    for qqid in range(len(sub_questions)):
        if qqid < qid:
            lines.append(
                "%d. %s (Answer: %s)" % (qqid + 1, sub_questions[qqid], answers[qqid])
            )
        elif qqid == qid:
            lines.append(
                "%d. %s (Current Sub-question)" % (qqid + 1, sub_questions[qqid])
            )
        else:
            lines.append("%d. %s" % (qqid + 1, sub_questions[qqid]))
    return "\n".join(lines)


def build_answered_state(sub_questions, answers):
    """Exact upstream merge state (DecompositionReason.py:35-36)."""
    return "\n".join(
        "%d. %s (Answer: %s)" % (qqid + 1, sub_questions[qqid], answers[qqid])
        for qqid in range(len(sub_questions))
    )


def solve_retrieval_key(question, sub_question):
    """Exact upstream knowledge_key for Decomposition_Solve (DecompositionReason.py:31)."""
    return f"(question) {question}\n (sub-question) {sub_question}"


def merge_retrieval_key(answered_state):
    """Exact upstream knowledge_key for Decomposition_Merge (DecompositionReason.py:40)."""
    return answered_state


def unique_rows(rows):
    """Deduplicate a result list by uid:rc_id, last row wins (a cut jsonl tail can repeat a key).

    Returns (rows_in_first_seen_order, duplicate_count). The scorer/validator use this so that an
    interrupted-then-resumed file cannot inflate n.
    """
    by, order, dup = {}, [], 0
    for r in rows:
        k = lib.qkey(r)
        if k in by:
            dup += 1
        else:
            order.append(k)
        by[k] = r
    return [by[k] for k in order], dup


def ensure_trailing_newline(path):
    """A jsonl file cut by a dying kernel can lack its final newline; the next append would then be
    glued to the cut line and lost by the JSON reader. Add the missing newline once before appending."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def default_index_factory(uid, data, need_dense, encoder):
    return lib.UserIndex(uid, data, need_dense=need_dense, encoder=encoder)


class Runner:
    def __init__(
        self, settings, prompts=None, chat=None, user_loader=None, index_factory=None
    ):
        self.s = settings
        self.P = prompts if prompts is not None else lib.load_prompts()
        self.chat_fn = chat or lib.chat
        self.user_loader = user_loader or lib.load_user
        self.index_factory = index_factory or default_index_factory
        self.topk = int(settings["retrieval"]["topk"])
        self.n = int(settings["method"]["decompose_num"])
        self.max_tokens = settings["decoding"]["max_tokens"]
        self.temperature = float(settings["decoding"]["temperature"])
        self.policy = settings["parser"]["policy"]
        self.data, self.index = {}, {}
        self.lock = threading.Lock()
        self.enc = None
        self.enc_lock = threading.Lock()
        self.settings_sha256 = config_hash(settings)

    # ---------- retrieval ----------
    def user(self, uid, need_dense):
        with self.lock:
            if uid not in self.data:
                self.data[uid] = self.user_loader(uid)
            if need_dense and self.enc is None:
                self.enc = lib.get_encoder()
            if uid not in self.index:
                self.index[uid] = self.index_factory(
                    uid, self.data[uid], need_dense, self.enc
                )
            elif (
                need_dense and self.index[uid].emb is None
            ):  # first built for BM25 only
                self.index[uid].add_dense(self.enc)
            return self.data[uid], self.index[uid]

    def refs(self, mem, ix, query, qa):
        if mem == "ignoramus":
            return None, ""  # upstream Ignoramus.response never retrieves
        if mem == "oracle":
            ids = list(qa["message"])  # upstream joins the gold statements as given
        elif mem == "sparse":
            ids = ix.sparse(query, self.topk)
        else:
            with self.enc_lock:
                ids = ix.dense(query, self.topk)
        return ids, "\n".join(ix.messages[m] for m in ids)

    # ---------- prompts and calls ----------
    def template_name(self, kind, mem):
        suffix = "Ignoramus" if mem == "ignoramus" else "RAG"
        return f"{kind}_{suffix}_Prompt_Template"

    def format_prompt(self, kind, mem, params):
        template = getattr(self.P, self.template_name(kind, mem))
        return template.format(**params)

    def call(
        self, kind, mem, params, max_tokens, acc, trace, references=None, stage=None
    ):
        p = dict(params)
        if mem != "ignoramus":
            p["references"] = (
                references  # RAGMethod.response:42 sets it; Ignoramus templates have no {references}
            )
        prompt = self.format_prompt(kind, mem, p)
        try:
            text, pt, ct, secs = self.chat_fn(
                prompt, max_tokens, temperature=self.temperature
            )
        except Exception as e:  # noqa: BLE001 - transport, server or parse errors all fail the question
            trace["stage"] = stage or kind
            raise QuestionFailure(stage or kind, e, trace) from e
        acc["calls"] += 1
        acc["prompt_tokens"] += pt
        acc["completion_tokens"] += ct
        acc["secs"] += secs
        trace["calls"].append(
            {
                "kind": kind,
                "prompt_sha256": sha256_text(prompt),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "secs": round(secs, 4),
                "max_tokens": max_tokens,
                "empty_response": not str(text).strip(),
            }
        )
        return text

    # ---------- the DR flow ----------
    def solve(self, mem, item):
        assert mem in MEMS, mem
        data, ix = self.user(item["uid"], mem == "dense")
        qa = data["qa"][str(item["hop"])][item["rc_id"]]
        question = qa["question"]
        acc = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "secs": 0.0}
        trace = {"stage": "divide", "calls": []}

        divide_raw = self.call(
            "Decomposition_Divide",
            mem,
            {"question": question, "max_sub_question": self.n},
            self.max_tokens["divide"],
            acc,
            trace,
            stage="divide",
        )
        subs, pmeta = parse_division(divide_raw, self.n, question, self.policy)

        answers, states, solve_queries, solve_ids = [], [], [], []
        trace["stage"] = "solve"
        for qid, sub_question in enumerate(subs):
            current_state = build_current_state(subs, answers, qid)
            key = solve_retrieval_key(question, sub_question)
            ids, references = self.refs(mem, ix, key, qa)
            answer = self.call(
                "Decomposition_Solve",
                mem,
                {
                    "question": question,
                    "current_state": current_state,
                    "sub_question": sub_question,
                },
                self.max_tokens["solve"],
                acc,
                trace,
                references=references,
                stage=f"solve[{qid}]",
            )
            answers.append(answer)
            states.append(current_state)
            solve_queries.append(key)
            solve_ids.append(ids)

        trace["stage"] = "merge"
        merge_state = build_answered_state(subs, answers)
        merge_key = merge_retrieval_key(merge_state)
        merge_ids, merge_references = self.refs(mem, ix, merge_key, qa)
        response = self.call(
            "Decomposition_Merge",
            mem,
            {"question": question, "current_state": merge_state},
            self.max_tokens["merge"],
            acc,
            trace,
            references=merge_references,
            stage="merge",
        )

        row = {
            "uid": item["uid"],
            "rc_id": item["rc_id"],
            "k_hop": int(item["hop"]),
            "memory": MEMS[mem],
            "reason": "DecompositionReason",
            "response": response,
            "ground_truth": qa["answer"],
            "divide_raw": divide_raw,
            "parser": pmeta,
            "sub_questions": subs,
            "sub_answers": answers,
            "current_states": states,
            "solve_queries": solve_queries,
            "solve_retrieved": solve_ids,
            "merge_query": merge_key,
            "merge_retrieved": merge_ids,
            "first_retrieved": solve_ids[0] if solve_ids else None,
            "calls": acc["calls"],
            "prompt_tokens": acc["prompt_tokens"],
            "completion_tokens": acc["completion_tokens"],
            "secs": acc["secs"],
            "call_trace": trace["calls"],
            "settings_sha256": self.settings_sha256,
        }
        return row

    def solve_naive(self, mem, item):
        """One-call NaiveReason, used only for the deployment sanity check against archived S1 rows.

        Upstream reasonings/NaiveReason.py:8 passes the question as both param and retrieval key.
        """
        data, ix = self.user(item["uid"], mem == "dense")
        qa = data["qa"][str(item["hop"])][item["rc_id"]]
        question = qa["question"]
        acc = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "secs": 0.0}
        trace = {"stage": "naive", "calls": []}
        ids, references = self.refs(mem, ix, question, qa)
        response = self.call(
            "Naive",
            mem,
            {"question": question},
            64,
            acc,
            trace,
            references=references,
            stage="naive",
        )
        return {
            "uid": item["uid"],
            "rc_id": item["rc_id"],
            "k_hop": int(item["hop"]),
            "memory": MEMS[mem],
            "reason": "NaiveReason",
            "response": response,
            "ground_truth": qa["answer"],
            "first_retrieved": ids,
            "calls": acc["calls"],
            "prompt_tokens": acc["prompt_tokens"],
            "completion_tokens": acc["completion_tokens"],
            "secs": acc["secs"],
            "call_trace": trace["calls"],
            "settings_sha256": self.settings_sha256,
        }

    def run_item(self, mem, item):
        """Return (row|None, failure|None); never falls back to a different reasoning structure."""
        try:
            return self.solve(mem, item), None
        except QuestionFailure as f:
            return None, {
                "uid": item["uid"],
                "rc_id": item["rc_id"],
                "memory": MEMS[mem],
                "stage": f.stage,
                "error": repr(f.error),
                "trace": f.trace,
                "ts": time.time(),
            }
        except Exception as e:  # noqa: BLE001 - a bug in the flow also fails the question, not the run
            return None, {
                "uid": item["uid"],
                "rc_id": item["rc_id"],
                "memory": MEMS[mem],
                "stage": "unknown",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "ts": time.time(),
            }

    # ---------- resumable run ----------
    def run(
        self,
        items,
        memories,
        out_dir,
        workers,
        deadline=None,
        on_progress=None,
        max_consecutive_failures=20,
    ):
        os.makedirs(out_dir, exist_ok=True)
        summary = {"memories": {}, "started": time.time(), "deadline": deadline}
        for mem in memories:
            name = f"DecompositionReason_{MEMS[mem]}_k{self.topk}"
            path = os.path.join(out_dir, name + ".jsonl")
            fpath = os.path.join(out_dir, f"failures_{MEMS[mem]}.jsonl")
            done = {lib.qkey(r) for r in lib.read_jsonl(path)}
            todo = [it for it in items if f"{it['uid']}:{it['rc_id']}" not in done]
            ms = {
                "name": name,
                "path": path,
                "rows_before": len(done),
                "todo": len(todo),
                "rows": len(done),
                "failures": 0,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "wall_s": 0.0,
                "stopped": None,
            }
            summary["memories"][name] = ms
            print(f"[{name}] {len(done)} done, {len(todo)} to do", flush=True)
            if not todo:
                continue
            ensure_trailing_newline(path)
            ensure_trailing_newline(fpath)
            w, fw = lib.JsonlWriter(path), lib.JsonlWriter(fpath)
            t0 = time.time()
            consec = 0
            stopping = False
            pending = {}
            it = iter(todo)

            def submit(ex):
                if stopping:
                    return False
                try:
                    nxt = next(it)
                except StopIteration:
                    return False
                pending[ex.submit(self.run_item, mem, nxt)] = nxt
                return True

            with ThreadPoolExecutor(workers) as ex:
                for _ in range(min(len(todo), workers * 2)):
                    if deadline and time.time() > deadline - 120:
                        stopping = True
                        ms["stopped"] = "deadline"
                        break
                    if not submit(ex):
                        break
                while pending:
                    finished, _ = wait(
                        list(pending), timeout=15, return_when=FIRST_COMPLETED
                    )
                    for fu in finished:
                        pending.pop(fu)
                        row, failure = fu.result()
                        if row is not None:
                            w.write(row)
                            ms["rows"] += 1
                            ms["calls"] += row["calls"]
                            ms["prompt_tokens"] += row["prompt_tokens"]
                            ms["completion_tokens"] += row["completion_tokens"]
                            consec = 0
                        else:
                            fw.write(failure)
                            ms["failures"] += 1
                            consec += 1
                            print(
                                f"  failure {failure['uid']}:{failure['rc_id']} {failure['memory']} "
                                f"stage={failure['stage']} error={failure['error'][:200]}",
                                flush=True,
                            )
                        if on_progress:
                            on_progress(summary)
                        if consec >= max_consecutive_failures:
                            stopping = True
                            ms["stopped"] = (
                                f"circuit_breaker after {consec} consecutive failures"
                            )
                            print(f"  CIRCUIT BREAKER: {ms['stopped']}", flush=True)
                        if deadline and time.time() > deadline - 120:
                            if not stopping:
                                stopping = True
                                ms["stopped"] = ms["stopped"] or "deadline"
                                print(
                                    "  deadline reached: no new questions, draining in-flight calls",
                                    flush=True,
                                )
                    while len(pending) < workers * 2 and not stopping:
                        if deadline and time.time() > deadline - 120:
                            stopping = True
                            ms["stopped"] = ms["stopped"] or "deadline"
                            break
                        if not submit(ex):
                            break
            w.f.close()
            fw.f.close()
            ms["wall_s"] = round(time.time() - t0, 1)
            summary["memories"][name]["stopped"] = ms["stopped"]
            if on_progress:
                on_progress(summary)
            print(
                f"[{name}] finished: rows={ms['rows']} failures={ms['failures']} "
                f"calls={ms['calls']} wall={ms['wall_s']}s stopped={ms['stopped']}",
                flush=True,
            )
            if stopping and str(ms["stopped"] or "").startswith("circuit_breaker"):
                break
        summary["finished"] = time.time()
        return summary


def config_hash(settings):
    return sha256_text(json.dumps(settings, sort_keys=True))


def run(
    slice_path,
    out_dir,
    settings,
    memories=("sparse", "dense"),
    workers=32,
    deadline=None,
    limit=0,
    on_progress=None,
    status_file=None,
    only_keys=None,
):
    items = lib.load_slice(slice_path)["items"]
    if limit:
        items = items[:limit]
    if only_keys:
        wanted = set(only_keys)
        items = [it for it in items if f"{it['uid']}:{it['rc_id']}" in wanted]
    runner = Runner(settings)
    if status_file:

        def progress(summary):
            tmp = status_file + ".tmp"
            cur = {}
            try:
                with open(status_file, encoding="utf-8") as f:
                    cur = json.load(f)
            except Exception:  # noqa: BLE001 - missing/cut status file
                pass
            cur["progress"] = {"updated": time.time(), "summary": summary}
            cur.setdefault("status", "running")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cur, f, default=str)
            os.replace(tmp, status_file)
    else:
        progress = None
    summary = runner.run(
        items,
        list(memories),
        out_dir,
        workers,
        deadline=deadline,
        on_progress=progress,
        max_consecutive_failures=20,
    )
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slice", required=True)
    ap.add_argument("--out", required=True, help="result directory")
    ap.add_argument("--settings", default=SETTINGS_PATH)
    ap.add_argument("--memories", default="sparse,dense")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--time-budget-s",
        type=float,
        default=0,
        help="stop submitting when this many seconds elapsed",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="debug: first N questions only"
    )
    ap.add_argument("--status-file", default=None)
    ap.add_argument(
        "--only-keys",
        default=None,
        help="comma-separated uid:rc_id subset (smoke stage)",
    )
    a = ap.parse_args()
    settings = load_settings(a.settings)
    deadline = (time.time() + a.time_budget_s) if a.time_budget_s else None
    summary = run(
        a.slice,
        a.out,
        settings,
        memories=a.memories.split(","),
        workers=a.workers or int(settings["run"]["workers"]),
        deadline=deadline,
        limit=a.limit,
        status_file=a.status_file,
        only_keys=a.only_keys.split(",") if a.only_keys else None,
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "memories"}, indent=None)
    )
    failures = sum(m["failures"] for m in summary["memories"].values())
    expected = int(settings["run"]["expected_rows_per_system"])
    short = [m["name"] for m in summary["memories"].values() if m["rows"] < expected]
    ok = not short and failures == 0
    print(
        f"DR run {'OK' if ok else 'INCOMPLETE'}: failures={failures} short={short}",
        flush=True,
    )
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
