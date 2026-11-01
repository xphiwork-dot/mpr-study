"""Run NaiveReason / SequentialReason with several memory types against the local vLLM server.

Re-implements the call flow of upstream reasonings/NaiveReason.py, reasonings/SequentialReason.py and
baselines/{Ignoramus,Oracle,SparseRAG,DenseRAG}.py on an OpenAI-compatible endpoint, using the upstream
prompt templates (loaded from the pinned clone). Differences from upstream, all deliberate:
  - questions come from a slice file (all 12 users, stratified by hop), not one user;
  - greedy decoding (temperature 0); upstream's local-server decoding settings are not shipped;
  - answers are capped at 64 tokens, chain-of-thought steps at 200 tokens.
Resumable: every finished (memory, user, question) is one jsonl line; a rerun skips finished ones.

Usage: grid.py --slice cache/slice.json --reason naive --memories ignoramus,oracle,sparse,dense --tag s1
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

MEMS = {"ignoramus": "Ignoramus", "oracle": "Oracle", "sparse": "SparseRAG", "dense": "DenseRAG"}


class Runner:
    def __init__(self, a, prompts):
        self.a, self.P = a, prompts
        self.data, self.index = {}, {}
        self.lock = threading.Lock()
        self.enc = None
        self.enc_lock = threading.Lock()

    def user(self, uid, need_dense):
        with self.lock:
            if uid not in self.data:
                self.data[uid] = lib.load_user(uid)
            if need_dense and self.enc is None:
                self.enc = lib.get_encoder()
            if uid not in self.index:
                self.index[uid] = lib.UserIndex(uid, self.data[uid], need_dense=need_dense, encoder=self.enc)
            elif need_dense and self.index[uid].emb is None:  # index first built for BM25 only
                self.index[uid].add_dense(self.enc)
            return self.data[uid], self.index[uid]

    def refs(self, mem, ix, query, qa):
        if mem == "ignoramus":
            return None, ""
        if mem == "oracle":
            ids = list(qa["message"])  # upstream joins as given (ids can repeat)
        elif mem == "sparse":
            ids = ix.sparse(query, self.a.topk)
        else:
            with self.enc_lock:
                ids = ix.dense(query, self.a.topk)
        return ids, "\n".join(ix.messages[m] for m in ids)

    def call(self, kind, mem, params, max_tokens, acc):
        suffix = "Ignoramus" if mem == "ignoramus" else "RAG"
        prompt = getattr(self.P, f"{kind}_{suffix}_Prompt_Template").format(**params)
        text, pt, ct, secs = lib.chat(prompt, max_tokens)
        acc["calls"] += 1
        acc["prompt_tokens"] += pt
        acc["completion_tokens"] += ct
        acc["secs"] += secs
        return text

    def solve(self, mem, item):
        a = self.a
        data, ix = self.user(item["uid"], mem == "dense")
        qa = data["qa"][str(item["hop"])][item["rc_id"]]
        q = qa["question"]
        acc = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "secs": 0.0}
        first_ids = None
        if a.reason == "naive":
            first_ids, r = self.refs(mem, ix, q, qa)
            resp = self.call("Naive", mem, {"question": q, "references": r}, 64, acc)
            thoughts = None
        else:
            n = a.max_step
            thoughts = []

            def fmt():
                return "\n".join("(Step %d) %s" % (i + 1, t) for i, t in enumerate(thoughts))

            first_ids, r = self.refs(mem, ix, q, qa)
            thoughts.append(self.call("Sequential_Start", mem, {"max_step": n, "question": q, "references": r}, 200, acc))
            for step in range(1, n - 1):
                _, r = self.refs(mem, ix, thoughts[-1], qa)
                thoughts.append(self.call("Sequential_Middle", mem, {
                    "current_step": step + 1, "max_step": n, "question": q, "references": r,
                    "previous_thoughts": fmt()}, 200, acc))
            _, r = self.refs(mem, ix, thoughts[-1], qa)
            resp = self.call("Sequential_Final", mem, {"question": q, "references": r, "previous_thoughts": fmt()}, 64, acc)
        row = {"uid": item["uid"], "rc_id": item["rc_id"], "k_hop": item["hop"], "memory": lib_name(mem),
               "response": resp, "ground_truth": qa["answer"], "first_retrieved": first_ids, **acc}
        if thoughts is not None:
            row["thoughts"] = thoughts
        return row


def lib_name(mem):
    return MEMS[mem]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--reason", choices=["naive", "seq"], required=True)
    ap.add_argument("--memories", default="ignoramus,oracle,sparse,dense")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-step", type=int, default=5)
    ap.add_argument("--tag", required=True, help="output subdirectory under results/")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N questions")
    a = ap.parse_args()

    items = lib.load_slice(a.slice)["items"]
    if a.limit:
        items = items[: a.limit]
    runner = Runner(a, lib.load_prompts())
    t0 = time.time()
    failures = 0
    for mem in a.memories.split(","):
        name = f"{'NaiveReason' if a.reason == 'naive' else 'SequentialReason'}_{lib_name(mem)}_k{a.topk}"
        if a.reason == "seq":
            name += f"_s{a.max_step}"
        path = os.path.join(lib.OUT, a.tag, name + ".jsonl")
        done = {lib.qkey(r) for r in lib.read_jsonl(path)}
        todo = [i for i in items if f"{i['uid']}:{i['rc_id']}" not in done]
        print(f"[{name}] {len(done)} done, {len(todo)} to do", flush=True)
        if not todo:
            continue
        w = lib.JsonlWriter(path)
        tm = time.time()
        n = pt = ct = 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(runner.solve, mem, i) for i in todo]
            for k, fu in enumerate(as_completed(futs), 1):
                try:
                    row = fu.result()
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"  failure: {e}", flush=True)
                    continue
                w.write(row)
                n += row["calls"]
                pt += row["prompt_tokens"]
                ct += row["completion_tokens"]
                if k % 50 == 0 or k == len(todo):
                    el = time.time() - tm
                    print(f"  {k}/{len(todo)}  {el:.0f}s  {n / el:.1f} calls/s  in {pt / el:.0f} tok/s  out {ct / el:.0f} tok/s", flush=True)
        el = time.time() - tm
        if n == 0:
            continue
        with open(os.path.join(lib.OUT, a.tag, "timing.jsonl"), "a") as f:
            f.write(json.dumps({"file": name, "questions": len(todo), "calls": n, "prompt_tokens": pt,
                                "completion_tokens": ct, "wall_s": round(el, 1), "workers": a.workers}) + "\n")
    print(f"grid done in {time.time() - t0:.0f}s, failures={failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
