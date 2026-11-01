"""Throughput test against the running vLLM server, with real grid prompts (BM25 memory, top-20).

Phases (default ~8 min): NaiveReason at 8 / 32 / 64 concurrent requests, a single-request
decode-speed probe, and SequentialReason (5 calls per question) at 64 concurrent questions.
Each phase runs for a fixed wall time and counts finished calls and tokens from the server's
usage fields. Appends one json line per phase to results/throughput/throughput.jsonl.
Questions are drawn cyclically from the whole slice (all 12 users) and every request gets a unique
"[req N]" line in front, so vLLM's prefix cache cannot answer repeats; the numbers are therefore cold-cache
throughput (a first attempt without the nonce, on 2 users, was inflated by exact-prompt cache hits and was discarded).
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid  # noqa: E402
import lib  # noqa: E402


_n = itertools.count()
_orig_chat = lib.chat


def _chat_nonce(prompt, max_tokens, retries=5):
    return _orig_chat(f"[req {next(_n)}]\n" + prompt, max_tokens, retries)


lib.chat = _chat_nonce


def phase(name, runner, items, mem, workers, seconds, out):
    stop = time.time() + seconds
    cyc = itertools.cycle(items)
    lock = threading.Lock()
    tot = {"questions": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "lat": []}

    def work():
        while time.time() < stop:
            with lock:
                it = next(cyc)
            try:
                r = runner.solve(mem, it)
            except Exception as e:  # noqa: BLE001
                print("err", e, flush=True)
                continue
            with lock:
                tot["questions"] += 1
                tot["calls"] += r["calls"]
                tot["prompt_tokens"] += r["prompt_tokens"]
                tot["completion_tokens"] += r["completion_tokens"]
                tot["lat"].append(r["secs"] / max(r["calls"], 1))

    t = time.time()
    ths = [threading.Thread(target=work) for _ in range(workers)]
    [x.start() for x in ths]
    [x.join() for x in ths]
    el = time.time() - t
    lat = sorted(tot.pop("lat"))
    rec = {"phase": name, "workers": workers, "wall_s": round(el, 1), **tot,
           "calls_per_s": round(tot["calls"] / el, 2), "questions_per_min": round(tot["questions"] / el * 60, 1),
           "prompt_tok_per_s": round(tot["prompt_tokens"] / el), "completion_tok_per_s": round(tot["completion_tokens"] / el),
           "median_call_latency_s": round(lat[len(lat) // 2], 2) if lat else None,
           "avg_prompt_tokens_per_call": round(tot["prompt_tokens"] / max(tot["calls"], 1))}
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--seconds", type=int, default=90)
    a = ap.parse_args()
    items = lib.load_slice(a.slice)["items"]
    out = os.path.join(lib.OUT, "throughput", "throughput.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    P = lib.load_prompts()
    nr = grid.Runner(SimpleNamespace(reason="naive", topk=20, max_step=5), P)
    sr = grid.Runner(SimpleNamespace(reason="seq", topk=20, max_step=5), P)
    for u in lib.USERS:  # pre-build BM25 indexes outside the timed phases (shared by nr and sr)
        nr.user(u, False)
    sr.data, sr.index = nr.data, nr.index
    phase("warmup_naive_sparse", nr, items, "sparse", 8, 20, out)
    for w, s in ((8, a.seconds), (32, a.seconds), (64, a.seconds)):
        phase(f"naive_sparse_c{w}", nr, items, "sparse", w, s, out)
    text, pt, ct, secs = lib.chat("Write a numbered list of 40 short facts about rivers.", 200)
    rec = {"phase": "single_request_decode", "completion_tokens": ct, "wall_s": round(secs, 2),
           "completion_tok_per_s": round(ct / secs, 1)}
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    phase("seq5_sparse_c64", sr, items, "sparse", 64, a.seconds, out)


if __name__ == "__main__":
    main()
