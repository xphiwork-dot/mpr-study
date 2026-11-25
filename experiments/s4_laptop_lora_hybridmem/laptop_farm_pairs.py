"""Generate session-4 AskSFT pairs and question paraphrases on the LLM Farm.

Set T4_ROOT and MPR_QA_DIR as for laptop_sft.py. Export LLM_FARM_ENDPOINT,
LLM_FARM_KEY and SMALL_MODEL_NAME from the authorized local environment at
runtime; never persist their values. A partial generation can be resumed.
"""

# Written by us, no upstream reference.
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import AzureOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(
    0, str(HERE.parent / "_shared")
)  # lib.py, score.py; sft.py is in this folder
import lib
import sft

OUT = Path("results/laptop/sft")
OUT.mkdir(parents=True, exist_ok=True)
client = AzureOpenAI(
    api_key="dummy",
    azure_endpoint=os.environ["LLM_FARM_ENDPOINT"],
    api_version="2025-04-01-preview",
    default_headers={"genaiplatform-farm-subscription-key": os.environ["LLM_FARM_KEY"]},
)
model = os.environ["SMALL_MODEL_NAME"]
start = time.monotonic()
usage_path = OUT / "farm_usage.json"
usage = (
    json.loads(usage_path.read_text())
    if usage_path.exists()
    else {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
)
data = lib.load_user(0)
ids = list(data["message"])
random.Random(0).shuffle(ids)
ids = ids[:300]
(OUT / "slice.json").write_text(
    json.dumps({"user": 0, "seed": 0, "fact_ids": ids}, indent=2)
)
P = lib.load_prompts()


def call(prompt):
    """Written by us, no upstream reference."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
    )
    return response.choices[0].message.content, response.usage


def batch(prompts, name):
    """Written by us, no upstream reference."""
    path = OUT / f"{name}_raw.jsonl"
    records = lib.read_jsonl(path)
    outputs = [r["response"] for r in records]
    # Each completed response is persisted, including its token usage.
    usage["calls"] = sum(
        r["calls"]
        for r in lib.read_jsonl(OUT / "ask_raw.jsonl")
        + lib.read_jsonl(OUT / "paraphrase_raw.jsonl")
    )
    usage["prompt_tokens"] = sum(
        r["prompt_tokens"]
        for r in lib.read_jsonl(OUT / "ask_raw.jsonl")
        + lib.read_jsonl(OUT / "paraphrase_raw.jsonl")
    )
    usage["completion_tokens"] = sum(
        r["completion_tokens"]
        for r in lib.read_jsonl(OUT / "ask_raw.jsonl")
        + lib.read_jsonl(OUT / "paraphrase_raw.jsonl")
    )
    for i in range(len(outputs), len(prompts), 16):
        if usage["prompt_tokens"] + usage["completion_tokens"] >= 95000:
            raise RuntimeError("Farm token budget reached; generation incomplete")
        with ThreadPoolExecutor(max_workers=16) as pool, path.open("a") as f:
            for text, u in pool.map(call, prompts[i : i + 16]):
                record = {
                    "response": text,
                    "calls": 1,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                usage["calls"] += 1
                usage["prompt_tokens"] += u.prompt_tokens
                usage["completion_tokens"] += u.completion_tokens
                outputs.append(text)
        usage_path.write_text(json.dumps(usage, indent=2))
        print(
            f"farm calls={usage['calls']} tokens={usage['prompt_tokens'] + usage['completion_tokens']}",
            flush=True,
        )
    return outputs


prompts = [
    P.SelfAskSFT_Ask_Prompt_Template.format(message=data["message"][fid])
    for fid in ids
    for _ in range(2)
]
answers = batch(prompts, "ask")
pairs = []
for fid, answer in zip([f for f in ids for _ in range(2)], answers):
    parsed = sft.parse_pair(answer)
    if parsed:
        pairs.append(
            {
                "fid": fid,
                "message": data["message"][fid],
                "question": parsed[0],
                "answer": parsed[1],
            }
        )
with (OUT / "trainset.jsonl").open("w") as f:
    for row in pairs:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
# One seen question per distinct fact, avoiding a duplicate-fact evaluation sample.
seen = list({r["fid"]: r for r in reversed(pairs)}.values())[:60]
paraphrases = batch(
    [
        f"Rephrase the question below without changing its meaning or answer. Return only the rephrased question.\nQuestion: {r['question']}"
        for r in seen
    ],
    "paraphrase",
)
with (OUT / "probe.jsonl").open("w") as f:
    for r, paraphrase in zip(seen, paraphrases):
        f.write(
            json.dumps({**r, "paraphrase": paraphrase.strip()}, ensure_ascii=False)
            + "\n"
        )
usage["wall_seconds"] = round(time.monotonic() - start, 2)
usage["pairs"] = len(pairs)
usage["probe_facts"] = len(seen)
(OUT / "farm_usage.json").write_text(json.dumps(usage, indent=2))
