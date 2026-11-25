"""Session 4: small-model LoRA (AskSFT-style) and HybridMem (k-means clusters, one LoRA each) on a fact slice.

Our own thin implementation of the ideas in upstream baselines/SelfAskSFT.py and baselines/BlockSFT.py
(upstream code is not reused or vendored). Differences, all deliberate:
  - fp16 base model (a T4 has no bf16), LoRA weights in fp32, torch.autocast + GradScaler;
  - dynamic padding to the longest sample of the batch (upstream pads every sample to 4,608 tokens,
    docs/upstream-notes.md item 4); loss only on the answer tokens;
  - HybridMem adapters have distinct names and are reloaded from disk before evaluation, so training and
    evaluating in one run works (upstream crashes with `Adapter 1 not found`, item 5);
  - the fact slice is built around the evaluation questions: all gold statements of M user-0 questions
    (hops <= --max-hop) plus random other statements up to --facts, so the trained facts can matter.
Learning rate defaults to the upstream 1e-5 for --lr-preset upstream; --lr-preset strong uses 2e-4.

Resumable: self-ask pairs are appended to trainset.jsonl; each epoch saves the adapter (optimizer state is
not saved, so a resumed run restarts AdamW moments); finished conditions are skipped on rerun.

  sft.py --model Qwen/Qwen2.5-0.5B-Instruct --facts 600 --questions 60 --clusters 1 --lr-preset strong --tag sft_a
  sft.py ... --clusters 5      # HybridMem with 5 clusters (reuses the same facts and training pairs)
Two stages (session4.sh does both): `--stage ask` with the vLLM server up writes the self-ask pairs with the 7B model
(the 0.5B model cannot follow the two-line format; observed in the smoke test); then stop the server (t4.sh unserve)
and run `--stage train`, which trains and evaluates on the free GPU. `--ask-via local` lets the small model write them.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared")
)  # lib.py, grid.py, score.py (flat next to this file on the Colab VM)
import lib  # noqa: E402
from score import contains, em  # noqa: E402


def build_slice(a):
    d = lib.load_user(a.user)
    rows = [(h, rc, qa) for h, rc, qa in lib.all_questions(d) if h <= a.max_hop]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    qs = rows[: a.questions]
    gold = []
    for _, _, qa in qs:
        for m in qa["message"]:
            if m not in gold:
                gold.append(m)
    others = [m for m in d["message"] if m not in set(gold)]
    rng.shuffle(others)
    fact_ids = gold + others[: max(0, a.facts - len(gold))]
    return d, [(h, rc) for h, rc, _ in qs], fact_ids


def load_model(path, dtype_fp16=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16 if dtype_fp16 else torch.float32
    ).cuda()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def gen(model, tok, prompts, max_new=64, bs=32, sample=False):
    import torch

    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts[i : i + bs]
        ]
        enc = tok(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        with torch.no_grad():
            o = (
                model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                )
                if sample
                else model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            )
        outs += tok.batch_decode(
            o[:, enc.input_ids.shape[1] :], skip_special_tokens=True
        )
    return [x.strip() for x in outs]


def parse_pair(o):
    lines = [l.strip() for l in o.split("\n") if l.strip()]
    lines = [
        l.split(":", 1)[1].strip()
        if l.lower().startswith(("question:", "answer:"))
        else l
        for l in lines
    ]
    return (lines[0], lines[1]) if len(lines) == 2 and lines[0] and lines[1] else None


def ask_generate(a, model, tok, prompts):
    """Self-ask generation: the 7B vLLM server (default, as in the paper's 7B setting) or the small model itself."""
    if a.ask_via == "vllm":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(32) as ex:
            return [
                r[0]
                for r in ex.map(lambda p: lib.chat(p, 96, temperature=0.8), prompts)
            ]
    return gen(model, tok, prompts, max_new=96, bs=48, sample=True)


def make_trainset(a, model, tok, P, messages, fact_ids, path):
    done = {}
    for r in lib.read_jsonl(path):
        done.setdefault(r["fid"], []).append(r)
    todo = [f for f in fact_ids if len(done.get(f, [])) < a.ask]
    w = lib.JsonlWriter(path)
    for round_ in range(10):  # upstream allows up to 10 * ask tries per fact
        todo = [f for f in fact_ids if len(done.get(f, [])) < a.ask]
        if not todo:
            break
        prompts = [
            P.SelfAskSFT_Ask_Prompt_Template.format(message=messages[f])
            for f in todo
            for _ in range(a.ask - len(done.get(f, [])))
        ]
        owners = [f for f in todo for _ in range(a.ask - len(done.get(f, [])))]
        outs = ask_generate(a, model, tok, prompts)
        for f, o in zip(owners, outs):
            pr = parse_pair(o)
            if pr:
                row = {
                    "fid": f,
                    "message": messages[f],
                    "question": pr[0],
                    "answer": pr[1],
                }
                done.setdefault(f, []).append(row)
                w.write(row)
        print(
            f"self-ask round {round_}: facts with full pairs {sum(len(done.get(f, [])) >= a.ask for f in fact_ids)}/{len(fact_ids)}",
            flush=True,
        )
    return [r for f in fact_ids for r in done.get(f, [])[: a.ask]]


def encode_sft(tok, pairs):
    items = []
    for r in pairs:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        t = tok(r["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        items.append((p + t, [-100] * len(p) + t))
    return items


def train_adapter(a, base_path, pairs, out_dir, lr, tag):
    """Train one LoRA on pairs; returns after saving out_dir/epoch_<e>. Resumes from the latest epoch."""
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    model, tok = load_model(base_path)
    items = encode_sft(tok, pairs)
    os.makedirs(out_dir, exist_ok=True)
    have = sorted(
        int(x.split("_")[1]) for x in os.listdir(out_dir) if x.startswith("epoch_")
    )
    start = have[-1] if have else 0
    if start >= a.epochs:
        print(f"[{tag}] already trained {start} epochs", flush=True)
        del model
        return
    if start:
        model = PeftModel.from_pretrained(
            model, os.path.join(out_dir, f"epoch_{start}"), is_trainable=True
        )
        print(f"[{tag}] resume after epoch {start}", flush=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
            ),
        )
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params), (
        "LoRA params must be fp32 for GradScaler"
    )
    opt = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = (len(items) + a.bs - 1) // a.bs
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, 1 - (s + start * steps_per_epoch) / total)
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(start + 1, a.epochs + 1):
        order = list(range(len(items)))
        random.Random(a.seed * 1000 + ep).shuffle(order)
        t0, tot = time.time(), 0.0
        for s in range(0, len(order), a.bs):
            batch = [items[i] for i in order[s : s + a.bs]]
            L = max(len(x[0]) for x in batch)
            ids = torch.tensor(
                [x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch]
            ).cuda()
            att = torch.tensor(
                [[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch]
            ).cuda()
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += loss.item()
        model.save_pretrained(os.path.join(out_dir, f"epoch_{ep}"))
        print(
            f"[{tag}] epoch {ep}/{a.epochs} loss {tot / steps_per_epoch:.4f} {time.time() - t0:.1f}s",
            flush=True,
        )
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--facts", type=int, default=600)
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--max-hop", type=int, default=4)
    ap.add_argument("--user", type=int, default=0)
    ap.add_argument(
        "--ask",
        type=int,
        default=2,
        help="self-ask pairs per fact (upstream ask_proportion)",
    )
    ap.add_argument(
        "--clusters", type=int, default=1, help="1 = single LoRA, >1 = HybridMem"
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-preset", choices=["upstream", "strong"], default="strong")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ask-via",
        choices=["vllm", "local"],
        default="vllm",
        help="who writes the self-ask pairs: the 7B vLLM server (needs it running) or the small model itself",
    )
    ap.add_argument(
        "--stage",
        choices=["ask", "train", "all"],
        default="all",
        help="ask = only write the training pairs (server up); train = train and evaluate (server stopped)",
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--smoke", action="store_true", help="tiny run to check the code path"
    )
    a = ap.parse_args()
    if a.smoke:
        a.facts, a.questions, a.epochs = 40, 8, 2
    lr = 1e-5 if a.lr_preset == "upstream" else 2e-4
    base = os.path.join(lib.OUT, "sft", a.tag)
    run = os.path.join(base, f"c{a.clusters}_{a.lr_preset}")
    os.makedirs(run, exist_ok=True)
    P = lib.load_prompts()
    d, qids, fact_ids = build_slice(a)
    messages = d["message"]
    json.dump(
        {
            "args": vars(a),
            "questions": qids,
            "n_facts": len(fact_ids),
            "fact_ids": fact_ids,
        },
        open(os.path.join(base, "slice.json"), "w"),
    )
    print(f"slice: {len(qids)} questions, {len(fact_ids)} facts", flush=True)

    tpath = os.path.join(base, "trainset.jsonl")
    if a.stage != "train":
        model, tok = (None, None) if a.ask_via == "vllm" else load_model(a.model)
        pairs = make_trainset(a, model, tok, P, messages, fact_ids, tpath)
        print(f"train pairs: {len(pairs)}", flush=True)
        del model
        if a.stage == "ask":
            return
    else:
        pairs = [
            r
            for f in fact_ids
            for r in [x for x in lib.read_jsonl(tpath) if x["fid"] == f][: a.ask]
        ]
        print(f"train pairs (from cache): {len(pairs)}", flush=True)

    # ---- cluster (HybridMem) or one block (single LoRA) ----
    if a.clusters > 1:
        import numpy as np
        from sklearn.cluster import KMeans

        enc = lib.get_encoder()
        emb = enc.encode([r["message"] for r in pairs], normalize_embeddings=True)
        km = KMeans(n_clusters=a.clusters, random_state=42, n_init=10).fit(emb)
        pickle.dump(km, open(os.path.join(run, "cluster.pkl"), "wb"))
        blocks = {
            c: [r for r, l in zip(pairs, km.labels_) if l == c]
            for c in range(a.clusters)
        }
        json.dump(
            {str(c): len(v) for c, v in blocks.items()},
            open(os.path.join(run, "cluster_sizes.json"), "w"),
        )
    else:
        blocks = {0: pairs}
    t0 = time.time()
    for c, blk in blocks.items():
        if blk:
            train_adapter(
                a,
                a.model,
                blk,
                os.path.join(run, f"lora_{c}"),
                lr,
                f"c{c} n={len(blk)}",
            )
    train_s = time.time() - t0

    # ---- evaluation ----
    from peft import PeftModel

    model, tok = load_model(a.model)
    adapters = {}
    for c, blk in blocks.items():
        if blk:
            path = os.path.join(run, f"lora_{c}", f"epoch_{a.epochs}")
            if not adapters:
                model = PeftModel.from_pretrained(model, path, adapter_name=str(c))
            else:
                model.load_adapter(path, adapter_name=str(c))
            adapters[c] = str(c)
    model.eval()
    first = next(iter(adapters.values()))
    model.set_adapter(first)

    out = os.path.join(run, "eval.jsonl")
    done = {(r["cond"], r["id"]) for r in lib.read_jsonl(out)}
    w = lib.JsonlWriter(out)

    def emit(cond, id_, resp, gold, extra=None):
        if (cond, id_) not in done:
            w.write(
                {
                    "cond": cond,
                    "id": id_,
                    "response": resp,
                    "gold": gold,
                    **(extra or {}),
                }
            )

    # (1) recall of trained facts: raw question (train format) vs wrapped in the Naive_Ignoramus template (inference format)
    sub = pairs[:300]
    raw = [r["question"] for r in sub]
    wrapped = [
        P.Naive_Ignoramus_Prompt_Template.format(question=r["question"]) for r in sub
    ]
    for cond, prompts, use_ad in (
        ("recall_base_raw", raw, False),
        ("recall_lora_raw", raw, True),
        ("recall_lora_wrapped", wrapped, True),
    ):
        if all((cond, str(i)) in done for i in range(len(sub))):
            continue
        if use_ad:
            outs = gen(model, tok, prompts)
        else:
            with model.disable_adapter():
                outs = gen(model, tok, prompts)
        for i, (r, o) in enumerate(zip(sub, outs)):
            emit(cond, str(i), o, r["answer"])
    # (2) benchmark questions of the slice: no memory / BM25 RAG, with and without the adapter(s)
    ix = lib.UserIndex(a.user, d, need_dense=False)
    enc = lib.get_encoder() if a.clusters > 1 else None
    km = (
        pickle.load(open(os.path.join(run, "cluster.pkl"), "rb"))
        if a.clusters > 1
        else None
    )
    for cond in ("q_base_ignoramus", "q_lora_ignoramus", "q_base_rag", "q_lora_rag"):
        for h, rc in qids:
            id_ = f"{a.user}:{rc}"
            if (cond, id_) in done:
                continue
            qa = d["qa"][str(h)][rc]
            q = qa["question"]
            use_ad = "lora" in cond
            if "rag" in cond:
                refs_ids = ix.sparse(q, 20)
                prompt = P.Naive_RAG_Prompt_Template.format(
                    question=q, references="\n".join(messages[m] for m in refs_ids)
                )
            else:
                prompt = P.Naive_Ignoramus_Prompt_Template.format(question=q)
            if (
                use_ad and km is not None and "rag" in cond
            ):  # HybridMem: vote the cluster of the retrieved references
                labels = km.predict(
                    enc.encode(
                        [messages[m] for m in refs_ids], normalize_embeddings=True
                    )
                )
                top = int(max(set(labels), key=list(labels).count))
                model.set_adapter(str(top) if str(top) in adapters.values() else first)
            if use_ad:
                resp = gen(model, tok, [prompt])[0]
            else:
                with model.disable_adapter():
                    resp = gen(model, tok, [prompt])[0]
            emit(cond, id_, resp, qa["answer"], {"hop": h})
    # ---- summary ----
    rows = lib.read_jsonl(out)
    summ = {
        "args": vars(a),
        "train_seconds": round(train_s, 1),
        "n_train_pairs": len(pairs),
        "n_facts": len(fact_ids),
        "conditions": {},
    }
    for cond in sorted({r["cond"] for r in rows}):
        rs = [r for r in rows if r["cond"] == cond]
        summ["conditions"][cond] = {
            "n": len(rs),
            "em": round(sum(em(r["response"], r["gold"]) for r in rs) / len(rs), 4),
            "contains": round(
                sum(contains(r["response"], r["gold"]) for r in rs) / len(rs), 4
            ),
        }
    json.dump(summ, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
