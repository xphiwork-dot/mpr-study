"""Run session 4 on a local CUDA GPU using the farm-generated fact slice.

Set T4_ROOT to the repo root, MPR_QA_DIR to data/unz/mpr_dataset/question_answer,
and LAPTOP_SFT_WORK to a fresh scratch directory before running this script.
The runner replaces the result files; retain existing results before rerunning.
"""
# Written by us, no upstream reference.
import json
import os
import random
import sys
import runpy
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import PeftModel
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent / 't4'))
import lib
import sft
from score import contains, em

OUT = Path('results/laptop/sft')
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
WORK = Path(os.environ['LAPTOP_SFT_WORK'])
WORK.mkdir(parents=True, exist_ok=True)
if any(WORK.glob('c*_*/lora_*/epoch_*')):
    raise RuntimeError('Use a fresh LAPTOP_SFT_WORK to avoid misleading resumed training times')
read = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line]
pairs = read(OUT / 'trainset.jsonl')
probe = read(OUT / 'probe.jsonl')
slice_ = json.loads((OUT / 'slice.json').read_text())
data = lib.load_user(slice_['user'])
ids = slice_['fact_ids']
texts = [data['message'][fid] for fid in ids]
assert len(ids) >= 300 and len(probe) > 0 and all(p['fid'] in ids for p in pairs)
P = lib.load_prompts()
times = {}
t0 = time.monotonic()
# CPU encoder leaves the shared 6 GB device available for the language model.
encoder = lib.get_encoder(device='cpu')
fact_emb = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
queries = [r[k] for k in ('question', 'paraphrase') for r in probe]
query_emb = encoder.encode(queries, normalize_embeddings=True, show_progress_bar=False)
refs = np.argsort(-(query_emb @ fact_emb.T), axis=1)[:, :20]
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(fact_emb)
labels = km.labels_
# Stable tie-break: lowest cluster number.
votes = [min(Counter(labels[ix]).items(), key=lambda x: (-x[1], x[0]))[0] for ix in refs]
times['embedding_clustering_seconds'] = round(time.monotonic() - t0, 2)
(OUT / 'clusters.json').write_text(json.dumps({'fact_ids': ids, 'labels': labels.tolist(), 'sizes': dict(Counter(map(int, labels)))}, indent=2))
blocks = {c: [p for p in pairs if labels[ids.index(p['fid'])] == c] for c in range(5)}
assert all(blocks.values())
a = SimpleNamespace(epochs=10, bs=4, seed=0)


def evaluate(model, tokenizer, mode, run, rag, use_adapter):
    """Written by us, no upstream reference."""
    started = time.monotonic()
    prompts = []
    for i, q in enumerate(queries):
        if rag:
            prompts.append(P.Naive_RAG_Prompt_Template.format(question=q, references='\n'.join(texts[j] for j in refs[i])))
        else:
            prompts.append(P.Naive_Ignoramus_Prompt_Template.format(question=q))
    rows = []
    groups = sorted(set(votes)) if use_adapter and mode == 5 else [0]
    for group in groups:
        indices = [i for i in range(len(queries)) if not (use_adapter and mode == 5) or votes[i] == group]
        if use_adapter:
            model.set_adapter(str(group))
            responses = sft.gen(model, tokenizer, [prompts[i] for i in indices], bs=4, max_new=64)
        else:
            with model.disable_adapter():
                responses = sft.gen(model, tokenizer, [prompts[i] for i in indices], bs=4, max_new=64)
        for i, response in zip(indices, responses):
            r = probe[i % len(probe)]
            rows.append({'id': i % len(probe), 'fid': r['fid'], 'question_type': 'seen' if i < len(probe) else 'paraphrase',
                         'question': queries[i], 'gold': r['answer'], 'response': response,
                         'retrieved_fact_ids': [ids[j] for j in refs[i]], 'selected_cluster': int(votes[i]) if use_adapter and mode == 5 else None,
                         'memory': 'retrieval' if rag else 'closed', 'model': 'adapter' if use_adapter else 'base',
                         'em': bool(em(response, r['answer'])), 'contains': bool(contains(response, r['answer']))})
    with (run / 'eval.jsonl').open('a') as f:
        for row in sorted(rows, key=lambda r: (r['question_type'], r['id'])):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return round(time.monotonic() - started, 2)


for preset, lr in [('strong', 2e-4), ('upstream', 1e-5)]:
    for mode in (1, 5):
        run = OUT / f'c{mode}_{preset}'
        run.mkdir(parents=True, exist_ok=True)
        (run / 'eval.jsonl').write_text('')
        start = time.monotonic()
        train_blocks = {0: pairs} if mode == 1 else blocks
        for c, block in train_blocks.items():
            sft.train_adapter(a, MODEL, block, str(WORK / f'c{mode}_{preset}' / f'lora_{c}'), lr, f'{preset} c{c}')
        timing = {'training_seconds': round(time.monotonic() - start, 2)}
        start = time.monotonic()
        model, tok = sft.load_model(MODEL)
        for c in train_blocks:
            path = str(WORK / f'c{mode}_{preset}' / f'lora_{c}' / 'epoch_10')
            if c == 0:
                model = PeftModel.from_pretrained(model, path, adapter_name=str(c))
            else:
                model.load_adapter(path, adapter_name=str(c))
        model.eval()
        timing['load_seconds'] = round(time.monotonic() - start, 2)
        for rag in (False, True):
            for enabled in (False, True):
                key = ('adapter' if enabled else 'base') + ('_retrieval' if rag else '_closed')
                timing[key + '_seconds'] = evaluate(model, tok, mode, run, rag, enabled)
        (run / 'timing.json').write_text(json.dumps(timing, indent=2))
        del model, tok
        torch.cuda.empty_cache()

lines = ['# Laptop session 4', '', 'Qwen2.5-0.5B-Instruct fp16; 300 user-0 facts; farm AskSFT pairs; 10 epochs; LoRA rank 8. Dense e5 top-20 is restricted to the same fact slice. Hybrid adapter selection uses a vote over retrieved facts even for closed-book answers; only retrieval conditions expose references to the model.', '', 'Contains uses the repository whole-word scorer; counts are correct / evaluated. Seen and paraphrased questions share facts, so these are not independent samples. Single small slice; differences are descriptive, not a general superiority claim.', '', '| Preset | Adapters | Question | Memory | Model | Contains | EM |', '|---|---:|---|---|---|---:|---:|']
for preset in ('strong', 'upstream'):
    for mode in (1, 5):
        rows = read(OUT / f'c{mode}_{preset}' / 'eval.jsonl')
        for q in ('seen', 'paraphrase'):
            for mem in ('closed', 'retrieval'):
                for model in ('base', 'adapter'):
                    subset = [r for r in rows if r['question_type'] == q and r['memory'] == mem and r['model'] == model]
                    lines.append(f"| {preset} | {mode} | {q} | {mem} | {model} | {sum(r['contains'] for r in subset)}/{len(subset)} | {sum(r['em'] for r in subset)}/{len(subset)} |")
lines += ['', '## Wall time (seconds)', '', f"Embedding and clustering: {times['embedding_clustering_seconds']}. Farm generation: " + str(json.loads((OUT / 'farm_usage.json').read_text())['wall_seconds']) + '.', '']
for preset in ('strong', 'upstream'):
    for mode in (1, 5):
        timing = json.loads((OUT / f'c{mode}_{preset}' / 'timing.json').read_text())
        lines.append(f"- {preset}, {mode} adapter(s): " + ', '.join(f'{k}={v}' for k, v in timing.items()))
(OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
(OUT / 'timing.json').write_text(json.dumps(times, indent=2))
runpy.run_path('scripts/laptop_sft_compare.py')
