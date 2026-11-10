"""Append descriptive paired-condition comparisons to the session-4 generated summary.

Written by us, no upstream reference.
"""
import json
from pathlib import Path

root = Path('results/laptop/sft')
summary = root / 'summary.md'
text = summary.read_text().split('\n## Hybrid comparisons\n')[0].rstrip()
lines = [text, '', '## Hybrid comparisons', '', 'Counts use the same probe questions; these descriptive comparisons do not establish a general advantage.', '']
for preset in ('strong', 'upstream'):
    single = [json.loads(x) for x in (root / f'c1_{preset}' / 'eval.jsonl').read_text().splitlines()]
    hybrid = [json.loads(x) for x in (root / f'c5_{preset}' / 'eval.jsonl').read_text().splitlines()]
    for qtype in ('seen', 'paraphrase'):
        for memory in ('closed', 'retrieval'):
            def select(rows, model):
                """Written by us, no upstream reference."""
                return [r for r in rows if r['question_type'] == qtype and r['memory'] == memory and r['model'] == model]
            h = select(hybrid, 'adapter')
            s = select(single, 'adapter')
            b = select(hybrid, 'base')
            assert len(h) == len(s) == len(b) and len(h)
            hc, sc, bc = (sum(r['contains'] for r in x) for x in (h, s, b))
            relation = lambda x, y: 'beats' if x > y else 'ties' if x == y else 'does not beat'
            lines.append(f"- {preset}, {qtype}, {memory}: HybridMem {hc}/{len(h)} {relation(hc, sc)} single LoRA {sc}/{len(s)}; {relation(hc, bc)} base {'retrieval' if memory == 'retrieval' else 'closed-book'} {bc}/{len(b)}.")
summary.write_text('\n'.join(lines) + '\n')
