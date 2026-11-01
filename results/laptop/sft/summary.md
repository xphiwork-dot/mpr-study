# Laptop session 4

Qwen2.5-0.5B-Instruct fp16; 300 user-0 facts; farm AskSFT pairs; 10 epochs; LoRA rank 8. Dense e5 top-20 is restricted to the same fact slice. Hybrid adapter selection uses a vote over retrieved facts even for closed-book answers; only retrieval conditions expose references to the model.

Contains uses the repository whole-word scorer; counts are correct / evaluated. Seen and paraphrased questions share facts, so these are not independent samples. Single small slice; differences are descriptive, not a general superiority claim.

| Preset | Adapters | Question | Memory | Model | Contains | EM |
|---|---:|---|---|---|---:|---:|
| strong | 1 | seen | closed | base | 0/60 | 0/60 |
| strong | 1 | seen | closed | adapter | 6/60 | 5/60 |
| strong | 1 | seen | retrieval | base | 41/60 | 39/60 |
| strong | 1 | seen | retrieval | adapter | 45/60 | 42/60 |
| strong | 1 | paraphrase | closed | base | 1/60 | 0/60 |
| strong | 1 | paraphrase | closed | adapter | 7/60 | 6/60 |
| strong | 1 | paraphrase | retrieval | base | 43/60 | 41/60 |
| strong | 1 | paraphrase | retrieval | adapter | 42/60 | 41/60 |
| strong | 5 | seen | closed | base | 0/60 | 0/60 |
| strong | 5 | seen | closed | adapter | 3/60 | 3/60 |
| strong | 5 | seen | retrieval | base | 41/60 | 39/60 |
| strong | 5 | seen | retrieval | adapter | 38/60 | 37/60 |
| strong | 5 | paraphrase | closed | base | 1/60 | 0/60 |
| strong | 5 | paraphrase | closed | adapter | 3/60 | 2/60 |
| strong | 5 | paraphrase | retrieval | base | 43/60 | 41/60 |
| strong | 5 | paraphrase | retrieval | adapter | 36/60 | 35/60 |
| upstream | 1 | seen | closed | base | 0/60 | 0/60 |
| upstream | 1 | seen | closed | adapter | 0/60 | 0/60 |
| upstream | 1 | seen | retrieval | base | 41/60 | 39/60 |
| upstream | 1 | seen | retrieval | adapter | 43/60 | 43/60 |
| upstream | 1 | paraphrase | closed | base | 1/60 | 0/60 |
| upstream | 1 | paraphrase | closed | adapter | 0/60 | 0/60 |
| upstream | 1 | paraphrase | retrieval | base | 43/60 | 41/60 |
| upstream | 1 | paraphrase | retrieval | adapter | 43/60 | 42/60 |
| upstream | 5 | seen | closed | base | 0/60 | 0/60 |
| upstream | 5 | seen | closed | adapter | 1/60 | 1/60 |
| upstream | 5 | seen | retrieval | base | 41/60 | 39/60 |
| upstream | 5 | seen | retrieval | adapter | 37/60 | 37/60 |
| upstream | 5 | paraphrase | closed | base | 1/60 | 0/60 |
| upstream | 5 | paraphrase | closed | adapter | 0/60 | 0/60 |
| upstream | 5 | paraphrase | retrieval | base | 43/60 | 41/60 |
| upstream | 5 | paraphrase | retrieval | adapter | 37/60 | 35/60 |

## Wall time (seconds)

Embedding and clustering: 6.83. Farm generation: 311.86.

- strong, 1 adapter(s): training_seconds=241.08, load_seconds=2.23, base_closed_seconds=47.43, adapter_closed_seconds=20.69, base_retrieval_seconds=12.78, adapter_retrieval_seconds=17.0
- strong, 5 adapter(s): training_seconds=262.26, load_seconds=1.18, base_closed_seconds=48.57, adapter_closed_seconds=23.1, base_retrieval_seconds=12.68, adapter_retrieval_seconds=18.83
- upstream, 1 adapter(s): training_seconds=269.13, load_seconds=1.25, base_closed_seconds=45.66, adapter_closed_seconds=21.02, base_retrieval_seconds=12.09, adapter_retrieval_seconds=15.65
- upstream, 5 adapter(s): training_seconds=267.18, load_seconds=1.32, base_closed_seconds=51.57, adapter_closed_seconds=18.12, base_retrieval_seconds=12.35, adapter_retrieval_seconds=15.82

## Hybrid comparisons

Counts use the same probe questions; these descriptive comparisons do not establish a general advantage.

- strong, seen, closed: HybridMem 3/60 does not beat single LoRA 6/60; beats base closed-book 0/60.
- strong, seen, retrieval: HybridMem 38/60 does not beat single LoRA 45/60; does not beat base retrieval 41/60.
- strong, paraphrase, closed: HybridMem 3/60 does not beat single LoRA 7/60; beats base closed-book 1/60.
- strong, paraphrase, retrieval: HybridMem 36/60 does not beat single LoRA 42/60; does not beat base retrieval 43/60.
- upstream, seen, closed: HybridMem 1/60 beats single LoRA 0/60; beats base closed-book 0/60.
- upstream, seen, retrieval: HybridMem 37/60 does not beat single LoRA 43/60; does not beat base retrieval 41/60.
- upstream, paraphrase, closed: HybridMem 0/60 ties single LoRA 0/60; does not beat base closed-book 1/60.
- upstream, paraphrase, retrieval: HybridMem 37/60 does not beat single LoRA 43/60; does not beat base retrieval 43/60.
