# S4 - Laptop session 4: small LoRA (AskSFT) and HybridMem

Implicit memory at toy scale: a Qwen2.5-0.5B-Instruct model is fine-tuned with LoRA on farm-generated question-answer pairs of 300 user-0 facts, either with one adapter or with HybridMem
(k-means clusters of the facts, one adapter per cluster, adapter chosen by a vote over the retrieved facts). See [`../README.md`](../README.md) for the provenance labels.

## (a) The experiment in the paper

- Section 5 (implicit memory): the model is fine-tuned on the user's statements with LoRA, rank 8 and alpha 32, epochs tuned from 1 to 10 (Section 3.4.2), no retrieval at test time. Baselines MaskSFT and AskSFT (Section 5.1);
  AskSFT rewrites each statement into a question-answer pair. Claim (Section 5.2, **Figure 8**): implicit memory alone performs poorly, "SFT cannot effectively handle large-scale detailed information".
- Section 6 (hybrid memory): HybridMem / BlockSFT (Section 6.2): cluster all statements with K-means, fine-tune one LoRA adapter per cluster, build the dense index, and at inference pick the adapter of the cluster that wins a vote over
  the retrieved statements. Claim (Section 6.3, **Table 3**): HybridMem gives the best overall performance on multi-hop structures, with notable gains on long hops. Section 6.4 / **Figure 9**: 30 versus 50 blocks differ little;
  Section 6.5 / **Figure 10**: best at epoch 7.

What we test, at toy scale: whether a single LoRA stores facts at all (closed-book recall of trained facts) and whether HybridMem helps over plain retrieval and over a single LoRA. Not tested: MaskSFT, the multi-hop benchmark questions,
the 7B model, 30 or 50 blocks, epoch selection (we always use the last epoch).

Differences from the paper and from upstream: Qwen2.5-0.5B-Instruct in fp16 instead of Qwen2.5-7B; 300 facts of user 0 instead of the user's 13,100 statements; 5 clusters instead of 30 or 50; the training pairs are written by a farm LLM
(upstream asks the model being fine-tuned, `SelfAskSFT.py:25`; a 0.5B model cannot follow the two-line format, `docs/reproduction-status.md`); a probe of 60 trained questions and 60 farm paraphrases of them, not benchmark questions; two learning rates; our own
training loop with dynamic padding (upstream pads every sample to 4,608 tokens, `docs/upstream-notes.md` item 4).

## (b) What we ran

- Hardware: a laptop with an RTX A3000 (6 GB), fp16 base model, LoRA weights in fp32 with autocast and a gradient scaler. The sentence encoder (e5-base-v2) runs on the CPU to keep the GPU free.
- Data: 300 user-0 facts (seed 0), 2 self-ask pairs per fact from the farm model -> 600 pairs; probe = 60 pairs (one per distinct fact) as "seen" questions plus 60 farm paraphrases of them; dense e5 top-20 is restricted to the same 300 facts
  ([`results/laptop/sft/summary.md`](../../results/laptop/sft/summary.md)). 10 epochs, LoRA rank 8, two learning-rate presets: "strong" 2e-4 and "upstream" 1e-5.
- Cost of the farm part, from [`farm_usage.json`](../../results/laptop/sft/farm_usage.json): 660 calls, 66,276 input and 22,924 output tokens (89,200 in total), 311.86 s.
- Key numbers, from [`summary.md`](../../results/laptop/sft/summary.md) (contains-score, correct / 60; one small slice, seen and paraphrased questions share facts, so these are descriptive counts, not independent samples):
  strong, single LoRA, closed book: 6/60 seen and 7/60 paraphrased, versus 0/60 and 1/60 for the base model. With retrieval: single LoRA 45/60 seen and 42/60 paraphrased, base model 41/60 and 43/60.
  strong, HybridMem (5 adapters): closed book 3/60 and 3/60; with retrieval 38/60 seen and 36/60 paraphrased, below the base model with retrieval (41/60 and 43/60) and below the single LoRA with retrieval (45/60 and 42/60).
  upstream learning rate: single LoRA closed book 0/60 and 0/60, HybridMem 1/60 and 0/60; with retrieval single LoRA 43/60 and 43/60, HybridMem 37/60 and 37/60.
- Time, from the same file: embedding and clustering 6.83 s; training 241.08 s (single, strong), 262.26 s (5 adapters, strong), 269.13 s (single, upstream), 267.18 s (5 adapters, upstream).
- Also on a T4 (session `sft_smoke`, 2026-11-01): only a smoke test (2 epochs, 40 facts, 5 clusters, pairs written by the 7B vLLM server) to check the code path; its outputs are in `results/t4/sft/sft_smoke/` and mean nothing about quality
  (the first smoke attempts failed on real problems, listed in [`timing_log.md`](../../results/t4/timing_log.md)). The full T4 version `session4.sh` was **not run**; the laptop run replaced it.

## (c) High-level code walkthrough

1. `laptop_farm_pairs.py`: pick 300 fact ids of user 0 (seed 0) -> ask the farm model twice per fact with upstream's `SelfAskSFT_Ask_Prompt_Template` -> parse the two-line answers into pairs (`trainset.jsonl`) -> take one pair per distinct fact (60) and ask
   the farm model to paraphrase the question (`probe.jsonl`); every response is persisted with its token usage, with a 95,000-token budget stop, and the run can be resumed.
2. `laptop_sft.py`: embed the 300 facts and the 120 probe questions with e5 on the CPU -> top-20 facts per question inside the slice -> k-means (5 clusters, `random_state=42`, `n_init=10`) on the fact embeddings -> a vote of the cluster labels of the
   20 retrieved facts picks the adapter -> for each learning-rate preset and each of 1 adapter / 5 adapters: train the adapter(s) for 10 epochs (`sft.train_adapter`), reload them from disk under distinct names, and evaluate four conditions
   (closed book or retrieval, base model or adapter) on seen and paraphrased questions -> `summary.md`, `eval.jsonl`, `timing.json`.
3. `laptop_sft_compare.py`: appends the descriptive "HybridMem versus single LoRA versus base" comparison lines to `summary.md`.
4. `sft.py` is the standalone trainer behind the laptop scripts (model loading, generation, `train_adapter`, `parse_pair`) and also the T4 entry point (`--stage ask` with the vLLM server, `--stage train` after stopping it); `session4.sh` is its T4 driver.

| File | What it does |
|---|---|
| `laptop_farm_pairs.py` | farm-generated training pairs and paraphrased probe questions |
| `laptop_sft.py` | clustering, vote, training, four-condition evaluation, summary |
| `laptop_sft_compare.py` | paired descriptive comparison lines appended to the summary |
| `sft.py` | LoRA trainer with dynamic padding, HybridMem adapter handling, T4 two-stage entry point |
| `session4.sh` | T4 driver for `sft.py` (smoke test ran, full run did not) |

## (d) Provenance

| Script / function | Label | Reference |
|---|---|---|
| `sft.py: parse_pair` | ADAPTED | upstream `baselines/SelfAskSFT.py:26-34` (split the model output on newlines, keep it only if exactly two lines: question, answer). Ours also drops empty lines and strips leading `Question:` / `Answer:`. |
| `sft.py: make_trainset` | ADAPTED | upstream `SelfAskSFT.py:17-40` (`__generate_trainset__`: `ask_proportion` pairs per statement, at most `10 * ask_proportion` tries, the same `SelfAskSFT_Ask_Prompt_Template` from `prompts.py:328-335`). Ours batches the calls, resumes from a jsonl file, and takes the text from the 7B vLLM server instead of the model being fine-tuned. |
| `sft.py: ask_generate` | ORIGINAL TO THIS REPO | no upstream reference (upstream uses `llm_fast_inference` on the fine-tuned model, `SelfAskSFT.py:25`). |
| `sft.py: load_model`, `gen` | ADAPTED | upstream `baselines/BaseMethod.py:56-60` (`AutoModelForCausalLM.from_pretrained` on cuda, bf16 there, fp16 here) and `utils.py:121-139` (`llm_inference`: chat template, `generate`, decode only new tokens; 512 new tokens there, 64 here). Batching with left padding and the sampling option are ours. |
| `sft.py: encode_sft` | ADAPTED | upstream `utils.py:144-172` (`map_tokenizer_sft_batch`: chat-template prompt, answer plus EOS, labels -100 on the prompt). Ours pads dynamically per batch instead of to 4,096 + 512 tokens (`utils.py:159-160`, upstream-notes item 4). |
| `sft.py: train_adapter` | ADAPTED | LoRA config from upstream `SelfAskSFT.py:56-62` and `BlockSFT.py:97-103` (`r=8`, `lora_alpha=32`, `lora_dropout=0.1`, causal LM), 10 epochs (`SelfAskSFT.py:68`), learning rate 1e-5 for the "upstream" preset (`SelfAskSFT.py:71`). The training loop is ours (AdamW with linear decay, fp16 autocast and gradient scaler, per-epoch checkpoints, resume) in place of upstream's Hugging Face `Trainer` with batch size 1 and 16 accumulation steps (`SelfAskSFT.py:66-73`). The "strong" 2e-4 rate has no upstream reference. |
| `sft.py: main` (k-means blocks, adapter vote, evaluation) | ADAPTED | upstream `baselines/BlockSFT.py:146-158` (e5 embeddings, `KMeans(n_clusters, random_state=42, n_init=10)`, one block per label, pickled clusters), `BlockSFT.py:195-209` (predict the cluster of each retrieved reference, vote, `set_adapter`), adapter loading by name `BlockSFT.py:35-45`. Ours trains and evaluates in one run with distinct adapter names, which fixes upstream's `Adapter not found` crash (upstream-notes item 5). The evaluation conditions are ours (no upstream reference). |
| `sft.py: build_slice` | ORIGINAL TO THIS REPO | no upstream reference (upstream trains on all of the user's statements). |
| `laptop_farm_pairs.py` | ORIGINAL TO THIS REPO | no upstream reference; uses upstream's ask prompt (loaded from the clone) and `sft.parse_pair`. The farm client call follows the same Azure pattern as our `farm.patch` (itself ours). |
| `laptop_sft.py` (setup, vote, `evaluate`, summary writer) | ORIGINAL TO THIS REPO | no upstream reference for the orchestration and the seen/paraphrase probe. The clustering settings (`random_state=42`, `n_init=10`) and the vote follow `BlockSFT.py:151, 195-206`; the vote's tie-break (lowest cluster id) is ours. Prompts are upstream's `Naive_RAG_Prompt_Template` / `Naive_Ignoramus_Prompt_Template`, loaded at run time. |
| `laptop_sft_compare.py` | ORIGINAL TO THIS REPO | no upstream reference. |
| `session4.sh` | ORIGINAL TO THIS REPO | no upstream reference (driver script). |
| shared `../_shared/lib.py`, `score.py` | see [`../_shared/README.md`](../_shared/README.md) (d) | |

## (e) How to run

Laptop (needs a CUDA GPU with about 6 GB; uses `uv`):

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu118 ./setup.sh        # venv, pinned upstream clone, dataset; use a CUDA torch wheel
export T4_ROOT=$PWD MPR_QA_DIR=$PWD/data/unz/mpr_dataset/question_answer LAPTOP_SFT_WORK=$(mktemp -d)   # a fresh scratch dir
# 1. farm pairs: spends about 89k farm tokens; credentials come from the environment and are never written (see env.example)
.venv/bin/python experiments/s4_laptop_lora_hybridmem/laptop_farm_pairs.py
# 2. train and evaluate the four conditions (replaces results/laptop/sft/*; keep a copy of the committed results first)
.venv/bin/python experiments/s4_laptop_lora_hybridmem/laptop_sft.py
```

Expected runtime on the RTX A3000: farm generation about 5 minutes, then four trainings of about 4-4.5 minutes each plus short evaluations (times in `summary.md`).
T4 variant: `experiments/s4_laptop_lora_hybridmem/session4.sh` (needs the `colab` CLI; only its smoke test ran). No notebook: the run needs farm credentials and a local GPU, and a notebook would add nothing over the scripts.

Verified at restructuring: `py_compile` of all four scripts and `bash -n session4.sh`; `sft.py --help` runs (imports of `lib` and `score` resolve through `../_shared`). The laptop scripts import torch, peft and scikit-learn at module level and need a CUDA GPU
and farm credentials, so they were **not executed** in the restructuring task; their path handling (`../_shared`, the `runpy` call to `laptop_sft_compare.py`) was changed and is covered only by the compile check and by reading it.

## (f) Results

Files: [`results/laptop/sft/`](../../results/laptop/sft/) (`summary.md`, `trainset.jsonl`, `probe.jsonl`, `slice.json`, `clusters.json`, per-run `eval.jsonl` and `timing.json`, `farm_usage.json`), and the T4 smoke test in [`results/t4/sft/sft_smoke/`](../../results/t4/sft/sft_smoke/).

Outcome: a strong single LoRA stores a little (closed-book 6/60 and 7/60 against 0/60 and 1/60 for the base model), in line with the paper's "implicit memory alone performs poorly", and does not clearly help with retrieval. HybridMem with 5 clusters scored below plain retrieval and
below the single LoRA with retrieval at both learning rates on this slice, so the paper's Table 3 claim that HybridMem is best is **not reproduced** here; this is one small slice with a 0.5B model and 5 clusters, so it neither confirms nor refutes the paper.
