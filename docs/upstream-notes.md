# Upstream MPR (commit a24953c): gaps and bugs

Found by reading the code and confirmed by runs in the original scout (details with commands and
numbers: `scout-report.html`, section 5). Line numbers refer to the pinned commit. Only what is
needed to run on the LLM Farm is patched (`experiments/01_farm_small_run/farm.patch`); everything else is left as is.

| # | Issue | Where | Patched here |
|---|---|---|---|
| 1 | Local LLM server for Qwen is not shipped (`requests.post(<api_base>/inference/)`) | `utils.py:99`, `config.py:23` | no, farm client added instead |
| 2 | No scorer, no dataset-generation code, no plotting; `run_qa.py` only writes JSONL | `run_qa.py` | scorer: `experiments/_shared/score.py` |
| 3 | API path hard-codes `temperature=0.9`; gpt-5/6 deployments reject it | `utils.py:95` | yes (dropped on the farm path) |
| 4 | SFT pads every prompt to 4096 and appends a 512-token padded answer (4,608 tokens per sample, ~90x wasted compute; answer trained at positions 4096+ but generated at ~50) | `utils.py:159-160` | no (a contiguous-format ablation only lifted 0.5B recall from 0.07 to 0.11) |
| 5 | HybridMem cannot train and evaluate in one run: adapters are all named `default` in training but selected by `set_adapter(str(cluster_id))` at inference (`ValueError: Adapter 1 not found`); `lora_load_index: 0` loads the epoch-1 checkpoint while the paper reports epoch 7; `BlockSFT_default` lacks `distributional_training` | `BlockSFT.py:105,209`, `config.py:293-304,300` | no |
| 6 | AskSFT asks the same prompt twice per statement, so many training pairs are duplicates (600 pairs, 430 unique in the scout slice) | `SelfAskSFT.py:22` | no |
| 7 | e5-base-v2 used without its recommended `query:` / `passage:` prefixes (may handicap DenseRAG; `retrieval_recall.py --e5-prefix` lets you test it) | `DenseRAG.py:23` | no |
| 8 | One null question in the data (`question_answer_8.json`, hop 4) | dataset | skipped by our scripts |
| 9 | Train/test prompt mismatch for implicit memory: SFT trains on the raw question, inference wraps it in the `Naive_Ignoramus` template (0.5B recall of trained facts 0.07-0.11 raw vs 0.00-0.02 in repo format) | `utils.py:145-146`, `BaseMethod.py:76-79` | no |
| 10 | `scikit-learn` is imported (`BlockSFT.py:9`) but missing from `requirements.txt`; `accelerate` (needed by `Trainer`) too | `requirements.txt` | added to our `requirements.txt` |
| 11 | Dataset detail: `message` lists of a question can repeat a statement id; value-oriented hops have no statement, so a "10-hop" question has about 6.5 unique gold statements | dataset | handled in scripts |
| 12 | `run_qa.py` does `os.mkdir(results/<reason>)` and fails if `results/` does not exist | `run_qa.py` | `setup.sh` creates it |

Also: the paper text claims 108,000 QAs (and elsewhere 10,800); the release has 2,700 per user,
32,400 in total (300 per hop, hops 2-10).
