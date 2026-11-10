# What have I reproduced, and what can I present? {#mpr-progress}

**You have completed retrieval diagnostics and a seven-configuration QA pilot.** Present these as a reimplementation of selected explicit-memory baselines, plus your own analysis of why retrieval becomes difficult. Implicit memory and HybridMem are the next untested parts, not completed results.

- Finished: BM25 and e5 retrieval on 2,700 questions; a numeric-overlap check on 32,399 questions.[](source:retrieval)[](source:overlap)
- Finished: four Naive Reasoning runs and three Sequential Reasoning runs, all on the same nine questions.[](source:qa)
- Next with Colab only: extend retrieval, then rerun a complete local-model QA cohort if the GPU fit check succeeds.[](source:colab)
- Presentation: explain the task, show your results, compare the trend with the paper, then state what remains untested.

@answer scope="Recorded local experiments · one MPR paper in two PDF versions · audited 10 Nov 2026"
@meta locale=en style=clean
@source retrieval title="Completed retrieval probe: settings and per-hop results" path=results/probes/retrieval_recall.md lines=1:22
@source overlap title="Completed numeric-overlap diagnostic" path=results/probes/leak_check.md lines=1:15
@source qa title="Completed seven-run pilot and sample size" path=results/farm/summary.md lines=1:27
@source scorer title="Local normalized exact-match and contains scorer" path=scripts/score.py lines=19:35
@source probe-code title="Retrieval implementation and extension flags" path=scripts/retrieval_recall.py lines=47:94
@source subset title="Stratified subset generation" path=scripts/make_subset.py lines=18:37
@source runner title="Farm execution wrapper" path=scripts/farm_run.sh lines=1:30
@source setup title="Environment, upstream and dataset setup" path=setup.sh lines=12:68
@source notes title="Recorded upstream training and adapter issues" path=docs/upstream-notes.md lines=7:23
@source historical title="Historical scouting only; not current raw outputs" path=docs/scout-report.html
@source paper title="Published MPR paper: Figures 3–10, Table 3 and Appendix A" url=https://doi.org/10.1145/3770854.3780184
@source preprint title="arXiv v1: same study, with detailed reasoning prompts in Appendix B" url=https://arxiv.org/abs/2508.13250
@source colab title="Google Colab FAQ: hardware and runtime limits are variable" url=https://research.google.com/colaboratory/faq.html

## 1. Exactly what is completed? {#completed span=12}

Start with the [big-picture coverage and decision map](experiment-overview.html). Return here for result charts, experiment IDs, controls and notebook plans.

“Completed” below means an experiment was run and its output is saved. It does not mean every condition matches the paper.

| Completed work | What you actually ran | What you can show now | Output to use |
| --- | --- | --- | --- |
| Done: BM25 retrieval | User 0; 13,100 facts; 2,700 questions; top 20 | Fact recall and complete-evidence coverage by hop | `results/probes/retrieval_recall.md` |
| Done: e5 retrieval | Same corpus/questions; e5-base-v2 without prefixes | Sparse versus dense retrieval behavior | Same saved probe table |
| Done: numeric-overlap check | All 12 users; 32,399 valid questions | A dataset caveat, separate from the paper results | `results/probes/leak_check.md` |
| Done: four Naive QA runs | No memory, gold facts, BM25, e5 | Four baseline scores on nine paired questions | `results/farm/NaiveReason/` |
| Done: three Sequential QA runs | No memory, gold facts, BM25; five calls per question | Whether extra reasoning helped on this slice | `results/farm/SequentialReason/` |
| Working: scoring | Normalize answers, then compare with gold | Recomputed exact-match results; optional lenient score | `scripts/score.py` |

The seven QA files contain 63 answers but only **nine distinct questions**. The retrieval probes cover many more questions than the QA pilot.[](source:qa)[](source:retrieval)

## 2. What did I reimplement from the paper? {#method span=12}

@workflow id=experiment-path title="Your implemented explicit-memory experiment"
| memory | User facts | database |
| retrieval-step | Select evidence | backend |
| prompt | Question + evidence | backend |
| model | Farm LLM | cloud |
| prediction | Predicted answer | backend |
| score-step | Compare with gold | backend |
| memory > retrieval-step | full memory |
| retrieval-step > prompt | top 20 |
| prompt > model | NR or SR |
| model > prediction | response |
| prediction > score-step | local EM |

| Paper term | Plain meaning | Your implementation |
| --- | --- | --- |
| Ignoramus | Answer without user facts | No-memory control; NR and SR finished |
| Oracle | Supply the correct supporting facts | Gold-evidence control; NR and SR finished |
| SparseRAG | Retrieve facts with keyword matching | BM25; NR and SR finished |
| DenseRAG | Retrieve facts with semantic embeddings | e5; NR finished, SR still missing |
| NR | Ask for the answer in one call | Finished for all four controls/methods |
| SR | Generate intermediate thoughts before answering | Five calls; three methods finished |

The evaluation pipeline reuses pinned upstream code with your Farm patch and local scripts. The recorded QA model is not identified precisely; do not call it Qwen2.5-7B.[](source:setup)[](source:runner)[](source:qa)

## 3. Retrieval result: why are long chains difficult? {#retrieval-result span=7}

```mermaid title="Recorded mean gold-fact recall@20 · user 0 · 300 questions per hop"
xychart-beta
  x-axis ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
  y-axis "Recall (%)" 0 --> 100
  line "BM25" [84.7, 43.3, 56.3, 53.6, 49.6, 40.6, 38.5, 37.0, 35.1]
  line "e5" [100.0, 64.3, 57.0, 48.1, 43.2, 36.7, 34.1, 31.0, 25.7]
```

**Read the chart:** each point measures the average fraction of required facts found by one question-only retrieval. It is not answer accuracy.

**Say on your slide:** “Dense retrieval is stronger at short hops in my probe; BM25 is stronger from hop 5 onward. Both lose coverage on long chains.”

**Paper cross-check:** Figure 3 and Section 4.2 describe dense retrieval as stronger on short hops and sparse retrieval on long hops. Your evidence supports that pattern with a different metric.[](source:paper)[](source:retrieval)

**Do not say:** “I matched the paper’s accuracy.” This chart measures retrieval coverage, while the paper measures final-answer exact match.

## 4. Is finding some facts enough? {#complete-evidence span=5}

```mermaid title="Recorded complete-evidence coverage@20 · user 0 · horizontal axis: hops"
xychart-beta
  x-axis ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
  y-axis "All facts found (%)" 0 --> 100
  line "BM25" [84.7, 13.7, 30.7, 17.7, 14.0, 5.7, 4.3, 2.7, 0.3]
  line "e5" [100.0, 39.0, 30.3, 10.0, 6.3, 2.7, 3.7, 1.0, 0.0]
```

**Read the chart:** this stricter measure asks whether all required facts were retrieved together, not merely some of them.

**Say on your slide:** “Partial evidence is common, but complete evidence nearly disappears for long chains. This helps explain why retrieval is a bottleneck.”

At hop 10, a question needs 6.54 unique gold statements on average. A hop is not always a separate fact.[](source:retrieval)

**Next experiment:** change retrieval count using `--k 10`, `--k 30` and `--k 50`. This extends your diagnostic; reproducing Figure 4 also requires final-answer QA at each setting.[](source:probe-code)[](source:paper)

## 5. QA result: did memory and extra reasoning help? {#qa-result span=7}

```mermaid title="Recomputed normalized exact match · same nine questions in every run"
xychart-beta horizontal
  x-axis ["NR: no memory", "NR: gold facts", "NR: BM25", "NR: e5", "SR: no memory", "SR: gold facts", "SR: BM25"]
  y-axis "Correct answers out of 9" 0 --> 9
  bar "Exact match" [1, 8, 7, 6, 1, 7, 6]
```

**Say on your slide:** “On my nine-question pilot, supplying memory helps relative to the no-memory control. Five-step reasoning does not improve exact match in the completed paired runs.”

**Paper cross-check:** Figure 3 reports SR/MR gains over NR/DR. Your pilot does not demonstrate that reasoning gain; it neither confirms nor disproves the paper.[](source:paper)[](source:qa)

**Interpretation:** Oracle checks reasoning when retrieval is perfect. The difference between Oracle and RAG mixes retrieval and answer-generation errors; it is not a pure retrieval-effect estimate.

**Limits:** each changed answer moves a score by 11.1 percentage points: 1 / 9 × 100. One user and one question per hop cannot establish a stable method ranking.

**Missing bar:** Sequential + DenseRAG was not run. Its absence is not a score of zero.

## 6. Dataset caveat: can the question itself reveal information? {#overlap-result span=5}

```mermaid title="Recorded numeric-overlap diagnostic · all 12 users · horizontal axis: hops"
xychart-beta
  x-axis ["2", "3", "4", "5", "6", "7", "8", "9", "10"]
  y-axis "Questions flagged (%)" 0 --> 100
  line "Any shared number" [13, 7, 27, 60, 65, 71, 75, 80, 82]
  line "3+ digit number" [10, 5, 20, 42, 44, 50, 53, 60, 62]
```

**Say on your slide:** “My extra dataset diagnostic finds more numeric overlap at higher hops. Some questions may restate useful values, so memory-free performance needs careful interpretation.”

This regex test detects shared number tokens. It does not establish that the shared value is the answer or that memory is unnecessary. It misses written-out number words.[](source:overlap)

**Relation to the paper:** this is your additional analysis, not a reproduced paper experiment. Label it that way on the slide.

**Next experiment:** partition questions by the overlap flag and compare no-memory versus RAG accuracy within each group. No such QA comparison is currently saved.

## 7. What is still missing from the paper? {#missing span=12}

| Paper experiment | Current state | Concrete work needed |
| --- | --- | --- |
| Explicit-memory baseline grid; Figure 3 | Selected methods executed; paper-scale comparison incomplete {caution:Extend} | Finish SR DenseRAG, enlarge the sample, match Qwen2.5-7B, then add MR/DR |
| Retrieval-count / reasoning-step sweeps; Figures 4–5 | No saved sweeps {bad:Missing} | Hold questions fixed; vary one setting at a time; record final-answer scores |
| TreeRAG / GraphRAG; Section 4 | No local result {bad:Missing} | Build the summary/graph indexes, then evaluate on the same questions |
| Backbone comparison; Figure 6 | No controlled 3B / 7B comparison {bad:Missing} | Use the same dataset, prompts, retrieval and scoring for both models |
| Implicit memory; Figure 8, Appendix A | No current training/evaluation artifacts {bad:Missing} | Train AskSFT and MaskSFT; evaluate standalone and with Oracle; check epochs and prompts |
| Direct SFT+RAG and HybridMem; Table 3 | No current matched evaluation {bad:Missing} | Validate adapters and checkpoint loading; compare hybrid against unchanged RAG controls |
| Hybrid cluster / epoch sweeps; Figures 9–10 | No saved sweep {bad:Missing} | Vary clusters and epochs after one end-to-end HybridMem run works |
| Efficiency; Figure 7 | Token summary only, no matched timing study {caution:Different measure} | Record wall time per question; distinguish index/training cost from inference cost |

Historical notes describe a 45-question grid and 0.5B training probes. Their raw outputs and adapters are not retained here; do not add them to the current completed list.[](source:historical)

The paper’s Table 3 shows mixed HybridMem changes across cells. Test its claims rather than assuming every hybrid must beat RAG.[](source:paper)

## 8. With only Colab, what should I do next? {#extensions span=12}

**Recommended minimum:** E1 retrieval coverage, E2 prefix sensitivity, then E3 local-model QA only if a GPU smoke run succeeds. No Farm credentials are assumed.

No `.ipynb` files are present in this checkout. These are notebook experiment specifications, not claims that your personal Colab notebooks have been inspected.

@workflow id=colab-path title="Colab decision path: plans, not completed experiments"
| inspect | Inspect runtime | backend |
| probes | E1/E2 retrieval | backend |
| gpu | GPU run fits? | backend |
| qa-local | E3 paired QA | backend |
| present-probes | Present probe study | frontend |
| compare | E4/E5 controls | backend |
| inspect > probes | CPU first |
| probes > gpu | measured fit |
| gpu > qa-local | yes |
| gpu > present-probes | no | dashed |
| qa-local > compare | outputs saved |

| Available resource | What to prioritize | What must not be assumed |
| --- | --- | --- |
| CPU notebook only | BM25, overlap analysis, rescoring, charts | No GPU-dependent training plan; dense retrieval may run slowly |
| GPU available, capacity unknown | Measure memory and time on a small inference batch | GPU name, VRAM and session lifetime are not guaranteed |
| GPU passes full-context 7B smoke | Qwen2.5-7B local QA, batch size 1, resumable output | A quantized model is an adaptation; record precision and limits |
| Only a smaller model fits | Keep all matched baselines on the same 0.5B–3B model | Do not label small-model results as reproduced 7B numbers |

These are feasibility judgments, not measured runtime promises. Colab resources vary and runtimes can terminate; save records and checkpoints outside the VM.[](source:colab)

**Existing-code boundary:** retrieval flags and subset generation already exist. `farm_run.sh` needs a paid/external API; it is not local Colab inference. E3 needs a notebook inference adapter that preserves prompts, memory selection and scoring.[](source:probe-code)[](source:subset)[](source:runner)

```python title="Notebook resource inventory; run after selecting the runtime"
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
    print("BF16 supported:", torch.cuda.is_bf16_supported())
```

**Before E3:** run nine questions, one per hop, with the actual top-20 contexts. Measure peak VRAM and elapsed time; record truncation and failures.

**Before training:** complete a forward/backward step, save an adapter, reload it, and generate an answer. Do not schedule an epoch sweep before that works.

For dense retrieval, start with smaller encoding batches than the script's fixed 256 if memory is tight. That change needs a notebook adaptation, not an existing CLI flag.[](source:probe-code)

## 9. How should I present this work? {#slides span=12}

| Slide | Show | Say | Paper anchor |
| --- | --- | --- | --- |
| 1. Problem | A question that needs a chain of personal facts | “The task tests reasoning over personal memory, not generic knowledge.” | Figure 1; Section 3 |
| 2. Reimplementation | The pipeline in panel 2 and the completed-work list | “I reused the upstream evaluation path and implemented the Farm integration, probes and scoring.” | Figure 2 |
| 3. Main observation | Retrieval charts in panels 3–4 | “Longer chains make complete evidence coverage difficult; my probe agrees with the paper’s sparse/dense trend.” | Figure 3; Section 4.2 |
| 4. End-to-end check | The paired QA pilot in panel 5 | “Memory helps on this slice, but my small run does not reproduce the paper’s SR gain.” | Figure 3 |
| 5. Extra insight | Numeric-overlap chart in panel 6 | “This is my own diagnostic and a caution about interpreting high-hop accuracy.” | Additional analysis, not a paper claim |
| 6. Status and next step | Completed list, missing methods, next runnable command | “The explicit-memory pilot is finished. Larger matched evaluation and implicit/hybrid methods remain.” | Figures 4–10; Table 3 |

@callout level=note title="A precise closing sentence for your presentation"
| “I reimplemented and ran selected explicit-memory baselines, measured their retrieval behavior, and checked their trends against the paper. I have not yet reproduced the full implicit-memory or HybridMem comparison.” |

## 10. What must I double-check before presenting? {#checks span=12}

- [ ] Label retrieval recall separately from final-answer accuracy.
- [ ] Say nine unique QA questions, not 63 independent examples.
- [ ] State that the Farm model differs from the paper model and its exact deployment was not saved.
- [ ] Use exact match as the main local score; explain lenient contains only as an extra check.
- [ ] Do not treat the missing SR DenseRAG result as zero.
- [ ] Label numeric overlap as a risk heuristic, not proven answer leakage.
- [ ] Keep historical 45-question and 0.5B claims separate from retained local outputs.
- [ ] Cite the correct paper section beside every comparison.

**Scoring example:** `320 pages` versus gold `320` fails exact match but passes contains. An answer with two alternative dates can also pass contains.[](source:scorer)

**Dataset note:** the paper tables say 108,000 questions while prose says 10,800. The release summaries report 32,400, with one null question skipped. Do not hide that discrepancy.[](source:paper)[](source:notes)

**Version note:** the two PDFs are versions of one paper. Both include Appendix A; arXiv v1 additionally supplies Appendix B’s detailed prompts.[](source:preprint)

**Audit check:** the seven saved JSONL files were rescored for this revision. Their scores match the committed summary. The retrieval and overlap tables are saved outputs, not fresh reruns.[](source:qa)[](source:retrieval)[](source:overlap)

## 11. Extension experiments: what continues my completed work? {#extension-ledger span=12}

**Extension** means a retained result supplies the starting point. All E-series items below are proposed, not completed. The linked parent is evidence, not a promise of numerical agreement.

| ID / priority | Parent result and code | Paper link and comparison | Proposed Colab design | Output and completion criterion |
| --- | --- | --- | --- | --- |
| E1 · first: retrieval count and users | [Completed recall curves](block:retrieval-result); `retrieval_recall.py` flags `--user`, `--k`, `--no-e5` | [Figure 4, §4.3, PDF p.6](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=6): recall/noise motivation, not the same metric | User 0; k=10/20/30/50; full memory; all valid questions. Then users 1–2, one at a time. Start BM25 on CPU; add e5 if practical. | Save recall@k and all-gold@k per hop/user. Compare to original user-0 k=20. Done when all planned cells exist; no QA accuracy claim. |
| E2 · first: e5 prefix sensitivity | [Completed e5 probe](block:retrieval-result); same script, `--e5-prefix` | [§4.1, PDF p.5](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=5): DenseRAG implementation audit; no paper prefix-ablation figure | User 0, k=20; identical model, corpus and questions; prefixes off versus on | Two matched curves and per-hop deltas. Call this an implementation-sensitivity extension, not a paper result reproduction. |
| E3 · next: local QA baseline grid | [Nine-question QA pilot](block:qa-result); subset generator and scorer | [Figure 3, §4.2, PDF pp.5–6](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=5): memory ranking and NR/SR comparison | New local model cohort: 45 questions, five per hop, seed 0; full user-0 memory. Four NR controls first, then four SR controls. Prefer stated 7B model only after fit check. | Save eight raw answer files with IDs, prompts and settings; EM by hop, paired differences. Rerun every cell locally; never merge one local SR DenseRAG cell into the old Farm cohort. |
| E4 · after E3: question-overlap audit | [Completed overlap diagnostic](block:overlap-result) plus E3 answer records | [Task definition §3.1, PDF p.3](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=3) and [Figure 3](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=5); new audit question, not a paper figure reproduction | Join numeric-overlap flags to raw answers by user and question ID. Compare no-memory, Oracle and RAG within hop and overlap strata. | Report group sizes and EM, not only pooled rates. Empty/tiny strata remain inconclusive; expand questions rather than claim causality. |
| E5 · after E3: inference cost | E3 answer records; old Farm `usage.md` is token-only context | [Figure 7(a), §4.6, PDF p.7](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=7): accuracy versus inference time | Same runtime/model/questions; separate loading/index time, retrieval time and generation time. Record generated tokens and actual calls; repeat timings after warm-up. | Per-hop latency and accuracy-cost plot. Compare trends, not A800 seconds versus Colab seconds. Exclude no failed requests silently. |

E1/E2 can reuse the existing CLI after setup. E3 needs local generation code; E4 needs a joined analysis; E5 needs timing instrumentation. Those additions are not already implemented.[](source:probe-code)[](source:runner)

## 12. New experiments: what starts a missing paper family? {#new-ledger span=12}

**New** means no retained matching local result. Reusing E3's harness does not turn a never-run paper ablation into a completed extension.

| ID / priority | What to link in the paper | Proposed experiment and existing dependency | Colab gate | Evidence needed to claim completion |
| --- | --- | --- | --- | --- |
| N1 · after E3: QA retrieval-count sweep | [Figure 4, §4.3, PDF p.6](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=6) | Use E3 fixed questions, BM25/e5 and NR first; k=10/30/50. Use E1 recall as an explanation alongside final-answer EM. | Local generation fit; measure k=50 context memory before scaling | Raw answers and contexts for every k; EM per hop and setting. A recall-only sweep is not N1. |
| N2 · after E3: reasoning budget | [Figure 5, §4.4, PDF p.6](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=6); [arXiv Appendix B prompts](../papers/2508.13250v1.pdf#page=12) | E3 Oracle and BM25, same questions; SR 3 versus 5 calls, with NR as a separate one-call control | More inference time; configurable staged prompts required, not a current wrapper flag | EM and actual calls/tokens; use paper buckets 2–4/5–7/8–10. Do not implement one-step SR by merely relabeling NR. |
| N3 · optional: reasoning structures | [Figure 3 and §3.4.3](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=5); [Appendix B](../papers/2508.13250v1.pdf#page=12) | Add MR and DR to E3 Oracle/BM25 before all methods; preserve branch and divide/solve/merge behavior | New local integration and extra calls; only after NR/SR outputs are complete | Traces proving the intended structure plus answer scores. A longer prompt alone is not MR or DR. |
| N4 · stretch: implicit memory | [Figure 8, §§5.1–5.2, PDF p.7](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=7); [Appendix A, pp.10–11](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=10) | E3 supplies no-memory and Oracle controls. Train AskSFT first on a fixed fact slice, then MaskSFT. Separate held-out fact paraphrases from multi-hop QA. | Smaller 0.5B–3B LoRA pilot; successful train/save/reload smoke; padding and prompt issues resolved | Base and tuned-model results on identical questions, trained facts, data lineage and adapter checkpoints. Small-model/small-memory results are scaled adaptations, not Figure 8's 7B reproduction. |
| N5 · stretch: direct hybrid, then HybridMem | [§§6.1–6.3 and Table 3, PDF pp.8–9](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=9) | After N4, compare unchanged RAG versus SFT+RAG; only then add a few clustered adapters. Keep corpus, questions and retriever fixed across all variants. | Adapter loading/checkpoint bugs resolved; persistent storage; sequential training to fit memory | Raw answers for base RAG, direct hybrid and clustered hybrid; adapter routing and selected checkpoints. Use Table 3 buckets 2–6/7–10, not 2–4/5–7/8–10. |
| N6 · defer: broad ablations and indexes | [Figure 6](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=7), [Figures 9–10](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=8), [§4.1 TreeRAG/GraphRAG](https://dl.acm.org/doi/pdf/10.1145/3770854.3780184#page=6) | 3B/7B comparison, cluster/epoch sweeps, full TreeRAG/GraphRAG indexing | Only after a stable base run; index-building and many adapters are costly in time as well as memory | Matched settings and complete grids; keep reduced-index or few-cluster studies explicitly separate from paper-scale claims |

The recorded upstream notes flag SFT padding, prompt mismatch and adapter/checkpoint issues. Verify those on the pinned code before training; fixing them changes the implementation condition.[](source:notes)

## 13. What should each Colab notebook contain? {#notebook-contract span=12}

These are suggested notebook units, not files already created in this repository. Keep one question per notebook so interrupted runs remain recoverable.

| Suggested notebook | Experiment IDs | Reuse | New notebook work |
| --- | --- | --- | --- |
| 01: dataset and retrieval audit | E1, E2 | Dataset loader, retrieval probe, numeric-overlap script | Runtime setup, loops over settings, persistent per-question retrieval records, chart export |
| 02: local paired QA | E3 | Full-memory subset generator, upstream prompts, scorer | In-process local model adapter, NR/SR integration, JSONL checkpoint/resume; no Farm dependency |
| 03: explain the QA result | E4, E5, N1, N2 | Notebook 02 outputs and fixed IDs | Join overlap flags; time stages; vary k or reasoning calls one factor at a time |
| 04: memory training feasibility | N4, N5 | Frozen question/corpus manifest and base controls | LoRA preparation, fit test, checkpoint reload, held-out evaluations and adapter routing |

**Start small without changing the question:** nine questions for runtime/debug checks; 45 for the first paired pilot; then 90 or more and additional users if the measured runtime allows.

Nine questions are one per hop; 45 are five per hop; 90 are ten per hop. These are proposed workload sizes, not statistical power guarantees.

For a 45-question NR/SR four-memory grid, NR needs 4 × 45 = 180 model calls. Five-call SR needs 4 × 45 × 5 = 900; total 1,080.

This arithmetic counts calls, not tokens or wall time. Estimate time from observed cells before starting the next batch.

**Notebook setup boundary:** do not blindly run the laptop setup over Colab's environment. It defaults to CPU Torch wheels and a separate Python 3.10 venv. Reuse its dataset and upstream pins while keeping a GPU-compatible notebook environment.[](source:setup)

**Resume contract:** append one result per completed question; use `(run_id, user_id, question_id, method, reasoning, setting)` as the identity. Save to Drive or another persistent location.

## 14. What links and evidence must every experiment save? {#audit-contract span=12}

| Required field | Extension example | New-family example |
| --- | --- | --- |
| Experiment ID and status | E1; planned/running/completed | N1; planned/running/completed |
| Paper target and intended claim | Figure 4 / §4.3; retrieval coverage as supporting diagnostic | Figure 4 / §4.3; final-answer accuracy versus k |
| Parent artifact | `results/probes/retrieval_recall.md`, user-0 k=20 | No prior N1 result; E3 supplies the baseline harness and IDs |
| Frozen protocol | Corpus checksum, question IDs, user, hops, retriever version, prefix flag, k | Same plus exact model revision, precision, prompt, generation settings, seed |
| Raw evidence | Gold IDs and retrieved IDs per question, then aggregate recall | Response, gold, retrieved contexts, reasoning trace, score and timing |
| Comparison chart | Original k=20 versus new matched k values; separate new users | EM versus k; separate paper reference from local measurements |
| Interpretation | What changed, what stayed fixed, why this tests the stated claim | Agreement/disagreement/uncertain; sample size and implementation changes |

Record GPU model, software versions, memory limits, context/output limits and truncation. Never silently drop failed or resumed records from denominators.

**Comparison rule:** exact model, sample, scorer and prompt parity permit a closer numeric comparison. Otherwise compare trends and call the run a scaled adaptation.

**Presentation rule:** connect every slide as “paper claim → my experiment → measured result → interpretation → limit.” Cite the figure and the raw artifact together.

**Decision checkpoint:** E1–E3 are the core recommendation; E4/E5 or N1/N2 add explanation. N4/N5 are optional stretch work, not required to present a coherent explicit-memory study.
