# MPR: where we are, what Colab can add, and how to tell the story {#mpr-big-picture}

**Explicit-memory coverage includes a historical complete 45-question NR/SR × four-memory grid, plus a smaller retained seven-cell run.** Earlier 0.5B LoRA and HybridMem pilots also ran. The historical runs survive as summarized results, not raw outputs or adapters in this checkout; none is a paper-scale 7B reproduction.

- Retained now: BM25/e5 retrieval, numeric-overlap analysis, and seven QA configurations on nine paired questions.[](source:recall)[](source:qa)[](source:overlap)
- Historical only: all eight explicit NR/SR cells on 45 questions; 0.5B AskSFT variants, single-hop probes, SFT+RAG and five-cluster HybridMem.[](source:historical)
- Realistic next target: retrieval sweeps and a **retained, same-model** QA grid if Colab GPU fits. The 45-question Farm run already happened; do not present E3 as its first execution.
- Stretch target: recover or rebuild the 0.5B pilots with raw evidence, then expand only if Colab resources permit.

@answer scope="Current saved evidence + proposed Colab work · feasibility conditional on runtime · 10 Nov 2026"
@meta locale=en style=clean
@source recall title="Current retrieval probe: 2,700 questions, user 0" path=results/probes/retrieval_recall.md lines=1:22
@source qa title="Current QA: four NR runs, three SR runs, nine questions" path=results/farm/summary.md lines=1:27
@source overlap title="Current numeric-overlap diagnostic: 32,399 questions" path=results/probes/leak_check.md lines=1:15
@source notes title="Implementation and training caveats" path=docs/upstream-notes.md lines=7:23
@source historical title="Historical 45-question eight-cell QA grid and 0.5B AskSFT/HybridMem pilots" path=docs/scout-report.html
@source paper title="Published MPR paper" url=https://doi.org/10.1145/3770854.3780184
@source colab title="Colab FAQ: changing hardware, memory and runtime availability" url=https://research.google.com/colaboratory/faq.html

## 1. The whole paper at a glance {#paper-map span=12}

Use this page to choose scope and presentation flow. Open the [detailed experiment plan](reproduction-audit.html#extension-ledger) for E/N experiment IDs, parent evidence, paper targets and controls.

@tree id=paper-territory title="Paper experiment families and our current coverage"
| paper-root | MPR study | - |
| benchmark | Benchmark: inspected | paper-root |
| explicit | Explicit: partial | paper-root |
| implicit | Implicit: historical pilot | paper-root |
| hybrid | Hybrid: historical pilot | paper-root |
| controls | Paper sweeps: missing | paper-root |
| evidence | Facts + multi-hop QA | benchmark |
| retrieval-pilot | BM25/e5: done | explicit |
| qa-pilot | NR/SR: 8 historical, 7 retained | explicit |
| complex-rag | Tree/Graph: unrun | explicit |
| implicit-methods | AskSFT 0.5B: ran | implicit |
| implicit-controls | MaskSFT: unrun | implicit |
| direct | SFT+RAG: ran at 0.5B | hybrid |
| clustered | HybridMem 5 blocks: ran | hybrid |
| settings | k / steps / size | controls |
| training | 0.5B pad/lr variants: ran | controls |
| runtime | Efficiency timing | controls |

**How to read:** “retained” means raw local output exists here; “historical” means the run and summary exist in the scout report, but raw outputs or adapters were discarded. “Missing” means a specific paper condition has no matching run.

The historical Farm grid covered Ignoramus, Oracle, SparseRAG and DenseRAG under both NR and SR (45 shared questions). The retained nine-question run covers all NR cells and three SR cells; **SR DenseRAG is missing only from that retained run**. A 0.5B AskSFT/HybridMem pilot separately used 300 facts and 25 multi-hop questions.[](source:historical)[](source:qa)

| Earlier experiment | What the summary actually records | What cannot be re-audited from this checkout |
| --- | --- | --- |
| Farm QA, 45 questions | Eight NR/SR × four-memory cells; Oracle 0.756 NR / 0.800 SR EM; SparseRAG 0.444 / 0.422; DenseRAG 0.311 / 0.333 | Per-question answers, prompts and exact rerun artifacts |
| 0.5B AskSFT, 300 facts | Three training variants; 25-question MPR EM 0.04 / 0.00 / 0.08; raw-format single-hop recall ≤0.11 | Saved adapters, per-question predictions and training files |
| 0.5B hybrid, five clusters | Direct SFT+DenseRAG and HybridMem evaluated; HybridMem 0.00 EM on 25 MPR questions | Adapter files, routing records and raw hybrid answers |

These pilot scores describe **different models and question sets**. Do not put them on the paper's Qwen2.5-7B score axis.[](source:historical)

## 2. What is covered now, and what could be covered next? {#coverage-horizon span=12}

| Paper area and anchor | Current evidence | Sensible Colab target | Outside the core commitment |
| --- | --- | --- | --- |
| Benchmark; §§3.1–3.4, Tables 1–2 | Release inspected; overlap diagnostic done {good:Evidence exists} | Audit question IDs, gold facts and subset provenance | Reconstructing the dataset generation pipeline |
| Explicit memory; Figure 3 | 45-question eight-cell Farm grid **ran earlier** (summary only); seven-cell nine-question run retained {caution:Partial evidence} | E3 saved, matched local-model grid; N3 MR/DR | Claiming paper-equivalent results from a Farm model |
| Reasoning; Figures 3 and 5 | NR/SR only; no local MR/DR | N2 SR budget study; N3 MR/DR if time remains | Full four-structure, all-method grid before baseline completion |
| Retrieval count; Figure 4 | Retained probe at k=20; historical BM25 recall@10/50 for hop 10 only; **no QA k sweep** | E1 full recall sweep; N1 QA sweep after E3 | Calling recall a reproduced QA-accuracy heatmap |
| Model size; Figure 6 | No matched size comparison | Sequential 3B/7B runs only if both fit | Comparing different prompts or quantization and attributing all change to size |
| Implicit memory; Figure 8, Appendix A | 0.5B AskSFT **ran earlier**; summarized in scout report, raw artifacts absent {caution:Historical pilot} | Rebuild the 0.5B probe with saved adapters/outputs, then consider larger LoRA | Full 7B AskSFT/MaskSFT, Oracle variants and ablations remain uncovered |
| Hybrid memory; Table 3, Figures 9–10 | 0.5B SFT+DenseRAG and five-cluster HybridMem **ran earlier**; summarized only {caution:Historical pilot} | Rebuild matched baseline/hybrid results with retained adapters and outputs | Paper-scale 50-cluster HybridMem and its sweeps remain uncovered |
| Efficiency; Figure 7 | Farm tokens/calls, not latency | E5 local runtime and accuracy measurements | Matching A800 timing with a variable Colab runtime |
| TreeRAG / GraphRAG; §4.1 | No indexes or evaluation {bad:Not covered} | Small indexed slice only as a separate optional study | Full-corpus LLM indexing as the first Colab task |

All “next” entries are proposals, not scheduled or completed runs. These categories are not equally weighted; there is no meaningful single completion percentage.

## 3. Which experiments should lead to which others? {#roadmap span=12}

@workflow id=experiment-roadmap title="Recommended order: finish the main evidence path before the stretch branch"
| lane base | Start from retained evidence |
| lane extend | Extend without training |
| lane decide | Optional new families |
| existing | Existing probes | database | base |
| retrieval-more | E1/E2 retrieval | backend | extend |
| local-grid | E3 local QA | backend | extend |
| explain | E4/E5 audit | backend | extend |
| reasoning-new | N1/N2 QA sweeps | backend | decide |
| training-new | N4 training pilot | backend | decide |
| hybrid-new | N5 hybrid pilot | backend | decide |
| existing > retrieval-more | extend |
| retrieval-more > local-grid | GPU fits |
| local-grid > explain | saved answers |
| local-grid > reasoning-new | stable baseline |
| local-grid > training-new | stretch | dashed |
| training-new > hybrid-new | reload works | dashed |

**Core route:** E1/E2 → E3 → one explanatory study from E4/E5 or N1/N2. Stop there if that answers your presentation question.

**Optional routes:** N3 adds MR/DR; N4/N5 require training validation. The arrows show dependencies, not promises that Colab can finish everything.

[Extension designs and parent evidence](reproduction-audit.html#extension-ledger) · [New-family designs and paper targets](reproduction-audit.html#new-ledger)

## 4. What does “only Colab” change in the plan? {#resource-gates span=6}

@steps
| runtime-check | Inspect the actual runtime | Record GPU, VRAM, precision support and persistent storage. No GPU model or session duration is assumed. |
| cpu-route | Keep a CPU-safe route | BM25, saved-output scoring, overlap analysis and plotting still support a useful presentation. |
| inference-check | Test real-context inference | Use nine questions and full top-20 contexts. Measure memory, time, truncation and failures before expanding. |
| training-check | Gate every training plan | Train one step, save, reload and infer before any epoch or cluster sweep. |

**Conditional, not guaranteed:** 4-bit 7B inference may be feasible on a suitable GPU. Context length and generation cache still consume memory; a loaded model is not proof of a runnable experiment.

A smaller model or smaller memory corpus changes the scientific condition. Label that as a scaled study, not a paper-equivalent reproduction.[](source:colab)

No notebooks were supplied in this checkout. The plan does not claim to know what is already executed inside your personal Colab notebooks.

## 5. Pick a presentation scope before spending compute {#presentation-options span=6}

| Story | Evidence needed | When to choose it |
| --- | --- | --- |
| A. Why retrieval struggles | Current probes, QA pilot and clearly labeled overlap diagnostic | You need a defensible talk now {good:Available now} |
| B. Can I reproduce explicit-memory trends? | A plus E1/E2 and completed local E3; preferably N1 or N2 | Best next target for notebook resources {good:Recommended} |
| C. Does learned memory help retrieval? | B plus matched N4/N5 controls and saved adapters | Only after training/reload works {caution:Stretch} |

**A does not require pretending the whole paper is reproduced.** Explain the entire paper in the literature portion, then narrow your experimental claim to the part you tested.

**B remains useful if the trend disagrees.** A controlled disagreement is a result; inspect sample size, prompts, model and retrieval before explaining it away.

**C must show matched controls.** A trained adapter without a base-model and RAG comparison does not test the hybrid claim.

## 6. How should the talk flow? {#talk-flow span=12}

@workflow id=talk-argument title="Presentation logic: explain the paper, then test a bounded claim"
| task | Personal-memory task | frontend |
| claim | Paper claim | backend |
| implementation | My reimplementation | backend |
| observed | Measured result | database |
| interpretation | Explain agreement/gap | backend |
| scope | Limits + next test | frontend |
| task > claim | why it matters |
| claim > implementation | what I test |
| implementation > observed | evidence |
| observed > interpretation | comparison |
| interpretation > scope | decision |

| Talk section | Current version: what to show | Upgrade after Colab work |
| --- | --- | --- |
| Paper overview | Memory families and the coverage map; explain HybridMem conceptually | Keep the same literature overview |
| Experimental scope | Distinguish historical 45-question Farm grid, retained nine-question grid and historical 0.5B training | Add E3 manifest and exact local model settings |
| Main evidence | Retained retrieval curves and QA pilot; cite historical eight-cell summary separately | Add matched k/step or multi-user comparisons |
| Critical discussion | Numeric overlap is an extra diagnostic, not proven answer leakage; 0.5B LoRA recall was low in the historical pilot | Add E4 within-hop comparisons if group sizes allow |
| Final conclusion | “I tested selected explicit methods and earlier small-model LoRA/hybrid pilots; none reproduces the full paper.” | Claim only new conditions that have complete saved evidence |

Link each slide to a paper figure and one raw artifact. Do not mix paper EM, local EM and retrieval recall on a common score axis.

## 7. What decision should I make today? {#decision span=12}

**Choose Story B as the target and keep Story A as the fallback.** Run E1/E2 first. Attempt E3 only after the real-context smoke run succeeds. Add one explanatory experiment rather than an unfinished full grid.

- [ ] Select the paper claim: retrieval bottleneck, reasoning benefit, or hybrid benefit.
- [ ] Choose the matching experiment IDs from the detailed plan.
- [ ] Record the runtime and model before committing to a sample size.
- [ ] Freeze the questions, full memory and scoring across paired comparisons.
- [ ] Save raw outputs outside the Colab VM before adding more configurations.
- [ ] Update “planned” to “completed” only after all chosen cells and their evidence exist.
- [ ] Frame the talk around observed evidence, including disagreements and missing scope.

[Open the detailed report and Colab specifications](reproduction-audit.html#extensions) · [Notebook organization](reproduction-audit.html#notebook-contract) · [Audit evidence checklist](reproduction-audit.html#audit-contract)
