# MPR: where we are, what Colab can add, and how to tell the story {#mpr-big-picture}

**Our strongest current story is explicit-memory retrieval, not a completed reproduction of the entire paper.** We have working diagnostics and a small QA pilot. Use Colab to strengthen that evidence first; treat implicit memory and HybridMem as optional, untested extensions.

- Covered now: BM25/e5 retrieval, numeric-overlap analysis, and seven QA configurations on nine paired questions.[](source:recall)[](source:qa)[](source:overlap)
- Realistic next target: retrieval sweeps, followed by a matched local QA grid if the available GPU passes a fit check.
- Stretch target: small-model implicit/hybrid pilots. Full paper-scale training and indexing are not the default Colab commitment.
- Decision rule: choose the presentation claim first, then run only experiments that test that claim.

@answer scope="Current saved evidence + proposed Colab work · feasibility conditional on runtime · 10 Nov 2026"
@meta locale=en style=clean
@source recall title="Current retrieval probe: 2,700 questions, user 0" path=results/probes/retrieval_recall.md lines=1:22
@source qa title="Current QA: four NR runs, three SR runs, nine questions" path=results/farm/summary.md lines=1:27
@source overlap title="Current numeric-overlap diagnostic: 32,399 questions" path=results/probes/leak_check.md lines=1:15
@source notes title="Implementation and training caveats" path=docs/upstream-notes.md lines=7:23
@source paper title="Published MPR paper" url=https://doi.org/10.1145/3770854.3780184
@source colab title="Colab FAQ: changing hardware, memory and runtime availability" url=https://research.google.com/colaboratory/faq.html

## 1. The whole paper at a glance {#paper-map span=12}

Use this page to choose scope and presentation flow. Open the [detailed experiment plan](reproduction-audit.html#extension-ledger) for E/N experiment IDs, parent evidence, paper targets and controls.

@tree id=paper-territory title="Paper experiment families and our current coverage"
| paper-root | MPR study | - |
| benchmark | Benchmark: inspected | paper-root |
| explicit | Explicit: partial | paper-root |
| implicit | Implicit: unrun | paper-root |
| hybrid | Hybrid: unrun | paper-root |
| controls | Ablations: unrun | paper-root |
| evidence | Facts + multi-hop QA | benchmark |
| retrieval-pilot | BM25/e5: done | explicit |
| qa-pilot | NR/SR: pilot done | explicit |
| complex-rag | Tree/Graph: unrun | explicit |
| implicit-methods | Ask/Mask SFT | implicit |
| implicit-controls | Oracle + SFT | implicit |
| direct | Direct SFT+RAG | hybrid |
| clustered | HybridMem adapters | hybrid |
| settings | k / steps / size | controls |
| training | Clusters / epochs | controls |
| runtime | Efficiency timing | controls |

**How to read:** “done” describes retained local experiments; “partial” describes coverage of a larger paper family. “Unrun” means no matching retained local output, not a failed method.

The benchmark was used, not regenerated. Historical 45-question and 0.5B notes are not included as retained completed outputs.

## 2. What is covered now, and what could be covered next? {#coverage-horizon span=12}

| Paper area and anchor | Current evidence | Sensible Colab target | Outside the core commitment |
| --- | --- | --- | --- |
| Benchmark; §§3.1–3.4, Tables 1–2 | Release inspected; overlap diagnostic done {good:Evidence exists} | Audit question IDs, gold facts and subset provenance | Reconstructing the dataset generation pipeline |
| Explicit memory; Figure 3 | BM25/e5 probes done; seven adapted QA runs {good:Pilot exists} | E1/E2 retrieval sweeps; E3 same-model NR/SR grid | Claiming the paper's numerical results from a Farm pilot |
| Reasoning; Figures 3 and 5 | NR/SR only; no local MR/DR | N2 SR budget study; N3 MR/DR if time remains | Full four-structure, all-method grid before baseline completion |
| Retrieval count; Figure 4 | Fixed k=20 only | E1 recall sweep; N1 QA sweep after E3 | Calling recall a reproduced QA-accuracy heatmap |
| Model size; Figure 6 | No matched size comparison | Sequential 3B/7B runs only if both fit | Comparing different prompts or quantization and attributing all change to size |
| Implicit memory; Figure 8, Appendix A | No retained training/evaluation {bad:Not covered} | N4 reduced 0.5B–3B LoRA pilot after train/reload smoke | Full 7B training and all implicit ablations |
| Hybrid memory; Table 3, Figures 9–10 | No retained hybrid result {bad:Not covered} | N5 direct hybrid, then few-cluster feasibility study | Paper-scale many-adapter training and cluster/epoch sweeps |
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
| Paper overview | Memory families and the first map on this page; explain HybridMem conceptually | Keep the same literature overview |
| Experimental scope | Completed probes and nine-question QA pilot; state model mismatch | Add E3 manifest and exact local model settings |
| Main evidence | Gold-fact coverage drops with hops; memory helps in the tiny QA slice | Add matched k/step or multi-user comparisons |
| Critical discussion | Numeric overlap is an extra diagnostic, not proven answer leakage | Add E4 within-hop comparisons if group sizes allow |
| Final conclusion | “I tested selected explicit-memory behavior; implicit/hybrid work remains.” | Claim only the new families that actually have complete saved evidence |

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
