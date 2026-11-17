# 4-Page LaTeX Report Template Guide (`template.tex`)

**Paper Studied:** *Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information* (Zhang et al., KDD '26, pp. 1964–1974)  
**Advisor (GVHD):** Nguyễn Võ Lê Duy  
**Team Members:** Nguyễn Xuân Phi · Lê Hải Phong · Trần Nguyên Phong · Phan Văn Phú

---

## Purpose

[`template.tex`](./template.tex) is a concise, **4-page ACM `sigconf` scaffold** (`\documentclass[sigconf,nonacm,10pt]{acmart}`, no bibliography section, UTF-8 / `T5` Vietnamese diacritic support) aligned with the presentation template flow (`build_template_slides.py`).

Instead of walls of finished prose, each section contains:
- **Space budgets** summing to **4.00 pages** (`0.35 + 0.55 + 0.95 + 1.15 + 0.60 + 0.40 = 4.00p`).
- **`[Key Idea]` bullet prompts** telling members what argument and verified numbers to write.
- **Standalone framed `\figplaceholder` boxes** and **skeleton tables** so `template.tex` compiles cleanly right away (even before images are placed in `figures/`) and visually reserves space.

---

## Section Map & Page Budget

| Section | Budget | Key Content to Fill In |
|---|---|---|
| **Title, Authors & Abstract** | ~0.25p (shared on P1) | Paper/GVHD/Team header + 4-bullet abstract summary |
| **1. Introduction** | ~0.35p | LaMP/PerLTQA gap, explicit vs. implicit tension, reproduction scope |
| **2. Problem Setup & MPR Dataset** | ~0.55p | Definition 1 (2 conditions), 5-step KG pipeline, release count audit (`32,399` valid Qs) + Fig. placeholder |
| **3. Methodology & Proposed `HybridMem`** | ~0.95p | Explicit/Implicit taxonomy, `NR`/`SR`/`MR`/`DR` flows, dedicated `HybridMem` (`BlockSFT`) offline/online design + Fig. placeholder |
| **4. Experiments & Group Reproduction** | ~1.15p | 4.1 Paper findings (Figs. 3–10, Table 3); 4.2 Group `S0`–`S6` reproduction (`s900`/`s306`, Farm, `S4` 0.5B, Table 3 audit, brief code-fix note) + Skeleton Table & Fig. placeholder |
| **5. Independent Discussion & Analysis** | ~0.60p | Why `DR` collapses, why `MR` ties `SR` at $2.4\times$ cost, `53%` numeric prompt leakage (`82%` hop 10), 3 practical agent rules + Fig. placeholder |
| **6. Conclusion & Member Tasks** | ~0.40p | Confirmed vs. refined summary + 4-row editable member task table |

---

## How Members Fill It In & Compile

1. **Upload / Open on Overleaf** (or compile locally with `pdflatex` / `xelatex` / `lualatex` once a TeX Live distribution is installed).
2. **Replace `[Key Idea]` bullets** in each section with tight academic prose matching that section's space budget.
3. **Drop in figures**: Place your exported PDF/PNG charts inside `figures/` and replace each `\figplaceholder{...}` block with `\includegraphics` (sample commands are printed inside each placeholder box).
4. **Verify & finalize tables**: Check Table 1 (`tab:repro_skeleton`) and Table 2 (`tab:members`), remove `[TODO: ...]` markers, and mute/remove the `\guide` / `\budget` helper macros before final export.
5. **Check page count**: Ensure the compiled PDF remains strictly **4 pages** with no bibliography section.
