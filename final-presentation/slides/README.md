# Group 7 Presentation Slide Template (MPR — KDD '26)

Adapted 25-slide presentation scaffold for **Group 7** (*Explicit v.s. Implicit Memory: Multi-hop Reasoning Over Personalized Information*, Zhang et al., KDD '26), built on the Group 7 Slidesgo hexagonal geometric theme (`720 × 405 pt`, 16:9 widescreen).

- **GVHD:** Nguyễn Võ Lê Duy
- **Team Members (Nhóm 7):** Nguyễn Xuân Phi · Lê Hải Phong · Trần Nguyên Phong · Phan Văn Phú

## Files

| File | Purpose |
| :--- | :--- |
| `MPR_KDD26_Group7_Template.pptx` | Ready-to-edit 25-slide PowerPoint template (clean Group 7 master/theme, zero leftover clustering slides or media). |
| `MPR_KDD26_Group7_Template.deckcraft` | Matching DeckCraft source presentation for headless CLI or DeckCraft GUI editing. |
| `build_group7_template.py` | Reproducible Python builder generating both `.pptx` and `.deckcraft` from the reference Group 7 `.pptx` template via `deckcraft-cli`. |

## Deck Structure (25 Slides, ~32 min)

- **Slide 1 (`TITLE` + right-side hex cascade):** Title hero, advisor (`Nguyễn Võ Lê Duy`), and all 4 team members.
- **Slides 2–24 (`BLANK_2` with 4-corner hex chrome):**
  - **1 · Agenda (Slide 2):** 5-part presentation roadmap (~32 min).
  - **2 · Problem & Motivation / Definition / Challenge (Slides 3–6):** MPR vs. prior personalization tasks, 3-hop bridge-entity mismatch, Definition 1, and Explicit vs. Implicit dual bottleneck.
  - **3 · Paper Pipeline (Slides 7–9):** Meta-Graph & 12 user graphs, random walks & QA generation, unified EM evaluation pipeline.
  - **4 · Taxonomy (Slides 10–11):** Explicit/Implicit memory baselines and 4 reasoning structures (`NR`, `SR`, `MR`, `DR`).
  - **5 · Paper Results (Slides 12–14):** Explicit results (`SR`/`MR` vs `NR`/`DR`), ablations (Top-$k$, steps, model size), and Implicit SFT results.
  - **6 · Proposed HybridMem (Slides 15–19):** Motivation, offline clustering & per-cluster LoRA, online retrieval & adapter voting, Table 3 comparison, sensitivity & latency.
  - **7 · Reproduction & Audit (Slides 20–23):** Consumer-GPU scope (`S0–S6`), paired reasoning & Pareto results, retrieval crossover & Table 3 audit, prompt number leakage & repo bug fixes.
  - **8 · Conclusion (Slide 24):** Verified claims and 3 practical takeaways.
- **Slide 25 (`TITLE` + right-side hex cascade):** Team contributions table placeholder, advisor/members credit, and Q&A prompt.

Each slide retains its section badge, pacing hint (`~X.X min`), exact short title, `IDEA / NỘI DUNG CẦN TRÌNH BÀY` bullet card, dashed visual/figure placeholder box, and speaker notes.

## Rebuild & Render Verification

```bash
python3 final-presentation/slides/build_group7_template.py \
  --template /path/to/docs-group7-slides-template.pptx

deckcraft-cli info final-presentation/slides/MPR_KDD26_Group7_Template.pptx
deckcraft-cli render final-presentation/slides/MPR_KDD26_Group7_Template.pptx --all /tmp/g7_out
```
