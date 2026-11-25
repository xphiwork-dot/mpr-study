#!/usr/bin/env python3
"""Adapt the Group 7 Slidesgo PowerPoint template into a clean 25-slide
KDD '26 MPR presentation scaffold (MPR_KDD26_Group7_Template.pptx / .deckcraft).

Preserves the Group 7 master, theme, layouts, and hexagonal decorative chrome
on a 720x405 pt widescreen canvas while removing all old Hierarchical Clustering
slides, text, and unused images. Populates all 25 slides from SLIDES_SPEC AS-IS.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

SLIDES_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = Path(
    "/home/xphi/.local/state/herdr/mpr-study/assets/docs-group7-slides-template.pptx"
)
OUT_PPTX = SLIDES_DIR / "MPR_KDD26_Group7_Template.pptx"
OUT_DECKCRAFT = SLIDES_DIR / "MPR_KDD26_Group7_Template.deckcraft"

FONT = "Arial"
CHARCOAL = "#384655"
MUTED = "#697684"
TEAL = "#348BAA"
WHITE = "#FFFFFF"
LIGHT_BG = "#F8FAFC"
DIVIDER = "#DCE7F0"
FOOTER_DIVIDER = "#E2E8F0"
BORDER = "#CBD5E1"

NAVY = "#1E3A5F"
BLUE = "#2563EB"
BLUE_BG = "#EFF7FD"
PURPLE = "#7C3AED"
PURPLE_BG = "#F5F3FF"
GREEN = "#16A34A"
GREEN_BG = "#F0FDF4"
AMBER = "#D97706"
AMBER_BG = "#FFFBEB"

BG_BY_ACCENT = {
    NAVY: BLUE_BG,
    BLUE: BLUE_BG,
    PURPLE: PURPLE_BG,
    GREEN: GREEN_BG,
    AMBER: AMBER_BG,
}

GVHD = "Nguyễn Võ Lê Duy"
TEAM_MEMBERS = [
    "Nguyễn Xuân Phi",
    "Lê Hải Phong",
    "Trần Nguyên Phong",
    "Phan Văn Phú",
]

# Original 53-slide template has 53 slides (indices 0..52) and 4 text shapes on Slide 0.
OLD_SLIDE_COUNT = 53
OLD_TITLE_TEXT_IDS = [66, 67, 68, 69]

SLIDES_SPEC = [
    # (section_tag, color, title, idea_lines, visual_slot_hint, minutes)
    (
        "1 · Title Slide",
        NAVY,
        "Explicit v.s. Implicit Memory:\nMulti-hop Reasoning Over Personalized Information",
        [
            "Giới thiệu tên bài báo (Zhang et al., KDD '26), GVHD: Nguyễn Võ Lê Duy, và 4 thành viên nhóm.",
            "Nêu mục tiêu buổi trình bày: phân tích luồng nghiên cứu, phương pháp HybridMem, và kết quả tái lập.",
        ],
        "TITLE HERO: Tên đề tài + Logo trường/môn học + GVHD & 4 thành viên",
        "0.5 min",
    ),
    (
        "1 · Agenda",
        BLUE,
        "Presentation Outline & Story Flow",
        [
            "Trình bày lộ trình 5 phần (~32 phút): (1) Problem Setup, (2) Dataset & Evaluation, (3) Memory & Reasoning Taxonomy, (4) Proposed HybridMem & Paper Results, (5) Reproduction & Key Findings.",
        ],
        "TIMELINE / 5-STEP ROADMAP: Phân bổ thời lượng từng phần",
        "0.5 min",
    ),
    (
        "2 · Problem & Motivation",
        BLUE,
        "Why Personalization Needs Multi-Hop Reasoning",
        [
            "So sánh tác vụ cũ (LaMP: khớp style; PerLTQA/LongMemEval: hỏi-đáp 1-hop) với nhu cầu thực tế.",
            "Nhấn mạnh 2 khoảng trống: không khớp phân phối train/test và cần tổng hợp nhiều fact cá nhân.",
        ],
        "COMPARISON TABLE / PAPER FIG. 1(b): Previous Personalization Tasks vs. MPR",
        "1.5 min",
    ),
    (
        "2 · Problem & Motivation",
        BLUE,
        "Illustrative 3-Hop Example: Bridge-Entity Mismatch",
        [
            "Dẫn ví dụ chuỗi Alice -> Bob -> David -> New York (Paper Fig. 1a) cho câu hỏi về nơi làm việc của chồng Alice.",
            "Giải thích trực giác: fact cuối ('David works in New York') không hề trùng từ khóa với câu hỏi gốc.",
        ],
        "DIAGRAM (Paper Fig. 1a): Chuỗi 3 hop + User Statement pool",
        "1.5 min",
    ),
    (
        "2 · Problem Definition",
        BLUE,
        "Formal MPR Task Definition (Definition 1)",
        [
            "Định nghĩa tập fact S = {s_1..s_n}, câu hỏi q, dự đoán a_hat = f(q; S) đo bằng Exact Match (EM).",
            "Nêu rõ 2 điều kiện bắt buộc: (1) Chỉ suy ra được từ S cá nhân; (2) Không thể giải chỉ từ 1 fact đơn lẻ s_i.",
        ],
        "EQUATION / 2-CONDITION CARD: Def. 1 + Điều kiện (1) & (2)",
        "1.0 min",
    ),
    (
        "2 · Core Challenge",
        BLUE,
        "Dual Bottleneck: Explicit Retrieval vs. Implicit Weights",
        [
            "Explicit Memory (RAG): Không thể nhồi toàn bộ S vào prompt; single-shot retrieval miss các hop sau.",
            "Implicit Memory (SFT/LoRA): Khó nén hàng chục nghìn fact chính xác vào trọng số & dễ suy giảm suy luận.",
        ],
        "SIDE-BY-SIDE FORMULA BOX: Explicit g(q || S^; theta) vs. Implicit g(q; theta_hat)",
        "1.0 min",
    ),
    (
        "3 · Paper Pipeline (Dataset)",
        BLUE,
        "MPR Dataset Pipeline I: Meta-Graph & User Instantiation",
        [
            "Giải thích bước 1–2 (Paper Fig. 2a): Thiết kế Meta-Graph G (108 entity/attr, 34 relations: entity, attribute, value).",
            "Sinh 12 đồ thị người dùng riêng biệt G~ (~5,902 nodes/user) để đảm bảo đa dạng và không trùng tri thức công khai.",
        ],
        "FLOWCHART (Paper Fig. 2a Steps 1–2): Meta-Graph -> Specific User Graphs",
        "1.5 min",
    ),
    (
        "3 · Paper Pipeline (Dataset)",
        BLUE,
        "MPR Dataset Pipeline II: Random Walks & QA Generation",
        [
            "Giải thích bước 3–5: LLM viết edge thành ~13.1k statements S -> lấy mẫu đường đi k = 2..10 hops có khử nhập nhằng -> viết lại thành câu hỏi q.",
            "Chiếu thống kê Table 2 & so sánh Table 1 (12 users, 2–10 hops, ưu thế vượt trội so với HotpotQA/PerLTQA).",
        ],
        "FLOWCHART + TABLE (Paper Fig. 2a Steps 3–5 & Table 1/2): Stats của MPR dataset",
        "1.5 min",
    ),
    (
        "3 · Paper Pipeline (Eval)",
        BLUE,
        "Unified Evaluation Pipeline",
        [
            "Trình bày luồng đánh giá thống nhất (Paper Fig. 2b): Store/Index hoặc SFT trên S_i -> Suy luận đa bước trên q -> So sánh Ground-truth (EM).",
            "Nêu cấu hình gốc của paper: Qwen2.5-7B trên 8x A800-80G, subset 2,700 câu hỏi (2–10 hops).",
        ],
        "PIPELINE DIAGRAM (Paper Fig. 2b): Store -> Perceive -> Reason -> Evaluate EM",
        "1.0 min",
    ),
    (
        "4 · Memory Mechanisms",
        PURPLE,
        "Taxonomy of Memory Baselines (Explicit vs. Implicit)",
        [
            "Nhóm Explicit (k=20): SparseRAG (BM25s), DenseRAG (e5), TreeRAG (MemTree), GraphRAG + 2 cận Ignoramus / Oracle.",
            "Nhóm Implicit (LoRA r=8, alpha=32): MaskSFT (điền thực thể bị che) và AskSFT (sinh cặp QA tự hỏi-đáp).",
        ],
        "TAXONOMY DIAGRAM (Paper Fig. 2c): Explicit RAG vs. Implicit LoRA SFT",
        "1.5 min",
    ),
    (
        "4 · Reasoning Structures",
        PURPLE,
        "Four Multi-Hop Reasoning Structures (NR, SR, MR, DR)",
        [
            "Minh họa luồng chạy của 4 cấu trúc (Paper Fig. 2d): NR (1 bước), SR (CoT chuỗi + truy xuất lại sau mỗi bước).",
            "MR (ToT rẽ 2 nhánh + chọn nhánh tốt nhất ở mỗi bước) và DR (chia sub-questions -> giải tuần tự -> gộp kết quả).",
        ],
        "4-PANEL FLOWCHART (Paper Fig. 2d / Fig. 1 repo): NR vs. SR vs. MR vs. DR",
        "2.0 min",
    ),
    (
        "5 · Paper Results (Explicit)",
        GREEN,
        "Explicit Memory Results: SR & MR Dominate NR & DR",
        [
            "Trình bày Paper Fig. 3: SR và MR vượt trội NR/DR (+10–20%); DR yếu hơn SR do chia nhỏ mù quáng từ câu hỏi gốc.",
            "Nêu đối lập DenseRAG (mạnh hop ngắn nhờ ngữ nghĩa) vs SparseRAG (mạnh hop dài nhờ từ khóa), và GraphRAG kém do nhiễu thực thể.",
        ],
        "CHART (Paper Fig. 3): 4 đường EM theo hop (2..10) trên NR / SR / MR / DR",
        "1.5 min",
    ),
    (
        "5 · Paper Results (Ablations)",
        GREEN,
        "Impact of Top-k, Reasoning Steps, and Backbone Size",
        [
            "Top-k (Fig. 4): hop ngắn lợi khi tăng k, hop dài đạt đỉnh ở k trung bình rồi giảm vì nhiễu.",
            "Steps & Size (Fig. 5–6): tăng bước giúp đến mốc 3 steps rồi bão hòa; model 3B tụt mạnh nhất trên NR/DR.",
        ],
        "3-PANEL CHARTS (Paper Fig. 4, Fig. 5, Fig. 6): Heatmaps & Step/Size curves",
        "1.5 min",
    ),
    (
        "5 · Paper Results (Implicit)",
        GREEN,
        "Implicit Memory Results: SFT Alone Fails & Hurts Reasoning",
        [
            "Trình bày Paper Fig. 8 & Appendix A: MaskSFT/AskSFT thuần gần 0% dù tăng epoch hay tăng bước suy luận.",
            "Nhấn mạnh phát hiện nghịch lý: ghép SFT vào Oracle (MSO/ASO) lại làm giảm độ chính xác trên SR/MR vì SFT làm mòn khả năng suy luận.",
        ],
        "CHART (Paper Fig. 8): Explicit Oracle vs. MaskSFT / AskSFT / MSO / ASO",
        "1.5 min",
    ),
    (
        "6 · Proposed Method",
        PURPLE,
        "Why Naive Hybrid (SFT + RAG) Fails — Motivation for HybridMem",
        [
            "Phân tích hạn chế của cách ghép trực tiếp (MS+X / AS+X): xung đột tri thức khi nhồi toàn bộ S vào 1 LoRA & suy giảm reasoning.",
            "Nêu giả thuyết cốt lõi của nhóm tác giả: thông tin người dùng mang tính cụm cục bộ -> cần phân cụm tri thức.",
        ],
        "PROBLEM-TO-INSIGHT CARD: 3 nhược điểm của SFT+RAG trực tiếp -> Ý tưởng BlockSFT",
        "1.5 min",
    ),
    (
        "6 · Proposed Method",
        PURPLE,
        "HybridMem (BlockSFT) I: Offline Clustering & Per-Cluster LoRA",
        [
            "Trình bày pha huấn luyện: Dùng Dense Embedding (e5) biểu diễn từng fact -> phân cụm K-means thành C cụm ngữ cảnh.",
            "Train độc lập 1 LoRA adapter cho mỗi cụm song song với việc dựng chỉ mục tìm kiếm văn bản.",
        ],
        "ARCHITECTURE LANE 1 (Offline): User Facts S -> e5 + K-means -> C LoRAs + Text Index",
        "1.5 min",
    ),
    (
        "6 · Proposed Method",
        PURPLE,
        "HybridMem (BlockSFT) II: Online Retrieval & Adapter Voting",
        [
            "Trình bày pha suy luận: Truy xuất top-k statements S^ bằng RAG theo câu hỏi/trạng thái hiện tại.",
            "Dùng cơ chế bỏ phiếu (majority vote) trên tập S^ để chọn đúng LoRA adapter c* khớp nhất rồi sinh câu trả lời.",
        ],
        "ARCHITECTURE LANE 2 (Online): Query -> Top-k S^ -> Majority Vote c* -> Load LoRA_c*",
        "1.5 min",
    ),
    (
        "6 · Proposed Results",
        GREEN,
        "Paper Table 3: HybridMem vs. Explicit & Naive Hybrids",
        [
            "Chiếu kết quả Table 3 (Short 2–6 vs Long 7–10 hops): chỉ ra điểm sáng HybridMem trên SR SparseRAG hop dài (0.200 -> 0.232).",
            "So sánh đối xứng với MS+X / AS+X bị tụt điểm mạnh trên SR và MR.",
        ],
        "TABLE (Paper Table 3): So sánh X, MS+X, AS+X, và HybridMem trên 4 cấu trúc",
        "1.5 min",
    ),
    (
        "6 · Proposed Analysis",
        GREEN,
        "HybridMem Sensitivity: Cluster Count, Epochs & Latency",
        [
            "Cluster count (Fig. 9): ổn định giữa 30 và 50 cụm (50 nhỉnh hơn nhẹ). Epochs (Fig. 10): đạt đỉnh ở epoch 7 rồi quá khớp.",
            "Efficiency (Fig. 7): trả giá bằng thời gian phân cụm và nạp adapter khi chuyển ngữ cảnh.",
        ],
        "CHARTS (Paper Fig. 7, Fig. 9, Fig. 10): Cluster bars + Epoch heatmap + Radar latency",
        "1.0 min",
    ),
    (
        "7 · Reproduction Summary",
        AMBER,
        "Our Reproduction Scope on Consumer GPUs (S0–S6)",
        [
            "Giới thiệu ngắn thiết lập tái lập sạch: Qwen2.5-7B-AWQ (Colab T4, slice 900/306 Qs, paired CI) + 0.5B Laptop S4 + Farm sanity.",
            "Giải thích vì sao dùng chênh lệch ghép cặp (paired delta EM) thay vì so thẳng số tuyệt đối với 8x A800.",
        ],
        "COMPACT MATRIX: 6 sessions S1–S6 + Offline Probes (32,399 Qs)",
        "1.0 min",
    ),
    (
        "7 · Reproduction Results",
        AMBER,
        "Reproduction I: Explicit Reasoning Ordering & 3-Step Pareto",
        [
            "Xác nhận kết luận lớn của paper: SR (+7.8–11.8 pp) & MR vượt NR; DR tụt ngang NR; 3-step SR là điểm tối ưu chi phí-độ chính xác.",
            "Giải thích gọn vì sao MR tốn 2.4x call mà hòa SR (buffer nông & trùng nhánh) và DR thất bại (chia mù không retrieval).",
        ],
        "CHART (fig3_reasoning_and_pareto.png): Paired s306 EM & Compute Pareto frontier",
        "1.5 min",
    ),
    (
        "7 · Reproduction Results",
        AMBER,
        "Reproduction II: Retrieval Crossover, Top-k & HybridMem Check",
        [
            "Xác nhận e5 thắng hop 2–4, BM25 thắng hop 5–10 ở Recall@20 (all-gold@20 <= 1.6% tại hop 10); nhưng tăng k=10->50 không tăng NR EM.",
            "Thử nghiệm 0.5B S4 + kiểm tra lại Table 3 cho thấy lợi ích trung bình của HybridMem rất nhỏ (+0.002 EM trên 40 ô).",
        ],
        "DUAL CHART (slide_sweeps_s3_s4.png): S3a Top-k flat EM + S4 / Table 3 audit",
        "1.5 min",
    ),
    (
        "7 · Key Audit Notes",
        AMBER,
        "Concise Audit Notes: Prompt Number Leakage & Repo Bugs",
        [
            "Data Audit: 53% câu hỏi (82% ở hop 10) lặp lại số từ fact -> cần báo cáo song song tập Leak-Free (không-memory giảm từ 6.5% xuống 1.4%).",
            "Code Audit (a24953c): thiếu scorer, BlockSFT lỗi tên adapter 'default', và padding SFT dư 4,608 tokens (đã vá trong repo tái lập).",
        ],
        "2-COLUMN AUDIT CARD: Prompt Leakage Example (RC000000#07) + Key Repo Fixes",
        "1.0 min",
    ),
    (
        "8 · Conclusion",
        GREEN,
        "Conclusion & Practical Takeaways for Personal Agents",
        [
            "Tổng hợp các khẳng định được xác nhận vs các điểm cần lưu ý từ bài báo KDD '26.",
            "3 khuyến nghị thực tiễn: (1) Dùng SR 3 bước; (2) Không chia nhỏ mù trước khi retrieval; (3) Lưu fact bằng Hybrid BM25+e5 thay vì LoRA.",
        ],
        "3 TAKEAWAY PILLARS + VERDICT SUMMARY: Xác nhận / Tinh chỉnh / Khuyến nghị",
        "1.0 min",
    ),
    (
        "8 · Team & Q&A",
        NAVY,
        "Team Contributions & Thank You",
        [
            "Hiển thị bảng phân công công việc 4 thành viên (Nguyễn Xuân Phi, Lê Hải Phong, Trần Nguyên Phong, Phan Văn Phú) & GVHD Nguyễn Võ Lê Duy.",
            "Kết thúc phần trình bày (~32 phút) và mở phiên hỏi đáp Q&A.",
        ],
        "TEAM TABLE + Q&A PROMPT: Phân công thành viên + Lời cảm ơn",
        "0.5 min",
    ),
]


class Group7TemplateBuilder:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.cmds: list[str] = []

    def emit(self, cmd: str, params: dict | None = None) -> None:
        if params is None:
            self.cmds.append(cmd)
        else:
            self.cmds.append(f"{cmd}={json.dumps(params, ensure_ascii=False)}")

    def prepare_base_slides(self, total: int) -> None:
        """Keep Slide 0 (with right-side hex cluster) for Title + duplicate for Closing,
        remove all old Group 7 text and slides 1..52, and insert BLANK_2 content slides."""
        self.emit("slide.go", {"index": 0})
        self.emit("edit.delete", {"ids": OLD_TITLE_TEXT_IDS})
        self.emit("slide.duplicate", {"index": 0})
        # Delete original slides 2..53 (after duplicating slide 0, old slides 1..52 shifted to 2..53).
        for _ in range(OLD_SLIDE_COUNT - 1):
            self.emit("slide.delete", {"index": 2})
        # Insert content slides (indices 1..total-2) with BLANK_2 (4-corner hex chrome).
        self.emit("slide.go", {"index": 0})
        for _ in range(total - 2):
            self.emit("slide.new", {"layout": "BLANK_2"})

    def rect(
        self,
        rect: list[float],
        fill: str,
        line: str | None = None,
        lw: float = 1.1,
        dash: str | None = None,
        text: str = "",
        size: float = 10.0,
        color: str = CHARCOAL,
        bold: bool = False,
        align: str = "left",
        anchor: str = "middle",
        preset: str = "roundRect",
    ) -> None:
        p: dict = {"preset": preset, "rect": rect}
        if text:
            p["text"] = text
        self.emit("shape.insert", p)
        self.emit("shape.fill", {"color": fill})
        if line is None:
            self.emit("shape.line", {"none": True})
        else:
            lp: dict = {"color": line, "width": lw}
            if dash:
                lp["dash"] = dash
            self.emit("shape.line", lp)
        if text:
            self.emit("format.font", {"family": FONT})
            self.emit("format.size", {"size": size})
            self.emit("format.color", {"color": color})
            if bold:
                self.emit("format.bold", {"on": True})
            self.emit("format.align", {"align": align})
            self.emit("format.anchor", {"anchor": anchor})

    def textbox(
        self,
        rect: list[float],
        text: str,
        size: float = 10.5,
        color: str = CHARCOAL,
        bold: bool = False,
        align: str = "left",
        anchor: str = "top",
        ls: float = 1.2,
        bullets: bool = False,
    ) -> None:
        self.emit("insert.textBox", {"rect": rect, "text": text})
        self.emit("format.font", {"family": FONT})
        self.emit("format.size", {"size": size})
        self.emit("format.color", {"color": color})
        if bold:
            self.emit("format.bold", {"on": True})
        self.emit("format.align", {"align": align})
        self.emit("format.anchor", {"anchor": anchor})
        self.emit("format.lineSpacing", {"lines": ls})
        self.emit("format.margins", {"left": 3, "top": 2, "right": 3, "bottom": 2})
        if bullets:
            self.emit("format.bullets", {"on": True, "char": "•"})

    def populate_hero_slide(
        self,
        num: int,
        total: int,
        tag: str,
        title: str,
        ideas: list[str],
        slot: str,
        mins: str,
    ) -> None:
        # Left column stays strictly within x=34..392 to stay clear of right-side hex group (x >= 402.75).
        pill_text = f"SLIDE {num} · {tag.upper()}"
        pill_w = max(128.0, len(pill_text) * 5.2 + 22.0)
        self.rect(
            [34, 18, pill_w, 18],
            fill=TEAL,
            text=pill_text,
            size=8.0,
            color=WHITE,
            bold=True,
            align="center",
        )
        self.rect(
            [34 + pill_w + 8, 18, 64, 18],
            fill=BLUE_BG,
            line="#B8D4E8",
            lw=0.8,
            text=f"~{mins}",
            size=8.0,
            color=TEAL,
            bold=True,
            align="center",
        )

        title_size = 17.5 if num == 1 else 20.0
        self.textbox(
            [34, 42, 358, 74],
            title,
            size=title_size,
            color=CHARCOAL,
            bold=True,
            anchor="bottom" if num == total else "top",
            ls=1.14,
        )
        self.textbox(
            [34, 121, 358, 18],
            f"Nhóm 7  ·  GVHD: {GVHD}",
            size=10.5,
            color=TEAL,
            bold=True,
        )

        # Idea card
        self.rect([34, 144, 358, 100], fill=BLUE_BG, line=TEAL, lw=1.1)
        self.textbox(
            [45, 150, 336, 15],
            "IDEA / NỘI DUNG CẦN TRÌNH BÀY:",
            size=8.0,
            color=TEAL,
            bold=True,
        )
        self.textbox(
            [45, 166, 336, 74],
            "\n".join(ideas),
            size=9.2,
            color=CHARCOAL,
            ls=1.2,
            bullets=True,
        )

        # Visual placeholder card + GVHD & 4 team members
        self.rect(
            [34, 252, 358, 114],
            fill=LIGHT_BG,
            line=MUTED,
            lw=1.1,
            dash="dash",
        )
        if num == 1:
            slot_txt = (
                f"[ PLACEHOLDER HÌNH / HERO ]\n{slot}\n\n"
                f"GVHD: {GVHD}\n"
                f"Nhóm 7: {TEAM_MEMBERS[0]} · {TEAM_MEMBERS[1]}\n"
                f"{TEAM_MEMBERS[2]} · {TEAM_MEMBERS[3]}"
            )
        else:
            slot_txt = (
                f"[ PLACEHOLDER BẢNG PHÂN CÔNG & Q&A ]\n{slot}\n\n"
                f"GVHD: {GVHD}\n"
                f"Thành viên: {TEAM_MEMBERS[0]} · {TEAM_MEMBERS[1]}\n"
                f"{TEAM_MEMBERS[2]} · {TEAM_MEMBERS[3]}"
            )
        self.textbox(
            [44, 258, 338, 102],
            slot_txt,
            size=8.8,
            color=CHARCOAL,
            bold=True,
            align="center",
            anchor="middle",
            ls=1.24,
        )

        self.textbox(
            [34, 376, 358, 16],
            f"MPR (KDD '26) Presentation Template · Group 7   |   Slide {num} / {total} ({mins})",
            size=7.8,
            color=MUTED,
        )

    def populate_content_slide(
        self,
        num: int,
        total: int,
        tag: str,
        accent: str,
        title: str,
        ideas: list[str],
        slot: str,
        mins: str,
    ) -> None:
        # Header stays inside x=92..628 to stay strictly clear of top corner hexes.
        pill_w = max(116.0, len(tag) * 5.6 + 20.0)
        self.rect(
            [92, 14, pill_w, 18],
            fill=accent,
            text=tag.upper(),
            size=8.0,
            color=WHITE,
            bold=True,
            align="center",
        )
        self.rect(
            [556, 14, 72, 18],
            fill=LIGHT_BG,
            line=BORDER,
            lw=0.8,
            text=f"~{mins}",
            size=8.0,
            color=MUTED,
            bold=True,
            align="center",
        )

        self.textbox(
            [92, 35, 536, 40],
            title,
            size=15.5,
            color=CHARCOAL,
            bold=True,
            anchor="middle",
            ls=1.12,
        )
        self.rect([92, 77, 536, 1.2], fill=DIVIDER, preset="rect")

        # Main cards occupy y=84..322, strictly between top (y<=79.5) and bottom (y>=325) corner hexes.
        card_bg = BG_BY_ACCENT.get(accent, BLUE_BG)
        self.rect([50, 84, 620, 102], fill=card_bg, line=accent, lw=1.2)
        self.textbox(
            [64, 90, 592, 16],
            "IDEA / NỘI DUNG CẦN TRÌNH BÀY:",
            size=8.5,
            color=accent,
            bold=True,
        )
        self.textbox(
            [64, 107, 592, 75],
            "\n".join(ideas),
            size=10.2,
            color=CHARCOAL,
            ls=1.2,
            bullets=True,
        )

        self.rect(
            [50, 196, 620, 126],
            fill=LIGHT_BG,
            line=MUTED,
            lw=1.2,
            dash="dash",
        )
        self.textbox(
            [72, 208, 576, 102],
            f"[ PLACEHOLDER CHO HÌNH / BẢNG / SƠ ĐỒ ]\n\n{slot}\n\n(Chèn figure từ paper hoặc assets/ khi hoàn thiện slide)",
            size=10.5,
            color=CHARCOAL,
            bold=True,
            align="center",
            anchor="middle",
            ls=1.28,
        )

        # Footer stays inside x=94..626 to avoid bottom corner hexes.
        self.rect([94, 362, 532, 0.8], fill=FOOTER_DIVIDER, preset="rect")
        self.textbox(
            [94, 368, 390, 16],
            "MPR (KDD '26) Presentation Template — Group 7 · Title + Idea Scaffold",
            size=8.0,
            color=MUTED,
        )
        self.textbox(
            [506, 368, 120, 16],
            f"Slide {num} / {total}",
            size=8.0,
            color=MUTED,
            bold=True,
            align="right",
        )

    def build(self, out_pptx: Path, out_deckcraft: Path) -> None:
        total = len(SLIDES_SPEC)
        self.prepare_base_slides(total)

        for idx, (tag, accent, title, ideas, slot, mins) in enumerate(SLIDES_SPEC):
            num = idx + 1
            self.emit("slide.go", {"index": idx})
            if num in (1, total):
                self.populate_hero_slide(num, total, tag, title, ideas, slot, mins)
            else:
                self.populate_content_slide(
                    num, total, tag, accent, title, ideas, slot, mins
                )
            note_text = (
                f"[{mins}] {title}\n"
                + "\n".join(f"- {x}" for x in ideas)
                + f"\nVisual: {slot}"
            )
            self.emit("slide.notes", {"text": note_text})

        self.emit("slide.go", {"index": 0})
        self.emit(
            "file.properties",
            {
                "title": "MPR (KDD '26) Presentation Template — Group 7",
                "subject": "Explicit v.s. Implicit Memory: Multi-hop Reasoning Over Personalized Information",
                "author": ", ".join(TEAM_MEMBERS),
            },
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tf:
            tf.write("\n".join(self.cmds) + "\n")
            cmd_path = Path(tf.name)

        try:
            subprocess.run(
                [
                    "deckcraft-cli",
                    "run",
                    "--in",
                    str(self.template_path),
                    "--cmd-file",
                    str(cmd_path),
                    "--save",
                    str(out_pptx),
                ],
                check=True,
            )
            subprocess.run(
                ["deckcraft-cli", "convert", str(out_pptx), str(out_deckcraft)],
                check=True,
            )
        finally:
            cmd_path.unlink(missing_ok=True)

        print(
            f"Built {out_pptx} and {out_deckcraft} ({total} slides, {len(self.cmds)} cmds)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Path to the Group 7 reference .pptx template",
    )
    parser.add_argument("--out-pptx", type=Path, default=OUT_PPTX)
    parser.add_argument("--out-deckcraft", type=Path, default=OUT_DECKCRAFT)
    args = parser.parse_args()
    Group7TemplateBuilder(args.template).build(args.out_pptx, args.out_deckcraft)


if __name__ == "__main__":
    main()
