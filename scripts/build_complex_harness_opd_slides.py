from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "slides"
OUTPUT = ROOT / "slides" / "Complex-Harness-OPD-and-Multi-turn-Agent-OPD-zh.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

FONT = "PingFang SC"
FONT_MONO = "Menlo"

INK = RGBColor(27, 35, 40)
MUTED = RGBColor(91, 103, 109)
PAPER = RGBColor(247, 246, 242)
WHITE = RGBColor(255, 255, 255)
TEAL = RGBColor(22, 126, 117)
TEAL_LIGHT = RGBColor(221, 240, 236)
AMBER = RGBColor(221, 151, 34)
AMBER_LIGHT = RGBColor(249, 235, 206)
CORAL = RGBColor(205, 87, 67)
CORAL_LIGHT = RGBColor(248, 226, 221)
BLUE = RGBColor(49, 102, 151)
BLUE_LIGHT = RGBColor(224, 235, 246)
GREEN = RGBColor(72, 137, 79)
GRAY = RGBColor(222, 225, 224)
LIGHT = RGBColor(237, 238, 235)
DARK = RGBColor(18, 30, 35)


def add_shape(slide, shape_type, x, y, w, h, fill, line=None, radius=False):
    shape = slide.shapes.add_shape(shape_type, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    *,
    size=18,
    color=INK,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0,
    line_spacing=1.05,
):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.clear()
    frame.margin_left = margin
    frame.margin_right = margin
    frame.margin_top = margin
    frame.margin_bottom = margin
    frame.vertical_anchor = valign
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = line_spacing
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_bullets(slide, items, x, y, w, h, *, size=17, color=INK, gap=7, bullet_color=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = f"• {item}"
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(gap)
        paragraph.line_spacing = 1.05
    return box


def add_header(slide, title, kicker=None, section_color=TEAL):
    if kicker:
        add_text(slide, kicker.upper(), Inches(0.55), Inches(0.34), Inches(3.0), Inches(0.3),
                 size=10, color=section_color, bold=True)
    add_text(slide, title, Inches(0.55), Inches(0.65), Inches(12.1), Inches(0.65),
             size=27, color=INK, bold=True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.33), Inches(12.2), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = LIGHT
    line.line.color.rgb = LIGHT


def add_footer(slide, index, source=None):
    add_text(slide, f"{index:02d}", Inches(12.45), Inches(7.07), Inches(0.35), Inches(0.2),
             size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    if source:
        add_text(slide, source, Inches(0.55), Inches(7.02), Inches(11.4), Inches(0.24),
                 size=8, color=MUTED)


def add_pill(slide, text, x, y, w, *, fill=TEAL_LIGHT, color=TEAL):
    shape = add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, Inches(0.34), fill, fill)
    add_text(slide, text, x, y + Inches(0.01), w, Inches(0.28), size=10, color=color,
             bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    return shape


def add_card(slide, title, body, x, y, w, h, *, accent=TEAL, fill=WHITE, title_size=17, body_size=13):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, GRAY)
    add_shape(slide, MSO_SHAPE.RECTANGLE, x, y, Inches(0.08), h, accent, accent)
    add_text(slide, title, x + Inches(0.25), y + Inches(0.18), w - Inches(0.42), Inches(0.38),
             size=title_size, color=INK, bold=True)
    add_text(slide, body, x + Inches(0.25), y + Inches(0.68), w - Inches(0.42), h - Inches(0.82),
             size=body_size, color=MUTED, line_spacing=1.0)


def add_image_contain(slide, path, x, y, w, h, *, border=True, fill=WHITE):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, GRAY if border else fill)
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(float(w) / iw, float(h) / ih)
    pw = int(iw * scale)
    ph = int(ih * scale)
    px = int(x + (w - pw) / 2)
    py = int(y + (h - ph) / 2)
    slide.shapes.add_picture(str(path), px, py, width=pw, height=ph)


def add_arrow(slide, x1, y1, x2, y2, *, color=MUTED, width=2.0):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    line.line.color.rgb = color
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def add_metric(slide, value, label, x, y, w, *, color=TEAL):
    add_text(slide, value, x, y, w, Inches(0.52), size=24, color=color, bold=True,
             align=PP_ALIGN.CENTER)
    add_text(slide, label, x, y + Inches(0.56), w, Inches(0.46), size=11, color=MUTED,
             align=PP_ALIGN.CENTER)


def set_background(slide, color=PAPER):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color


def build_deck():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    # 1. Title
    slide = prs.slides.add_slide(blank)
    set_background(slide, DARK)
    add_shape(slide, MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), SLIDE_H, TEAL, TEAL)
    add_text(slide, "复杂 Harness-OPD\n与 Multi-turn Agent OPD", Inches(0.78), Inches(1.25), Inches(8.7), Inches(1.8),
             size=36, color=WHITE, bold=True, line_spacing=0.95)
    add_text(slide, "从可执行 Harness 内化，到跨 turn temporal credit 的研究边界",
             Inches(0.82), Inches(3.35), Inches(8.9), Inches(0.55), size=18, color=RGBColor(197, 211, 214))
    add_pill(slide, "HARNESS-ZERO", Inches(0.82), Inches(4.28), Inches(1.65), fill=RGBColor(33, 75, 76), color=RGBColor(151, 230, 219))
    add_pill(slide, "OPRD", Inches(2.62), Inches(4.28), Inches(0.95), fill=RGBColor(71, 57, 25), color=RGBColor(244, 205, 114))
    add_pill(slide, "SKILL-SD / TCOD / SOD", Inches(3.72), Inches(4.28), Inches(2.35), fill=RGBColor(70, 42, 38), color=RGBColor(242, 173, 159))
    add_pill(slide, "LONGHORIZON", Inches(6.22), Inches(4.28), Inches(1.55), fill=RGBColor(31, 51, 74), color=RGBColor(161, 197, 231))
    add_text(slide, "2026.11", Inches(10.95), Inches(6.75), Inches(1.4), Inches(0.3), size=11, color=RGBColor(150, 166, 170), align=PP_ALIGN.RIGHT)

    # 2. Executive map
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "先分清三个不同的“上限”", "Executive map")
    add_card(slide, "严格 Code OPD", "OPRD\n学生自己的长程 coding 轨迹\nReminder-conditioned reverse KL", Inches(0.6), Inches(1.75), Inches(3.85), Inches(2.65), accent=AMBER)
    add_card(slide, "完整 Harness-to-weight", "Harness-Zero\ntools + middleware + skills + memory\nLive review + SFT", Inches(4.73), Inches(1.75), Inches(3.85), Inches(2.65), accent=TEAL)
    add_card(slide, "运行时复杂度上限", "LongHorizon-Harness\nManager–Executor–Auditor\n持久审计状态 + 权限隔离", Inches(8.85), Inches(1.75), Inches(3.85), Inches(2.65), accent=BLUE)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.15), Inches(5.05), Inches(11.05), Inches(1.05), DARK, DARK)
    add_text(slide, "当前前沿不是“把 LongHorizon 蒸完了”，而是明确：哪些能力能进权重，哪些结构必须留在 runtime。",
             Inches(1.45), Inches(5.32), Inches(10.45), Inches(0.48), size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 2, "Sources: arXiv:2609.24974, 2610.21496, 2608.01964")

    # 3. Landscape
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "两条轴：Harness 复杂度 × 内化严格性", "Landscape")
    x0, y0 = Inches(1.15), Inches(5.95)
    x1, y1 = Inches(12.05), Inches(1.75)
    add_arrow(slide, x0, y0, x1, y0, color=INK)
    add_arrow(slide, x0, y0, x0, y1, color=INK)
    add_text(slide, "Harness 工程复杂度", Inches(8.8), Inches(6.12), Inches(2.8), Inches(0.3), size=12, color=MUTED, bold=True)
    add_text(slide, "进入权重的严格性", Inches(0.15), Inches(1.55), Inches(1.0), Inches(0.65), size=12, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    points = [
        ("OPHSD", 2.4, 2.0, AMBER),
        ("TCOD / SOD", 4.4, 3.0, CORAL),
        ("Skill-SD", 5.8, 3.4, CORAL),
        ("OPRD", 7.2, 2.1, AMBER),
        ("Harness-Zero", 8.8, 3.7, TEAL),
        ("AHE", 9.7, 5.0, GREEN),
        ("LongHorizon", 11.0, 5.4, BLUE),
    ]
    for label, x, y, color in points:
        px = Inches(x)
        py = Inches(y)
        add_shape(slide, MSO_SHAPE.OVAL, px, py, Inches(0.34), Inches(0.34), color, WHITE)
        add_text(slide, label, px - Inches(0.45), py + Inches(0.37), Inches(1.3), Inches(0.3), size=11, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "高复杂但不更新权重", Inches(9.25), Inches(1.62), Inches(2.6), Inches(0.3), size=11, color=BLUE, bold=True)
    add_text(slide, "严格 OPD 仍集中在局部 privilege", Inches(1.5), Inches(5.1), Inches(3.5), Inches(0.3), size=11, color=AMBER, bold=True)
    add_footer(slide, 3, "位置为概念性比较，不代表统一实验尺度。")

    # 4. Harness-Zero overview
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "Harness-Zero：目前最完整的 Harness-to-weight 案例", "Core work 01", TEAL)
    add_image_contain(slide, ASSETS / "harness-zero-overview.png", Inches(0.55), Inches(1.55), Inches(7.4), Inches(4.9))
    add_card(slide, "源 Harness h*", "三轮演化\ntools · middleware · skills · memory", Inches(8.25), Inches(1.65), Inches(4.45), Inches(1.25), accent=TEAL)
    add_card(slide, "Adapt(h*; h) → K", "把源能力改写成 review 条件与学生原生动作配方", Inches(8.25), Inches(3.05), Inches(4.45), Inches(1.25), accent=AMBER)
    add_card(slide, "部署", "移除 h*、K、harnessing agent\n保留极简目标 Harness h", Inches(8.25), Inches(4.45), Inches(4.45), Inches(1.25), accent=BLUE)
    add_metric(slide, "23.3 → 44.3", "三领域宏平均，h only", Inches(8.25), Inches(5.85), Inches(2.15), color=TEAL)
    add_metric(slide, "82.3%", "28 个行为恢复率", Inches(10.45), Inches(5.85), Inches(2.15), color=CORAL)
    add_footer(slide, 4, "Harness-Zero, arXiv:2609.24974")

    # 5. K implementation
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "K 不是模型：它是 Reviewer-side Harness Bank", "Harness-Zero deep dive", TEAL)
    add_text(slide, "h* 组件", Inches(0.75), Inches(1.6), Inches(2.0), Inches(0.35), size=15, color=MUTED, bold=True)
    add_text(slide, "Adapt 后的 K", Inches(5.0), Inches(1.6), Inches(2.0), Inches(0.35), size=15, color=MUTED, bold=True)
    rows = [
        ("Tool", "Action recipe (.md)", "Bash / Python 等价实现"),
        ("Middleware", "Review middleware (.py)", "检测 proposal / visible history"),
        ("Skill", "Review guidance (.md)", "判断缺失步骤，构造下一动作"),
        ("Memory", "Failure patterns (.md)", "trigger · evidence · intervention"),
    ]
    y = 2.05
    for left, middle, right in rows:
        add_card(slide, left, "student-side", Inches(0.75), Inches(y), Inches(2.25), Inches(0.88), accent=TEAL, title_size=15, body_size=10)
        add_arrow(slide, Inches(3.15), Inches(y + 0.43), Inches(4.55), Inches(y + 0.43), color=AMBER)
        add_card(slide, middle, right, Inches(4.75), Inches(y), Inches(4.15), Inches(0.88), accent=AMBER, title_size=14, body_size=10)
        y += 1.05
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.25), Inches(2.05), Inches(3.3), Inches(4.1), BLUE_LIGHT, BLUE)
    add_text(slide, "运行时组合", Inches(9.55), Inches(2.28), Inches(2.7), Inches(0.38), size=18, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "GPT-5.6 reviewer\n+ K 文件与 middleware\n+ 当前 student proposal\n+ student-visible history\n\n→ PASS / REPLACE",
             Inches(9.55), Inches(2.95), Inches(2.7), Inches(2.45), size=16, color=INK, align=PP_ALIGN.CENTER, line_spacing=1.15)
    add_text(slide, "真正的 policy 是 reviewer 模型；K 提供私有规则、证据与动作翻译。",
             Inches(9.55), Inches(5.45), Inches(2.7), Inches(0.48), size=11, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 5, "Implementation: metaevo-ai/harness-zero, harness_bank/<domain>/")

    # 6. OPRD
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "OPRD：Code Agent 中最直接的严格 Multi-turn OPD", "Core work 02", AMBER)
    add_image_contain(slide, ASSETS / "oprd-overview.png", Inches(0.55), Inches(1.55), Inches(7.25), Inches(4.6))
    add_card(slide, "Student rollout", "Reminder-free\nClaude Code scaffold\n最多 200 turns", Inches(8.05), Inches(1.62), Inches(2.2), Inches(1.55), accent=AMBER)
    add_card(slide, "成功轨迹", "Autoregressive\nself-imitation", Inches(10.45), Inches(1.62), Inches(2.2), Inches(1.55), accent=GREEN)
    add_card(slide, "失败轨迹", "Reminder-conditioned\nTop-100 reverse KL", Inches(8.05), Inches(3.4), Inches(2.2), Inches(1.55), accent=CORAL)
    add_card(slide, "Temporal scope", "Global constraint\n+ Turn-local feedback", Inches(10.45), Inches(3.4), Inches(2.2), Inches(1.55), accent=BLUE)
    add_metric(slide, "36.6 → 47.9", "Qwen3.5-4B 宏平均", Inches(8.1), Inches(5.35), Inches(2.15), color=AMBER)
    add_metric(slide, "446 → 222", "过程错误事件", Inches(10.4), Inches(5.35), Inches(2.15), color=CORAL)
    add_footer(slide, 6, "OPRD, arXiv:2610.21496v2")

    # 7. LongHorizon
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "LongHorizon-Harness：运行时结构上限，尚未被蒸馏", "Runtime ceiling", BLUE)
    add_image_contain(slide, ASSETS / "longhorizon-mea.png", Inches(0.55), Inches(1.55), Inches(7.4), Inches(4.9))
    add_bullets(slide, [
        "Manager：维护结构化任务状态，但无环境权限",
        "Executor：fresh context，唯一可写环境的角色",
        "Auditor：独立、只读、从真实环境取证",
        "最多 25 轮；只有 audited state 跨轮持久化",
        "GUI / CLI 路由与 role-scoped permissions",
    ], Inches(8.25), Inches(1.8), Inches(4.2), Inches(3.55), size=15, gap=9)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.25), Inches(5.35), Inches(4.2), Inches(0.9), CORAL_LIGHT, CORAL)
    add_text(slide, "不能简单进权重：动态账本 · 独立性 · 权限边界 · 跨上下文恢复",
             Inches(8.48), Inches(5.58), Inches(3.75), Inches(0.42), size=13, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 7, "LongHorizon-Harness, arXiv:2608.01964")

    # 8. Multi-turn gap
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "核心 Gap：在错误状态上纠正，不等于教会模型别进入它", "Multi-turn OPD")
    xs = [0.75, 3.15, 5.55, 7.95, 10.35]
    labels = [
        ("Turn 1", "错误目录", TEAL),
        ("Turn 2", "错误搜索", AMBER),
        ("Turn 3", "修改错文件", CORAL),
        ("Turn 4", "测试失败", CORAL),
        ("Local KL", "只教此刻如何继续", BLUE),
    ]
    for idx, (top, body, color) in enumerate(labels):
        add_card(slide, top, body, Inches(xs[idx]), Inches(2.15), Inches(1.85), Inches(1.2), accent=color, body_size=12)
        if idx < len(labels) - 1:
            add_arrow(slide, Inches(xs[idx] + 1.9), Inches(2.75), Inches(xs[idx + 1] - 0.05), Inches(2.75), color=MUTED)
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.1), Inches(4.15), Inches(11.15), Inches(1.35), DARK, DARK)
    add_text(slide, "Learning at student states does not by itself assign credit for reaching them.",
             Inches(1.45), Inches(4.42), Inches(10.45), Inches(0.38), size=19, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "真正缺失的是：后续 environment outcome 如何反向更新早期 action。",
             Inches(1.45), Inches(4.92), Inches(10.45), Inches(0.32), size=14, color=RGBColor(188, 208, 211), align=PP_ALIGN.CENTER)
    add_text(slide, "multi-turn trajectory 上有 loss  ≠  已解决 cross-turn causal credit",
             Inches(2.25), Inches(6.05), Inches(8.85), Inches(0.4), size=18, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 8, "Source: OPD Survey, arXiv:2604.00626v5, §2.7 & §7")

    # 9. Skill-SD
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "Skill-SD：Multi-turn OPSD 中的动态 Teacher 信息", "Core work 03", CORAL)
    add_image_contain(slide, ASSETS / "skill-sd-overview.png", Inches(0.55), Inches(1.55), Inches(7.25), Inches(4.8))
    add_card(slide, "为什么需要 Skill", "Agent 任务存在多条有效 action path；固定答案会压制探索。", Inches(8.05), Inches(1.62), Inches(4.55), Inches(1.15), accent=CORAL)
    add_card(slide, "动态 privileged context", "整条轨迹 → success_analysis / mistake_analysis / golden_workflow", Inches(8.05), Inches(2.95), Inches(4.55), Inches(1.25), accent=AMBER)
    add_card(slide, "训练", "Student-owned rollout + 动态同步 Teacher\nGRPO + importance-weighted sampled-token reverse KL", Inches(8.05), Inches(4.38), Inches(4.55), Inches(1.35), accent=TEAL)
    add_text(slide, "定位：解决“Teacher 应该知道什么”，不是完整 temporal credit。",
             Inches(8.12), Inches(6.05), Inches(4.35), Inches(0.38), size=13, color=CORAL, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 9, "Skill-SD, arXiv:2604.10674")

    # 10. TCOD
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "TCOD：先学短 Horizon，再暴露完整轨迹", "Core work 04", BLUE)
    add_image_contain(slide, ASSETS / "tcod-overview.png", Inches(0.55), Inches(1.55), Inches(7.35), Inches(4.85))
    add_card(slide, "Failure mode", "Trajectory-Level KL Instability\nturn 越深，错误累积与 KL escalation 越强", Inches(8.18), Inches(1.7), Inches(4.3), Inches(1.45), accent=CORAL)
    add_card(slide, "F2B", "限制学生 rollout 深度\n由早期 turn 逐步扩展到完整 horizon", Inches(8.18), Inches(3.38), Inches(2.05), Inches(1.55), accent=TEAL)
    add_card(slide, "B2F", "Teacher 导航到近终态\n学生从后半段逐步向前接管", Inches(10.43), Inches(3.38), Inches(2.05), Inches(1.55), accent=AMBER)
    add_text(slide, "解决：horizon exposure 与训练稳定性\n未解决：最终失败归因到哪个早期动作",
             Inches(8.28), Inches(5.35), Inches(4.1), Inches(0.75), size=14, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 10, "TCOD, arXiv:2604.24005v3")

    # 11. SOD
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "SOD：不是每一个 Step 都值得同等蒸馏", "Core work 05", CORAL)
    add_image_contain(slide, ASSETS / "sod-overview.png", Inches(0.55), Inches(1.55), Inches(7.35), Inches(4.85))
    add_card(slide, "Tool-induced jump", "错误 tool call 注入整段 observation，状态偏移不是渐进的。", Inches(8.15), Inches(1.62), Inches(4.4), Inches(1.2), accent=CORAL)
    add_card(slide, "Step-wise reliability", "按每步 student–teacher divergence 调整 wₖ：\n偏离时减弱，恢复时重新增强。", Inches(8.15), Inches(3.0), Inches(4.4), Inches(1.35), accent=TEAL)
    add_card(slide, "边界", "dₖ 是 disagreement proxy，不是动作正确性或因果责任。", Inches(8.15), Inches(4.55), Inches(4.4), Inches(1.15), accent=AMBER)
    add_footer(slide, 11, "SOD, arXiv:2605.07725v3")

    # 12. Comparison
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "Multi-turn Agent OPD：已经形成专门研究分支", "Synthesis")
    headers = ["工作", "核心问题", "主要机制", "仍缺什么"]
    data = [
        ["Skill-SD", "Teacher 知道什么", "轨迹 → dynamic skills", "跨 turn 因果 credit"],
        ["TCOD", "训练到多深", "Short-to-long curriculum", "定位早期责任动作"],
        ["SOD", "哪些 step 可信", "Divergence reweighting", "disagreement ≠ correctness"],
        ["OPRD", "反馈作用多久", "Global + Turn-local", "scope ≠ trajectory gradient"],
        ["HINT-SD", "后见反馈给哪里", "Selected action spans", "span judge 仍是 heuristic"],
        ["MemOPD", "Replay 状态是否一致", "Memory-state alignment", "对齐 ≠ delayed credit"],
    ]
    table_shape = slide.shapes.add_table(len(data) + 1, len(headers), Inches(0.65), Inches(1.65), Inches(12.05), Inches(4.85))
    table = table_shape.table
    widths = [1.55, 2.45, 3.35, 4.7]
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    for col, header in enumerate(headers):
        cell = table.cell(0, col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        cell.text = header
        for p in cell.text_frame.paragraphs:
            p.font.name = FONT
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
    row_colors = [TEAL_LIGHT, BLUE_LIGHT, CORAL_LIGHT, AMBER_LIGHT, WHITE, LIGHT]
    for row_idx, row in enumerate(data, start=1):
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_colors[row_idx - 1]
            cell.text = value
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT
                p.font.size = Pt(11.5)
                p.font.bold = col_idx == 0
                p.font.color.rgb = INK
                p.alignment = PP_ALIGN.LEFT if col_idx else PP_ALIGN.CENTER
    add_text(slide, "共同点：都在学生访问状态上提供更好的局部监督；差异在于 state、scope、horizon、teacher information 与 credit。",
             Inches(1.05), Inches(6.68), Inches(11.2), Inches(0.28), size=12, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 12, "Agentic OPD map: arXiv:2604.00626v5, §7")

    # 13. Open challenges
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "尚未解决：Multi-turn OPD 的六个结构性难点", "Open problems", CORAL)
    challenges = [
        ("01", "State identity", "评分时必须还原 action 当时真正可见的 observation 与 memory。", TEAL),
        ("02", "Cross-turn credit", "后续 failure 如何低方差地更新早期 action。", CORAL),
        ("03", "Teacher reliability", "高 disagreement 可能是纠正，也可能是 teacher 失去校准。", AMBER),
        ("04", "Counterfactual", "更好的 action 往往必须执行后才知道长期后果。", BLUE),
        ("05", "Intervention shift", "Teacher 接管会改变后续 state distribution。", GREEN),
        ("06", "Compute allocation", "Fresh rollout、深 horizon、teacher query 与 replay 如何分配。", INK),
    ]
    positions = [(0.65, 1.65), (4.72, 1.65), (8.78, 1.65), (0.65, 4.15), (4.72, 4.15), (8.78, 4.15)]
    for (number, title, body, accent), (x, y) in zip(challenges, positions):
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(3.65), Inches(1.9), WHITE, GRAY)
        add_shape(slide, MSO_SHAPE.OVAL, Inches(x + 0.22), Inches(y + 0.22), Inches(0.55), Inches(0.55), accent, accent)
        add_text(slide, number, Inches(x + 0.22), Inches(y + 0.31), Inches(0.55), Inches(0.25), size=11, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, title, Inches(x + 0.95), Inches(y + 0.24), Inches(2.35), Inches(0.38), size=17, color=INK, bold=True)
        add_text(slide, body, Inches(x + 0.32), Inches(y + 0.88), Inches(3.0), Inches(0.75), size=12.5, color=MUTED, align=PP_ALIGN.CENTER)
    add_footer(slide, 13, "Research gaps synthesized from TCOD, SOD, OPRD, Skill-SD and the OPD survey.")

    # 14. Research agenda
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_header(slide, "下一步：从局部 Teacher Matching 到可信最小 Runtime", "Research agenda", TEAL)
    stages = [
        ("A", "Manager policy", "保留 task-state API\n蒸馏分解与状态更新", TEAL),
        ("B", "Auditor-guided OPD", "独立审计提供 privileged guidance\n蒸馏核验与恢复行为", AMBER),
        ("C", "Action translation", "Harness-Zero 式翻译\n统一到学生原生 action space", CORAL),
        ("D", "Harness-Minimal", "权重学策略\nruntime 留状态、权限、独立性", BLUE),
    ]
    x = 0.75
    for idx, (letter, title, body, color) in enumerate(stages):
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.85), Inches(2.55), Inches(2.35), WHITE, color)
        add_shape(slide, MSO_SHAPE.OVAL, Inches(x + 0.88), Inches(1.58), Inches(0.78), Inches(0.78), color, color)
        add_text(slide, letter, Inches(x + 0.88), Inches(1.77), Inches(0.78), Inches(0.28), size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, title, Inches(x + 0.25), Inches(2.55), Inches(2.05), Inches(0.42), size=16, color=INK, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, body, Inches(x + 0.25), Inches(3.15), Inches(2.05), Inches(0.78), size=12, color=MUTED, align=PP_ALIGN.CENTER)
        if idx < 3:
            add_arrow(slide, Inches(x + 2.62), Inches(3.0), Inches(x + 3.0), Inches(3.0), color=MUTED)
        x += 3.05
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.15), Inches(5.05), Inches(11.0), Inches(1.05), DARK, DARK)
    add_text(slide, "L = L_local teacher  +  λ_temporal L_future credit  +  λ_outcome L_verifier",
             Inches(1.55), Inches(5.34), Inches(10.2), Inches(0.32), size=15, color=WHITE, bold=True, font=FONT_MONO, align=PP_ALIGN.CENTER)
    add_footer(slide, 14, "Proposed synthesis: retain dynamic state and enforcement; distill procedures, verification habits and recovery policy.")

    # 15. Takeaways
    slide = prs.slides.add_slide(blank)
    set_background(slide, DARK)
    add_text(slide, "三条 Takeaway", Inches(0.75), Inches(0.75), Inches(5.0), Inches(0.6), size=30, color=WHITE, bold=True)
    takeaways = [
        ("01", "复杂 Harness 已可部分内化", "Harness-Zero 证明 tools、middleware、skills、memory 的可表达行为可以迁移到权重。", TEAL),
        ("02", "严格 Code OPD 已经存在", "OPRD 在真实长程 coding 轨迹上做 outcome-routed Reminder distillation。", AMBER),
        ("03", "Multi-turn gap 仍未闭合", "TCOD、SOD、Skill-SD 解决 horizon、reliability、teacher context；cross-turn causal credit 仍开放。", CORAL),
    ]
    y = 1.65
    for number, title, body, color in takeaways:
        add_shape(slide, MSO_SHAPE.OVAL, Inches(0.85), Inches(y), Inches(0.68), Inches(0.68), color, color)
        add_text(slide, number, Inches(0.85), Inches(y + 0.2), Inches(0.68), Inches(0.25), size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, title, Inches(1.85), Inches(y - 0.02), Inches(4.2), Inches(0.42), size=20, color=WHITE, bold=True)
        add_text(slide, body, Inches(1.85), Inches(y + 0.52), Inches(9.75), Inches(0.55), size=14, color=RGBColor(193, 207, 210))
        y += 1.65
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.65), Inches(5.85), Inches(3.6), Inches(0.7), RGBColor(35, 54, 60), RGBColor(35, 54, 60))
    add_text(slide, "目标：Harness-Minimal，而非 Harness-Free",
             Inches(8.78), Inches(6.05), Inches(3.35), Inches(0.28), size=13, color=RGBColor(164, 224, 216), bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, 15)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    output = build_deck()
    print(output)