from __future__ import annotations

import base64
from contextlib import suppress
import html
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import websocket


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "agentic-opd-reconstructed"
MARKDOWN = ROOT / "notes" / "Agentic-OPD-development-roadmap-2026.10-reconstructed.md"
HTML = ROOT / "documents" / "Agentic-OPD-development-roadmap-2026.10-reconstructed.html"
PDF = ROOT / "documents" / "Agentic-OPD-development-roadmap-2026.10-reconstructed.pdf"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


PAGES = [
    {
        "number": "01",
        "title": "为什么需要 Agentic OPD？",
        "subtitle": "从整段 token matching，走向错误传播下的 step-wise reliability",
        "methods": [
            {
                "name": "SOD",
                "paper": "Step-wise On-policy Distillation for Small Language Model Agents",
                "arxiv": "2605.07725v3",
                "image": "sod.png",
                "tag": "Step reliability",
                "lead": "多轮工具交互中，一个错误 action 会注入大段异常 observation。后续状态逐步离开 Teacher support，统一累加 OPD loss 会把高方差监督当成同等可靠。",
                "bullets": [
                    "按模型生成步骤聚合 Student–Teacher log-prob gap，而不是把整条 trajectory 视作同质 token 流。",
                    "前一步分歧变大时降低当前 step 权重；策略恢复时允许权重回升，并限制最大放大幅度。",
                    "核心贡献是 reliability weighting；它仍以 disagreement 作为 proxy，不等价于真实因果责任。",
                ],
                "equations": [
                    r"d_k=\frac{1}{|\mathcal I_k|}\sum_{t\in\mathcal I_k}\left|\log\pi_\theta(y_t\mid y_{<t})-\log\pi_{\mathrm{teacher}}(y_t\mid y_{<t})\right|",
                    r"w_k=\min\!\left(w_1\prod_{u=1}^{k-1}\frac{d_u+\epsilon}{d_{u+1}+\epsilon},\,1+\delta\right)",
                ],
            }
        ],
        "bottom": "Agentic OPD 的第一条主线：先判断当前 step 的 Teacher signal 是否还可信。",
    },
    {
        "number": "02",
        "title": "把 Hindsight 变成 Step-level Credit",
        "subtitle": "StepOPSD：不改变在线 rollout，只在事后重塑 advantage",
        "methods": [
            {
                "name": "StepOPSD",
                "paper": "Step-Aware Online Preference Self-Distillation for Agent Reinforcement Learning",
                "arxiv": "2605.27140v2",
                "image": "stepopsd.png",
                "tag": "Hindsight shaping",
                "lead": "Student 仍按原策略与环境交互；rollout 完成后，将轨迹切成 causal action spans，并在加入 hindsight 的 Teacher context 下重评分同一 token。",
                "bullets": [
                    "Teacher–Student likelihood gap 提供 dense token preference，但不替换环境给出的 outcome advantage。",
                    "用 advantage 符号决定 gap 的方向，再经过 sigmoid 与 clip 得到有界 credit weight。",
                    "最终只调节 GRPO advantage 的幅度，保留原始 outcome 的正负方向。",
                ],
                "equations": [
                    r"\Delta_{k,j}=\log\pi_T(z_{k,j}\mid c_k^T,z_{k,<j})-\log\pi_S(z_{k,j}\mid c_k^S,z_{k,<j})",
                    r"\widetilde A_\ell=(1-\lambda_{\mathrm{mix}})A_\ell+\lambda_{\mathrm{mix}}(w_\ell A_\ell)",
                ],
            }
        ],
        "bottom": "变化：Teacher 不再只提供 imitation target，而开始参与 outcome credit 的局部再分配。",
    },
    {
        "number": "03",
        "title": "让 OPD 感知 Turn 与 Horizon",
        "subtitle": "TurnOPD：训练多深、每个 turn 占多大权重，都由在线统计控制",
        "methods": [
            {
                "name": "TurnOPD",
                "paper": "Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training",
                "arxiv": "2607.05804v2",
                "image": "turnopd.png",
                "tag": "Turn-aware horizon",
                "lead": "Vanilla OPD 在长 horizon 中出现外部错配与内部错配：Teacher uncertainty 和 reverse-KL 随 turn 加深，深层 token 又可能在 trajectory normalization 下占据过多 loss。",
                "bullets": [
                    "用每 turn KL 与 survivor ratio 估计有效 rollout 深度。",
                    "再用成功轨迹长度分位数给出 coverage lower bound，避免 horizon 过短。",
                    "从 trajectory-normalized loss 渐进切换到 turn-normalized loss，减少长 turn 对梯度的垄断。",
                ],
                "equations": [
                    r"H_{\mathrm{ctrl}}=\max(H_{\mathrm{eff}},H_{\mathrm{cov}})",
                    r"q_t^{\mathrm{blend}}=(1-\alpha)\frac{n_t}{\sum_jn_j}+\alpha\frac1T",
                ],
            }
        ],
        "bottom": "Agentic OPD 的第二条主线：监督不仅要可信，还要控制暴露到多深的状态分布。",
    },
    {
        "number": "04",
        "title": "控制 Student 访问的状态分布",
        "subtitle": "从 DAgger 的 Teacher mixing，到 Guided-OPD 的 curriculum turn guidance",
        "methods": [
            {
                "name": "Revisiting DAgger",
                "paper": "Revisiting DAgger in the Era of LLM-Agents",
                "arxiv": "2605.12913v1",
                "image": "dagger.png",
                "tag": "Mixed-policy rollout",
                "lead": "仅在 Teacher demonstrations 上 SFT 会遭遇 covariate shift；纯 Student rollout 又会在冷启动时迅速进入无效状态。DAgger 在每个 turn 随机选择 Teacher 或 Student，并收集 Teacher labels。",
                "bullets": ["逐 turn mixture 扩大可恢复状态覆盖。", "训练目标仍是 Teacher action 的交叉熵。"],
                "equations": [r"\mu_i=\beta_i\pi_e+(1-\beta_i)\pi_{\theta_i}"],
            },
            {
                "name": "Guided-OPD",
                "paper": "On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents",
                "arxiv": "2606.15912v3",
                "image": "guided-opd.png",
                "tag": "Curriculum mixing",
                "lead": "Guided-OPD 将 DAgger 式混合嵌入 OPD：训练早期 Teacher turns 较多，随后按 curriculum 衰减到纯 Student on-policy rollout。",
                "bullets": ["Teacher turn 使用 forward-KL；Student turn 使用 reverse-KL。", "最终阶段 β=0，与标准 OPD 一致。"],
                "equations": [r"\mathcal L_t=(1-\beta_t)\mathcal J_{\mathrm{rKL}}^{\pi_\theta}+\beta_t\mathcal J_{\mathrm{fKL}}^{\pi_\phi}"],
            },
        ],
        "bottom": "关键取舍：Teacher intervention 改善状态覆盖，但 intervention 后的 history 不再是纯 Student on-policy。",
    },
    {
        "number": "05",
        "title": "选择性干预与 Offline Prefix Replay",
        "subtitle": "不必在每个 turn 都调用 Teacher，也不必让环境在训练全程保持在线",
        "methods": [
            {
                "name": "SAGE-OPD",
                "paper": "Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation",
                "arxiv": "2606.19659v1",
                "image": "sage-opd.png",
                "tag": "Selective intervention",
                "lead": "Intervention Agent 依据 Student response、Teacher disagreement 与 confidence，在 Skip / Weak / Strong 三档之间选择。",
                "bullets": ["用 Teacher confidence 抑制不可靠干预。", "按实际被监督 token 数归一化，保持 loss scale 稳定。"],
                "equations": [r"s_t=i_t c_t,\qquad \mathcal L_{\mathrm{SAGE}}=Z\sum_t s_t\mathcal L_{\mathrm{OPD}}(h_t,a_t)"],
            },
            {
                "name": "ReOPD",
                "paper": "Multi-Turn On-Policy Distillation with Prefix Replay",
                "arxiv": "2607.04763v3",
                "image": "reopd.png",
                "tag": "Offline replay",
                "lead": "Teacher trajectories 只需离线收集一次；Student training 通过 prefix replay 构造状态，环境无需始终在线，并让较早 step 获得更高权重。",
                "bullets": ["用 Student / Teacher occupancy 的 geometric bridge 控制 replay distribution。", "用指数衰减近似长期 distribution-shift cost。"],
                "equations": [r"\rho_t^*(h_t\mid x)\propto[d_t^{\theta_{\mathrm{old}}}]^{\gamma_t}[d_t^T]^{1-\gamma_t},\qquad \omega(t,\kappa)=\kappa^t"],
            },
        ],
        "bottom": "这一阶段开始同时优化监督质量、Teacher query 成本与 environment availability。",
    },
    {
        "number": "06",
        "title": "动态 Teacher 信息：从固定答案到 Skill",
        "subtitle": "Skill-SD：Teacher privilege 来自 trajectory analysis，而不是唯一标准路径",
        "methods": [
            {
                "name": "Skill-SD",
                "paper": "Skill-Conditioned Self-Distillation for Multi-turn LLM Agents",
                "arxiv": "2604.10674v1",
                "image": "skill-sd.png",
                "tag": "Dynamic privileged context",
                "lead": "Agent task 往往存在多条有效 action path。Skill-SD 将完成轨迹异步总结为 success analysis、mistake analysis 与 golden workflow，并存入 UCB-indexed skill buffer。",
                "bullets": [
                    "Student 使用原始 prompt；Teacher 在同一 token prefix 上额外读取检索到的 skills。",
                    "GRPO 提供 task-level credit；importance-weighted SDL 迁移 token-level Teacher distribution。",
                    "动态 Skill 解决 Teacher 应该知道什么，但不直接解决后续 outcome 对早期 action 的因果归因。",
                ],
                "equations": [
                    r"e=\operatorname{Analyze}(\tau,x)=(e_{\mathrm{success}},e_{\mathrm{mistake}},e_{\mathrm{workflow}})",
                    r"\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{GRPO}}+\lambda\mathcal L_{\mathrm{SDL}}",
                ],
            }
        ],
        "bottom": "Agentic OPD 的第三条主线：privileged information 从静态 oracle 走向随经验演化的可检索技能。",
    },
    {
        "number": "07",
        "title": "信息不对称：Privileged Teacher 也可能误导",
        "subtitle": "AR-OPD 校准 support mismatch；HINT 把 hindsight 压缩到真正的 decision fork",
        "methods": [
            {
                "name": "AR-OPD",
                "paper": "Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation",
                "arxiv": "2606.10385v1",
                "image": "ar-opd.png",
                "tag": "Anchored residual",
                "lead": "Full privileged context 可能让 Teacher 依赖 Student 永远看不到的答案状态。AR-OPD 先构造 partial anchor，再只迁移 full view 相对 anchor 的 residual。",
                "bullets": ["Anchor 保留可由 Student 重建的支持。", "λ 控制 privileged residual 强度，避免绝对模仿。"],
                "equations": [r"\log q_\lambda=\log q_{\mathrm{part}}+\lambda(\log q_{\mathrm{full}}-\log q_{\mathrm{part}})-C_t"],
            },
            {
                "name": "HINT",
                "paper": "Multi-Turn Agent Self-Distillation with Compressed Hindsight",
                "arxiv": "2606.11559v2",
                "image": "hint.png",
                "tag": "Compressed hindsight",
                "lead": "Reflector 将完整 trajectory outcome 压缩为 turn-level diagnosis 与可选纠正动作，只在识别出的错误 turns 蒸馏。原始长图写作 HERO，但对应论文和方法实际为 HINT。",
                "bullets": ["Teacher context 注入 diagnosis，而不是泄露完整参考答案。", "JSD loss 聚焦错误 turn 的 action tokens。"],
                "equations": [r"h_t=(d_t,\hat a^t),\qquad \mathcal L_{\mathrm{HINT}}=\frac1{|I(\tau)|}\sum_{t\in I(\tau)}\ell_t"],
            },
        ],
        "bottom": "新问题：Teacher 拥有更多信息不代表它在 Student state 上一定校准良好。",
    },
    {
        "number": "08",
        "title": "Bayesian Credit：从证据增量到 Turn Advantage",
        "subtitle": "OPPO 给出 token-level value recursion；PBSD 将 privileged evidence 引入长程 Agent",
        "methods": [
            {
                "name": "OPPO",
                "paper": "Bayesian Value Recursion for Token-Level Credit Assignment in LLM Reasoning",
                "arxiv": "2605.21851v2",
                "image": "oppo.png",
                "tag": "Token Bayesian evidence",
                "lead": "把每个 token 的 likelihood ratio 视为 Bayesian evidence，递归更新成功概率；相邻 value 的变化就是局部 advantage。",
                "bullets": ["Token advantages telescoping 到最终 reward 与初始 value 的差。", "一阶近似把 credit 分解为 uncertainty × evidence。"],
                "equations": [r"V_{t+1}=\frac{\lambda_t^*V_t}{\lambda_t^*V_t+(1-V_t)},\qquad A_t=V_{t+1}-V_t", r"\sum_{t=1}^T A_t=R-V_0"],
            },
            {
                "name": "PBSD",
                "paper": "Privileged Bayesian Self-Distillation for Long-Horizon Credit Assignment",
                "arxiv": "2606.09348v2",
                "image": "pbsd.png",
                "tag": "Turn Bayesian evidence",
                "lead": "比较有/无 privileged outcome information 时当前 action 的 likelihood，得到 turn-level evidence score，再通过 decision gate 调整 sequence advantage。",
                "bullets": ["Bayes identity 将 outcome posterior 转换为 trajectory evidence ratio。", "保留 sequence reward 的方向，同时在 turn 内重新分配幅度。"],
                "equations": [r"s_{i,t}=\log p(a_{i,t}\mid x_i,\tau_{i,<t},y^*)-\log p(a_{i,t}\mid x_i,\tau_{i,<t})", r"\widetilde A_{i,t}=w_{i,t}A_i"],
            },
        ],
        "bottom": "Bayesian 视角把 Teacher gap 从“模仿误差”改写成“当前动作对成功结论提供了多少证据”。",
    },
    {
        "number": "09",
        "title": "递归与层级 Credit Assignment",
        "subtitle": "AgentOPSD 用 belief revision 分 turn；CrEST 让 verifier 决定方向、Teacher 只教幅度",
        "methods": [
            {
                "name": "AgentOPSD",
                "paper": "Recursive Self-Distillation for Agentic Reinforcement Learning",
                "arxiv": "2608.05987v1",
                "image": "agentopsd.png",
                "tag": "Recursive belief",
                "lead": "先将 token-level privileged gap 聚合为 turn evidence，再递归更新成功 belief。相邻 belief revision 作为每个 turn 的局部 credit。",
                "bullets": ["初始 belief 来自 group success rate。", "最终 turn advantage 与 sequence outcome 符号对齐。"],
                "equations": [r"e_k=\log\frac{\pi_\theta(a_k\mid s_k,c^+)}{\pi_\theta(a_k\mid s_k)},\qquad \Delta B_k=B_k-B_{k-1}"],
            },
            {
                "name": "CrEST",
                "paper": "Teach the Magnitude, Not the Direction",
                "arxiv": "2608.13179v2",
                "image": "crest.png",
                "tag": "Verifier-anchored hierarchy",
                "lead": "Teacher disagreement 不应决定更新方向。CrEST 让 environment verifier 提供 turn-level sign，self-teacher 只调节 token-level magnitude。",
                "bullets": ["Inter-turn credit 来自 verifier。", "Intra-turn magnitude 由 entropy-gated Teacher gap 提供。"],
                "equations": [r"\mathcal A_t=A_{[t]}^{\mathrm{turn}}\phi_t,\qquad \phi_t=1+\lambda_t^{\mathrm{eff}}(w_t-1)"],
            },
        ],
        "bottom": "方向与幅度解耦，是减少 privileged Teacher 误导风险的一条清晰设计原则。",
    },
    {
        "number": "10",
        "title": "TASPO：Outcome-Constrained Process Supervision",
        "subtitle": "用 verified successful siblings 构造 trajectory-aligned PI，同时守住 outcome advantage",
        "methods": [
            {
                "name": "TASPO",
                "paper": "Reconciling Process Supervision with Outcome-Based Credit in Agentic Policy Optimization",
                "arxiv": "2608.31077v2",
                "image": "taspo.png",
                "tag": "Outcome-constrained redistribution",
                "lead": "TASPO 从同任务的 verified successful siblings 提取与当前 trajectory 对齐的 privileged information，在 action level 计算 likelihood shift，并把它转换为有界、均值为 1 的 advantage weights。",
                "bullets": [
                    "权重始终为正，不翻转 outcome advantage 的符号。",
                    "每条 trajectory 上权重均值为 1，不改变总体 advantage budget。",
                    "局部 process supervision 只重新分配 credit，不覆盖 verifier 给出的成败结论。",
                ],
                "equations": [
                    r"\widetilde A_t^i=A_iw_t^i,\qquad 1-\epsilon_w\le w_t^i\le1+\epsilon_w,\qquad \frac1{T_i}\sum_t w_t^i=1",
                    r"w_t^i=1+\eta(q_t^i-\bar q_i)",
                ],
            }
        ],
        "bottom": "发展脉络：局部 KL → step reliability → state coverage → dynamic PI → Bayesian/recursive credit → outcome-constrained redistribution。",
    },
]


CSS = r"""
@page { size: 210mm 280mm; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #dfe4e2; color: #202b2f; }
body { font-family: "PingFang SC", "Hiragino Sans GB", sans-serif; }
.page {
  width: 210mm; height: 280mm; padding: 12mm 13mm 11mm; background: #f8f8f4;
  page-break-after: always; position: relative; overflow: hidden;
}
.page:last-child { page-break-after: auto; }
.topline { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4mm; }
.series { color: #167f77; font-size: 10px; font-weight: 700; text-transform: uppercase; }
.page-no { color: #8c9797; font: 700 11px/1 Menlo, monospace; }
h1 { margin: 0; font-size: 27px; line-height: 1.18; letter-spacing: 0; }
.subtitle { margin: 2mm 0 5mm; color: #667477; font-size: 13px; }
.methods { display: grid; gap: 4mm; }
.methods.two { grid-template-columns: 1fr 1fr; }
.method { background: white; border: 1px solid #dfe4e1; border-radius: 6px; padding: 4mm; position: relative; }
.method::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2mm; background: #1b8c81; border-radius: 6px 0 0 6px; }
.method-head { display: flex; align-items: center; justify-content: space-between; margin-left: 1mm; }
.method h2 { font-size: 20px; margin: 0; }
.tag { border-radius: 12px; background: #dff1ed; color: #116f68; padding: 1.2mm 2.4mm; font-size: 9px; font-weight: 700; }
.paper-title { margin: 1mm 0 3mm 1mm; color: #7a8587; font-size: 9px; line-height: 1.25; }
.lead { margin: 0 0 2.5mm 1mm; font-size: 12px; line-height: 1.58; }
ul { margin: 0 0 2.5mm 5mm; padding: 0; font-size: 11px; line-height: 1.5; }
li { margin-bottom: 1mm; }
.figure { width: 100%; max-height: 72mm; object-fit: contain; display: block; margin: 2mm auto; background: white; }
.two .figure { max-height: 49mm; }
.formula { background: #f1f5f3; border-left: 2px solid #d59a2c; padding: 2mm 3mm; margin: 1.7mm 0; overflow: hidden; font-size: 11px; text-align: center; }
.source { color: #879091; font-size: 8px; margin-top: 1.5mm; }
.bottom { position: absolute; left: 13mm; right: 13mm; bottom: 8mm; background: #203337; color: white; border-radius: 5px; padding: 3mm 4mm; font-size: 11px; line-height: 1.45; text-align: center; }
.accent { color: #d79b29; }
.two .method h2 { font-size: 17px; }
.two .lead { font-size: 10.5px; line-height: 1.48; }
.two ul { font-size: 9.5px; line-height: 1.42; }
.two .formula { font-size: 9px; padding: 1.5mm 2mm; }
.two .paper-title { min-height: 8mm; }
"""


def method_html(method: dict) -> str:
    equations = "".join(f'<div class="formula">\\[{html.escape(equation)}\\]</div>' for equation in method["equations"])
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in method["bullets"])
    source = f'https://arxiv.org/abs/{method["arxiv"].split("v")[0]}'
    return f"""
      <article class="method">
        <div class="method-head"><h2>{html.escape(method['name'])}</h2><span class="tag">{html.escape(method['tag'])}</span></div>
        <div class="paper-title">{html.escape(method['paper'])}</div>
        <p class="lead">{html.escape(method['lead'])}</p>
        <ul>{bullets}</ul>
        <img class="figure" src="../assets/agentic-opd-reconstructed/{method['image']}" alt="{html.escape(method['name'])} official figure">
        {equations}
        <div class="source">Primary source: <a href="{source}">{html.escape(method['arxiv'])}</a> · official paper figure and equations</div>
      </article>
    """


def build_html() -> str:
    sections = []
    for page in PAGES:
        methods_class = "methods two" if len(page["methods"]) == 2 else "methods"
        methods = "".join(method_html(method) for method in page["methods"])
        sections.append(
            f"""
            <section class="page">
              <div class="topline"><span class="series">Agentic On-Policy Distillation · 2026.10</span><span class="page-no">{page['number']} / 10</span></div>
              <h1>{html.escape(page['title'])}</h1>
              <div class="subtitle">{html.escape(page['subtitle'])}</div>
              <div class="{methods_class}">{methods}</div>
              <div class="bottom">{html.escape(page['bottom'])}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agentic OPD 发展脉络（2026.10）重制版</title>
  <style>{CSS}</style>
  <script>window.MathJax = {{tex: {{inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }}, svg: {{fontCache: 'global'}}}};</script>
    <script src="../assets/agentic-opd-reconstructed/mathjax-tex-svg-full.js"></script>
</head>
<body>{''.join(sections)}</body>
</html>"""


def build_markdown() -> str:
    lines = [
        "# Agentic On-Policy Distillation 发展脉络（2026.10）重制版",
        "",
        "> 本文档根据所给长图的内容结构重新制作。方法图来自对应论文官方 arXiv source，公式依据论文原式重新排版；中文总结为结构化重写，不是原图影印。",
        "",
    ]
    for page in PAGES:
        lines.extend([f"## {page['number']} · {page['title']}", "", page["subtitle"], ""])
        for method in page["methods"]:
            lines.extend(
                [
                    f"### {method['name']}",
                    "",
                    f"**论文：** [{method['paper']}](https://arxiv.org/abs/{method['arxiv'].split('v')[0]})（{method['arxiv']}）",
                    "",
                    method["lead"],
                    "",
                    *[f"- {item}" for item in method["bullets"]],
                    "",
                    f"![{method['name']} 官方方法图](../assets/agentic-opd-reconstructed/{method['image']})",
                    "",
                ]
            )
            for equation in method["equations"]:
                lines.extend(["$$", equation, "$$", ""])
        lines.extend([f"> **本页结论：** {page['bottom']}", "", "---", ""])
    return "\n".join(lines)


def render_pdf() -> None:
    if not CHROME.exists():
        raise FileNotFoundError(f"Google Chrome not found at {CHROME}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with tempfile.TemporaryDirectory(prefix="agentic-opd-chrome-") as profile:
        process = subprocess.Popen(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                HTML.resolve().as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            target = None
            document_url = HTML.resolve().as_uri()
            for _ in range(100):
                with suppress(requests.RequestException, ValueError):
                    targets = requests.get(f"http://127.0.0.1:{port}/json", timeout=0.2).json()
                    target = next((item for item in targets if item.get("url") == document_url), None)
                    if target:
                        break
                time.sleep(0.1)
            if not target:
                raise RuntimeError(f"Chrome document target did not become ready: {document_url}")

            connection = websocket.create_connection(
                target["webSocketDebuggerUrl"],
                timeout=60,
                http_proxy_host=None,
            )
            next_id = 0

            def command(method: str, params: dict | None = None) -> dict:
                nonlocal next_id
                next_id += 1
                connection.send(json.dumps({"id": next_id, "method": method, "params": params or {}}))
                while True:
                    response = json.loads(connection.recv())
                    if response.get("id") == next_id:
                        if "error" in response:
                            raise RuntimeError(f"Chrome DevTools error: {response['error']}")
                        return response.get("result", {})

            command("Page.enable")
            command("Runtime.enable")
            mathjax_available = False
            for _ in range(200):
                check = command(
                    "Runtime.evaluate",
                    {
                        "expression": "typeof MathJax !== 'undefined' && Boolean(MathJax.startup && MathJax.startup.promise)",
                        "returnByValue": True,
                    },
                )
                if check.get("result", {}).get("value") is True:
                    mathjax_available = True
                    break
                time.sleep(0.1)
            if not mathjax_available:
                raise RuntimeError("MathJax did not initialize")
            ready = command(
                "Runtime.evaluate",
                {
                    "expression": "Promise.all([MathJax.startup.promise, document.fonts.ready, Promise.all([...document.images].map(image => image.decode().catch(() => null)))]).then(() => true)",
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            if ready.get("result", {}).get("value") is not True:
                raise RuntimeError(f"Document assets did not become ready: {ready}")
            printed = command(
                "Page.printToPDF",
                {
                    "printBackground": True,
                    "preferCSSPageSize": True,
                    "displayHeaderFooter": False,
                    "marginTop": 0,
                    "marginBottom": 0,
                    "marginLeft": 0,
                    "marginRight": 0,
                },
            )
            PDF.write_bytes(base64.b64decode(printed["data"]))
            connection.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def build() -> tuple[Path, Path, Path]:
    if missing := [method["image"] for page in PAGES for method in page["methods"] if not (ASSETS / method["image"]).exists()]:
        raise FileNotFoundError(f"Missing official figure assets: {missing}")
    MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    HTML.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN.write_text(build_markdown(), encoding="utf-8")
    HTML.write_text(build_html(), encoding="utf-8")
    render_pdf()
    return MARKDOWN, HTML, PDF


if __name__ == "__main__":
    for path in build():
        print(path.relative_to(ROOT))