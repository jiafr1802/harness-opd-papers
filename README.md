# Harness & On-Policy Distillation Papers

围绕语言模型 Agent 的 harness、on-policy distillation 和反馈学习整理的论文集。

## 总体脉络

这些工作共同讨论两个问题：Agent 外部系统如何组织状态、工具与反馈，以及模型如何从在线轨迹中学习这些能力。HAD、ActionOPD 和 VPD侧重蒸馏信号；HERD 与 GUI-HARVEST侧重 harness 的积累和演化；Echo 则把真实用户修正转化为持续训练数据。

## 论文目录

- [HAD: Harness-Aware Distillation for Small Language Model Agents](HAD-Harness-Aware%20Distillation%20for%20Small%20Language%20Model%20Agents.pdf)：对比同一教师在有/无 harness 时的行动，显式训练小模型利用 harness。
- [ActionOPD](Action-OPD.pdf)：以语义 action 而非 token 为蒸馏单位，无需教师 logits 或相同 tokenizer。
- [OPHSD: On-Policy Harness Self-Distillation](OPHSD.pdf)：用 harness terminal context 形成 privileged teacher，在 student rollout token 上蒸馏程序性推理模式，并讨论动态外部信息的不可蒸馏边界。
- [Harness Policy Distillation](HARNESS%20POLICY%20DISTILLATION.pdf)：回放 reference trajectories 恢复 runtime control semantics，用 control document 与 harness-verified actions 联合训练 native/injected/control 三种视图，使模型在原 harness interface 下更符合执行契约。
- [HERD: Harness-Evolving Recursive Distillation](HERD-Harness-Evolving%20Recursive%20Distillation%20for%20Continual%20Learning.pdf)：联合演化外部 harness 与参数能力，以缓解持续学习中的遗忘。
- [GUI-HARVEST](GUI-Agent-Harness-Evolution.pdf)：根据 GUI 执行前后截图和重复运行证据自动改进 harness。
- [VPD: Variational Policy Distillation](VPD-Learning%20from%20Language%20Feedback%20via%20Variational%20Policy%20Distillation.pdf)：以变分 EM 共同更新教师和学生，从语言反馈获得动态监督。
- [Echo](Echo-Learning%20from%20Experience%20Data%20via%20User-Driven%20Refinement.pdf)：不涉及 harness 或 OPD，核心是收集用户与 Agent 交互中的 refinement 数据并用于 SFT；与本仓库主线关系较弱，简要了解即可。

## 阅读笔记

- [Harness 定义与 Formulation 对照](notes/harness-formulation-comparison.md)：统一抽象 HAD、OPHSD、HERD 与 GUI-HARVEST 对 harness 的不同定义、数学表示、更新对象和部署形态。
- [HAD 通用阅读笔记](notes/HAD-reading-notes.md)：论文主线、实验结果与方法边界。
- [HAD 针对性理解](notes/HAD-specific-clarifications.md)：重点解释 step/token、on-policy/SFT、KL 方向、两类 reasoning 前缀、preference loss 与有效性检查器。
- [ActionOPD 阅读笔记](notes/ActionOPD-reading-notes.md)：语义 action、黑箱教师、state-group rollout 与 GRPO-style 优化。
- [OPHSD 阅读笔记](notes/OPHSD-reading-notes.md)：paired prompt/token KL、Memory Bank draft–verify、流程与动态内容的蒸馏边界及 pseudo-case 风险。
- [HERD 阅读笔记](notes/HERD-reading-notes.md)：harness evolution、递归蒸馏、双锚点漂移约束与 TRACE+17。
- [GUI-HARVEST 阅读笔记](notes/GUI-HARVEST-reading-notes.md)：重复 GUI 证据、跨任务行为模式、harness 源码修改和双门验证。
- [VPD 阅读笔记](notes/VPD-reading-notes.md)：稀疏奖励瓶颈、Variational EM、teacher refinement、student distillation 与语言反馈边界。
- [LensVLM → Harness-Augmented OPD Research Proposal](notes/LensVLM-Harness-OPD-proposal.md)：将压缩视觉上下文的按需展开改造成 Teacher-only Harness，提出无工具 Student 的 OPD 训练方法、对照实验、评测指标及信息丢失边界。
- [LensVLM Harness-OPD 研究计划（中文版）](notes/LensVLM-Harness-OPD-proposal-zh.md)：中文完整研究方案，包含两张工作流图、OPD 公式、实验矩阵、消融、评测及分阶段实施方案。
- [PRISM：可迁移性驱动 Harness-OPD 研究计划（中文）](notes/PRISM-transferability-aware-harness-distillation-proposal-zh.md)：区别 Harness 的特权收益与 Student 可迁移收益，提出反事实证据干预、可观察性估计、动态 OPD 加权与可选的 Distillation-Aware Harness Optimization，并给出完整实验及投稿方案。
- [LensVLM 论文阅读笔记（中文）](notes/LensVLM-reading-notes.md)：包含作者 Figure 2 原图、压缩/Expand 三种返回形式、Hard Sample 数据、SFT+DAPO 训练与结果、对 Harness-OPD 的启发。
