# Harness & On-Policy Distillation Papers

围绕语言模型 Agent 的 harness、on-policy distillation 和反馈学习整理的论文集。

## 总体脉络

这些工作共同讨论两个问题：Agent 外部系统如何组织状态、工具与反馈，以及模型如何从在线轨迹中学习这些能力。HAD、ActionOPD 和 VPD侧重蒸馏信号；HERD 与 GUI-HARVEST侧重 harness 的积累和演化；Echo 则把真实用户修正转化为持续训练数据。

## 论文目录

- [HAD: Harness-Aware Distillation for Small Language Model Agents](HAD-Harness-Aware%20Distillation%20for%20Small%20Language%20Model%20Agents.pdf)：对比同一教师在有/无 harness 时的行动，显式训练小模型利用 harness。
- [ActionOPD](Action-OPD.pdf)：以语义 action 而非 token 为蒸馏单位，无需教师 logits 或相同 tokenizer。
- [HERD: Harness-Evolving Recursive Distillation](HERD-Harness-Evolving%20Recursive%20Distillation%20for%20Continual%20Learning.pdf)：联合演化外部 harness 与参数能力，以缓解持续学习中的遗忘。
- [GUI-HARVEST](GUI-Agent-Harness-Evolution.pdf)：根据 GUI 执行前后截图和重复运行证据自动改进 harness。
- [VPD: Variational Policy Distillation](VPD-Learning%20from%20Language%20Feedback%20via%20Variational%20Policy%20Distillation.pdf)：以变分 EM 共同更新教师和学生，从语言反馈获得动态监督。
- [Echo](Echo-Learning%20from%20Experience%20Data%20via%20User-Driven%20Refinement.pdf)：从用户对 Agent 输出的真实修正中提取持续学习信号。

## 阅读笔记

- [HAD 核心机制与疑问澄清](notes/HAD-reading-notes.md)
