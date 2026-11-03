# HERD 阅读笔记：Harness-Evolving Recursive Distillation

## Summary

HERD 面向 continual learning，把学习拆成快慢两层：harness 是快速、可编辑但容量有限的外部记忆；模型参数是缓慢、持久的长期记忆。每个 stage 先让 harness 从当前任务中提炼 skill，并保留 skill 仍解释不了的高残差样本；再把“上一 checkpoint + 新 harness”提供的能力蒸馏回同一模型的新 LoRA 中。蒸馏同时受 previous-checkpoint 与 base-model 两个 drift anchor 约束，前者防止单阶段忘掉旧任务，后者防止许多小偏移累积成通用能力崩塌。最终部署只有更新后的模型权重，不携带 harness。

## 一个 stage 的完整流程

第 `t` 阶段从上一模型 `θ_(t-1)` 和 harness `H_(t-1)` 开始。HERD 的 harness 不只有 skill：`H_t = (K_t, R_t)`，其中 `K_t` 是可路由的 skill registry，`R_t` 是有容量上限的 high-residual memory。

Harness evolution 先从当前任务抽取一个有预算的样本窗口 `A_t`。冻结的上一 checkpoint 在不使用新 skill 的情况下回答这些样本；错误样本交给固定的 base model `θ_0` 作为 skill writer，总结出可复用规则。这里不是让当前 student 自己任意改写 harness：skill writer 始终固定为 `θ_0`，避免随着递归训练一起漂移。候选 skill 只有在验证窗口上降低 verified answer 的平均负对数似然，也就是 residual 时才被接受。

蒸馏完成后，HERD 用新模型和新 skill 重新计算旧 memory 与当前窗口中每条记录的 residual，只保留最难预测的完整记录直到预算用完。`R_t` 不直接 replay，也不是训练 target；它主要提供下一阶段 checkpoint-drift constraint 的 probes。也就是说，skill 保存可抽象规律，memory 保存规律和权重仍未覆盖的例外。

## Harness 如何蒸进权重

用户给出的“带 harness 的 student 作为 teacher”方向接近，但更准确的说法是：teacher 和 student 来自**同一个上一 checkpoint 的两个视图**。

Teacher `p_H` 是冻结的 `θ_(t-1)`，看到 routed skill `K_t(x)`、prompt，以及 privileged verified-answer context。Student `p_S` 以相同的 `θ_(t-1)` 为 backbone，挂载新初始化的 LoRA，只看 bare prompt。训练后 LoRA 合并进 checkpoint，得到 `θ_t`；下一 stage 再重复该过程。

这也不是前面 HAD 笔记中“学生 token prefix 上的 reverse-KL OPD”。HERD 在 verified answer 自己的 prefix 上，对 teacher 与 student 的全词表分布做 forward KL：

```math
\mathcal{L}_{\mathrm{dist}}
= \mathbb{E}_{(x,y^*)}\frac{1}{|y^*|}\sum_k
D_{\mathrm{KL}}\left(p_k^H \parallel p_k^S\right)
```

使用 verified-answer prefix 可以避免退化路径：如果改在学生自由采样的错误回复上比较，teacher 与 student 可能在“复述 prompt”等坏行为上恰好一致，从而没有有效梯度。因此 HERD 属于递归 self-distillation / harness internalization，但不能简单等同于标准 reverse-KL OPD。

## 双锚点 drift constraint

HERD 只优化当前 stage 的 LoRA 参数，同时要求两个漂移不超过预算。Checkpoint anchor 使用上一模型 `θ_(t-1)` 和 `R_(t-1)` 中按任务分组的 probes，限制 per-stage drift，保护最近学到的任务；base anchor 使用不可变的 `θ_0` 和固定 general probes `G`，限制 cumulative drift，保护原始通用能力。每个约束取最差 probe group 的平均 KL，避免某一任务或能力的崩塌被总体均值掩盖。

优化通过 dual ascent 自动调节两个乘子：漂移在预算内时约束权重衰减，超预算时权重增加。只用 previous checkpoint 会让许多小偏移长期累积；只用 base model 又保护不了 base 从未学过的新任务，所以两个 anchor 缺一不可。

## Benchmark 与结果

主 benchmark 是 TRACE+17：8 个 TRACE 任务加 17 个 Super-NaturalInstructions 任务，共 25-stage stream。每个任务有 500/100/100 条 train/validation/test 数据，所有方法从 Qwen3.5-4B 开始，并使用相同更新预算。通用能力另用 G-test 监控五类各 200 个样本：MMLU、GSM8K、HellaSwag、IFEval、XCOPA。

指标包括新任务 acquisition、最终平均性能 AA、forgetting、BWT，以及 G-test 的平均变化 `ΔC` 和最差类别变化 `ΔCmin`。Stage 25 时 HERD 的 AA 为 0.706、acquisition 为 0.713、forgetting 为 0.010；G-test 平均下降 0.063，最差类别下降 0.130。它的 acquisition 接近最强参数方法，但 forgetting 和通用能力损失明显更小。消融中，去掉 skill 对 acquisition/AA 伤害最大；uniform memory 使 forgetting 翻倍；去掉 checkpoint anchor 主要增加旧任务遗忘，去掉 base anchor 则使通用能力持续漂移。

## 原文定位

主要依据：[HERD 原文](../HERD-Harness-Evolving%20Recursive%20Distillation%20for%20Continual%20Learning.pdf) §2.2–2.4（stage、harness evolution、drift-constrained distillation）、§3.1–3.3（TRACE+17、结果与消融）及 Figure 1–4。
