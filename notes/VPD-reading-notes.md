# VPD 阅读笔记：Learning from Language Feedback

## Summary

VPD 解决的是 sparse outcome reward 与静态 self-teacher 的双重瓶颈。RLVR 只在完整答案通过验证时给出奖励；如果一个 prompt 的所有 rollout 都失败，GRPO 的组内 advantage 会坍缩，无法指出具体错误。SDPO 等 self-distillation 方法利用 compiler error、环境反馈或 critique 构造 dense token supervision，但 feedback-conditioned teacher 只做 zero-shot 解释，从不学习如何更好地使用反馈；随着 student 提升，teacher 的指导能力达到上限。

VPD 把语言反馈学习形式化为 Variational EM。E-step 用 student 已产生的成功/失败轨迹训练 feedback-conditioned teacher，使其更会把诊断文本转成高奖励 token distribution；M-step 再把更新后的 teacher 蒸馏给不看反馈的 student。Teacher 和 student 实际共享同一套权重，只因输入是否附加反馈而表现为两个条件分布。

## Motivation：为什么不能只用 RL 或静态蒸馏

KL-regularized RL 的理论最优策略是 reward-tilted distribution：

```math
\pi^*(y\mid x)
\propto
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
```

它提高高奖励 response 的概率，但 partition function 无法直接计算。标准 RL 用 policy gradient 绕过归一化常数，却重新落回稀疏 reward 的采样问题。

SDPO 则让同一个模型扮演两个角色：student 只看问题 `x`，teacher 看 `x + feedback C`。它在 student 自己的 rollout prefix 上最小化 token-level KL，因此能提供 dense gradient；但 teacher 分支被 stop-gradient，只依赖模型原有的 zero-shot feedback interpretation。Teacher 不会随着任务训练而变成更好的 critic。

VPD 的核心问题是：能否让 teacher 也利用 outcome 学会解释反馈，再将这种改进后的 dense guidance 传给 student？

## Variational EM 视角

VPD 引入参数化 teacher `q_φ(y|x,C)`，作为不可计算最优策略 `π*` 的 variational approximation。训练交替执行两步：

```text
Student rollout
→ 环境产生 outcome reward 与语言反馈 C
→ E-step 更新 feedback-conditioned teacher
→ M-step 将 teacher 分布蒸馏给 bare student
→ 新 student 进入下一轮
```

数学上用 `φ` 和 `θ` 区分两个优化阶段；实现中 `φ = θ`，共享同一个网络。区别只来自 context：teacher 额外看到 feedback，student 只看原始输入。

## E-step：Teacher Refinement

E-step 希望 teacher 接近 reward-tilted optimal policy，其目标等价于：

```math
\max_\phi\;
\mathbb E_{y\sim q_\phi}[r(x,y)]
-\beta D_{\mathrm{KL}}(q_\phi\parallel\pi_\theta)
```

VPD 动态使用当前 student `π_θ` 作为 reference prior，而不是固定 base model。这形成 trust region：teacher 可以朝高奖励模式移动，但不会远离 student 当前可学习的分布。

Teacher 不重新进行昂贵的 on-policy 探索，而是复用 student 已生成的 trajectories 做 off-policy preference optimization。成功 rollout 是 positive，失败 rollout 是 negative。

传统 DPO 要求 chosen/rejected 共享同一个 prompt；但每条 trajectory 会产生独有反馈 `C_i`，因此不同 response 的 teacher context 不相同，不能构造严格配对。VPD 使用 unpaired BCO，将正负样本分别做 binary classification，并用移动 reward shift `δ` 校准边界。

E-step 的作用不是直接训练 student，而是提高 teacher 的 reward margin：看到反馈后，更明确地区分哪些 token distribution 对应成功轨迹、哪些对应失败轨迹。

## M-step：Student Distillation

Teacher 更新后，VPD 在 student 自己的 on-policy rollout prefix 上最小化 reverse KL：

```math
\mathcal L_{\mathrm{M}}
=\mathbb E_{x,y\sim\pi_\theta}
\sum_t D_{\mathrm{KL}}\left(
\pi_\theta(\cdot\mid x,y_{1:t-1})
\parallel
\operatorname{sg}[q_\phi(\cdot\mid x,C,y_{1:t-1})]
\right)
```

Student 推理时不需要反馈，但训练时吸收了 teacher 从反馈中提炼出的 token-level correction。动态 reference prior 保证 teacher 留在 student 附近，使 M-step 比从一个过强、分布断裂的 oracle 蒸馏更稳定。

因为共享权重按顺序更新，E-step 后最初收集的 rollout 对 M-step 严格说已略微 off-policy。论文认为 trust-region 约束使偏移足够小，因此省略 importance sampling，并报告没有可测性能损失。

## 三类 Language Feedback

VPD 在三种诊断反馈上验证：

1. **Environment feedback**：LiveCodeBench 的 compiler error、runtime error 和 failed unit-test assertion。
2. **Sibling rollout feedback**：利用同 prompt 下成功的兄弟轨迹诊断失败轨迹。
3. **LLM self-critique**：由 judge 对 response 生成语言批评。

模型包括 Qwen3-1.7B、Qwen3-8B 和 OLMo3-7B-Instruct；benchmark 覆盖 LiveCodeBench、SciKnowEval、DAPO-Math、Math500、AIME24/25 和 AMC23。Baselines 包括纯 RL 的 GRPO、纯 distillation 的 SDPO，以及 Joint Loss、Advantage Reshaping、Advantage Reweighting 等单阶段混合方法。

在 LiveCodeBench 上，Qwen3-8B base 为 28.05，GRPO 为 45.61，SDPO 为 47.33，VPD 达到 49.62。实验也显示简单把 RL 与 distillation loss 同时相加，并不等价于交替优化 teacher/student 两个角色。

## 边界

VPD 的效果依赖 feedback 是否包含可行动的诊断信息，以及基础模型是否已有一定任务能力。反馈只能帮助 teacher 在已有轨迹附近识别和强化高质量模式，不能凭空创造从未探索到的正确解。

在 cold-start 和高难数学中，VPD 能缓解、推迟 self-distillation collapse，但不能完全突破探索极限。论文最终结论并非“语言反馈始终优于 sparse RL”：当 student 很难采到任何接近正确的模式时，纯环境驱动 RL 最终仍可能更有效。

## 原文定位

主要依据：[VPD 原文](../VPD-Learning%20from%20Language%20Feedback%20via%20Variational%20Policy%20Distillation.pdf) §2、§3.1–3.4、§4.1 和 Appendix A/B。
