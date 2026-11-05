# Reading Is Not Reasoning 阅读笔记：CAPS

## Summary

论文研究 vision–text compression（VTC）在多步 agent 中造成的能力损失。VTC 把不断增长的文本交互历史渲染成图像，用更少的 visual tokens 表示相同内容；但 visual-history agent 明显弱于 text-history agent。作者发现差距并非主要来自 OCR：模型可能能够读出历史事实，却仍在搜索、停止、证据使用和 action 选择上系统性偏移。这被称为 **agentic policy gap**。

CAPS（Cross-modal Agentic Policy Self-distillation）用同一模型的 text-history policy 教 visual-history policy。第一阶段把成功 text trajectories 转成 visual-history SFT 数据；第二阶段联合 visual-history GRPO 与 student-visited states 上的 online forward-KL distillation。目标不是恢复更多文字，而是让压缩界面下的模型学会像 text policy 一样推理和行动。

![论文 Figure 1：视觉历史的主要问题是 policy gap，而不只是 OCR](../assets/caps-figure-1.png)

*图源：原文 Figure 1。*

## Motivation：能读，不等于会用

同一个 interaction history 可以表示为文本 token，也可以通过确定性 renderer 变成页面图像。两种输入包含相同 observation/action 内容，模型架构和 action space 也相同；唯一变化是历史模态。

作者从三层诊断差距：

1. **History recovery**：从真实 trajectory 构造 4,000 个 History QA，只要求定位并提取历史中的 query、document title、keyword 或 year。
2. **Matched-state decision**：在完全相同的语义状态上，只替换 text/image history，比较下一 action。
3. **Full trajectory quality**：比较 evidence uptake、entity/relation tracking、query progression、stopping calibration、answer grounding。

关键结果是：GRPO 将 7B visual agent 的 SearchQA EM 从 14.0 提升到 40.1，但 image History QA 反而从 82.78 降到 78.22。性能提升不是因为 OCR 变好。

在 matched states 上，29.13% 的 action type 不一致；两边都选择 Search 时，只有 5.22% 的 query 高度匹配；AgentOCR 在第一次 retrieval 后立即回答的比例为 55.41%，text policy 仅为 29.80%。视觉 policy 更容易提前停止，也更难形成连续搜索策略。

## 问题形式化

第 `t` 步的 canonical text history 为 `h_t`，确定性 renderer `R` 把它转换为 visual history `I_t`。于是同一个语义状态有两个 paired contexts：

```math
x_t^T=(I,h_t),
\qquad
x_t^I=(I,I_t)
```

其中 `I` 是任务 instruction。Frozen text policy `π_φ^T` 是 source，trainable visual policy `π_θ^I` 是 target。CAPS 希望在对应状态上缩小两者 response distribution 的差距。

这和 Teacher-only external-evidence distillation 不同：text teacher 没有额外答案、原文或 oracle，它只看到与 student 相同的 interaction content，只是表示形式更容易被模型利用。

## CAPS 两阶段方法

![论文 Figure 2：CAPS 的离线轨迹蒸馏、GRPO 与在线 policy distillation](../assets/caps-figure-2.png)

*图源：原文 Figure 2。*

### Stage 1：Offline Trajectory Self-Distillation

作者先训练一个 text-history GRPO policy，并收集其交互 trajectories。只保留完成任务、action 可解析、response 未截断的成功轨迹；同一任务有多个轨迹时，优先选择步数少、重复 action 少、response 和 context 更短的一个。

对每个决策 step，instruction、interaction content 和完整 teacher response 都不变，只把 history 从 text `h_t` 替换成其 rendered image `I_t`：

```text
(text instruction, text history, teacher reasoning + action)
                         ↓ replace history representation only
(text instruction, image history, same teacher reasoning + action)
```

Visual student 对完整 response 做 next-token SFT，因此同时学习 reasoning 与 action。这个阶段提供稳定初始化，但只覆盖 text teacher 访问过的成功状态。

#### NLL 是什么？这一阶段为何叫 Self-Distillation？

Next-token NLL 是 negative log-likelihood，即对 Teacher response 做标准 teacher-forcing cross-entropy：

```math
\mathcal L_{\mathrm{off}}
=-\frac{1}{|y^T|}\sum_k
\log \pi_\theta^I(y_k^T\mid I,I_t,y_{1:k-1}^T)
```

Student 给目标 token 的概率越高，NLL 越小。因此这一阶段的**优化形式就是 SFT**：不读取 Teacher logits，也不计算师生 KL，而是把成功 Teacher sequence 当 hard label。

这里有一个重要术语边界。Text Teacher 已经经过 text-history GRPO，Visual Student 则是随后训练的 target checkpoint；二者当前权重不同，输入模态也不同。因此如果把 self-distillation 严格限定为“同一时刻、同一组权重的两个 context view”，CAPS offline 阶段并不完全符合。

论文采用的是更宽泛的 self-distillation：Teacher 与 Student 来自同一个 Qwen2.5-VL base，架构、规模、tokenizer 和 action space 相同，没有额外更大的外部 Teacher；更强监督来自模型自身在 text interface 下训练出的 policy。更精确的表述应是：

```text
GRPO-trained text checkpoint
→ visual checkpoint 的 cross-modal sequence-level distillation
→ 优化上表现为 Teacher-generated synthetic trajectory SFT
```

所以“self”强调同源模型体系，“distillation”强调把 text policy 的能力迁移到 visual policy，“SFT/NLL”描述实际 loss；三者不是同一维度。

### Stage 2：Visual-History GRPO + Online Distillation

Visual student 随后在图像历史界面中自主 rollout，使用环境 return 做 GRPO。这样优化真实部署分布，但 sparse reward 不能为每个 token 提供明确的策略修正。

因此每当 visual student 到达状态，系统同时保留由**同一批 student actions 与 environment observations**组成的 canonical text history。Student 先采样 response；frozen text teacher 与 visual student 再在同一 student prefix 上计算全词表分布：

```math
p_k^S=\pi_\theta^I(\cdot\mid x_t^I,y_{1:k-1}),
\qquad
p_k^T=\pi_\phi^T(\cdot\mid x_t^T,y_{1:k-1})
```

Online loss 使用 teacher-to-student forward KL，并逐 position clip，避免极大 KL 主导梯度。Student 与 teacher response mask 的交集决定哪些 token 参与；若完整 response 没有合法环境 action，则整段 mask 为 0。

#### Online Teacher 是否独立 rollout？

不会。当前环境状态由 visual student 之前执行的 actions 与 observations 形成；到第 `t` 步后，visual student 先采样完整 response `y_t`，其中可以同时包含 reasoning 与 action。Frozen text teacher 不生成另一条 reasoning/action，也不向环境提交自己的 action，而是作为 sidecar scorer，在每个 **student-generated prefix** `y_{1:k-1}` 上输出全词表 next-token distribution。只有 student response 中解析出的 action 会推进环境。

因此“Teacher 没有继续生成 argument”并不是这里的问题。即使 student 已经生成到 function call 或 JSON argument 中间，Teacher 仍会对该 prefix 下的所有候选 token 给出概率；实际下一个 token 仍由 student policy 采样，KL 只提供分布级训练信号。

真正的失败模式是 student 最终得到截断或不可解析的 tool call。CAPS 在完整 response 层面检查 action：若没有合法可执行 action，就将整段 online-distillation mask 置零。这能避免用 malformed response 计算 KL，却不会直接教会模型如何补全括号或修复 argument；Teacher 在异常 student prefix 上的分布也可能是噪声。工程实现若要求严格结构合法，可额外使用 grammar/JSON-schema constrained decoding，再保留 action validation 与 invalid-response mask。论文方法本身依赖后两者，而不是让 Teacher 接管半截 tool call。

最终目标为：

```math
\mathcal L=\mathcal L_{\mathrm{GRPO}}+\lambda\mathcal L_{\mathrm{online}}
```

GRPO 保留环境 reward 作为主信号；online distillation 在 visual student 自己访问的状态上提供 dense policy guidance。

## 为什么必须两阶段

只做 offline distillation 会遇到 state-distribution mismatch：visual student 一旦执行不同 action，就进入 text teacher 成功轨迹没有覆盖的状态。

只做 online distillation 则缺少稳定的视觉策略初始化，student rollout 质量较低。SearchQA 消融中，两者单独使用均有效，但组合后的 CAPS 最好，说明 offline successful behavior 与 online student-state correction 互补。

## Benchmark 与结果

**SearchQA** 包含 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle；NQ 和 HotpotQA 用于训练，其余用于 OOD generalization。Agent 最多交互四步，每次搜索返回 top-3 passages。

**ALFWorld** 包含六类长程家庭任务。主实验保留完整历史 `H=50`，同时报告短窗口 `H=2`。

模型为 Qwen2.5-VL-3B/7B-Instruct，使用 size-matched text-history policy 作为 teacher。SearchQA 上 CAPS 相比 AgentOCR 提升 5.0 / 3.4 个百分点；full-history ALFWorld 提升 15.6 / 14.5 个百分点。

7B CAPS 在 SearchQA 达到 43.5 average EM，略高于 text teacher 的 43.0；ALFWorld 达到 95.7%，高于 text teacher 的 92.6%。同时相对 matched text-history policy，平均/峰值 memory-context cost 最多降低 63.3% / 83.4%。

## 与 Harness-OPD / PRISM Proposal 的关系

CAPS 可以看作最干净的 **representation-level OPD**：Teacher 与 Student 看到同一语义历史，没有 privileged external facts；差别只来自 text renderer 与 image renderer。它证明即使信息理论上相同，interface change 仍会造成 action policy mismatch。

因此它应成为 LensVLM/PRISM 的核心 baseline：

- **CAPS / OPD-Visual**：Teacher 只有 canonical text history，没有额外原文工具优势；测纯跨模态 policy transfer。
- **Harness-OPD**：Teacher 还能展开原始文本/OCR/高清页；测 privileged evidence 是否提供额外可迁移收益。
- **PRISM**：进一步判断这部分额外 Teacher advantage 是否受 Student 压缩图像中的可见线索支持。

CAPS 没有回答不可见事实能否蒸馏，因为它没有给 Teacher 注入 Student 永远看不到的新信息。它主要消除的是 **policy gap**，不是 **information gap**。

## 局限

CAPS 要求 text teacher 与 visual student 使用相同模型家族、action space 和 compatible vocabulary，才能在 matched prefix 上做 full-vocabulary KL。方法也依赖 deterministic renderer，使 text/image state 能严格配对。

Offline 阶段只学习筛选后的成功轨迹，可能丢失有价值的恢复行为；online 阶段把 text policy 当成固定上界，但 text policy 本身也可能决策错误。视觉压缩方案没有被联合优化，因此论文回答的是“如何对齐 policy”，不是“什么视觉 renderer 最优”。

## 原文定位

主要依据：[Reading Is Not Reasoning 原文](../READING%20IS%20NOT%20REASONING-BRIDGING%20THE%20AGEN-TIC%20POLICY%20GAP%20IN%20VISION%E2%80%93TEXT%20COMPRESSION.pdf) Figure 1–2、§3、§4.1–4.4、§5.1–5.3。
