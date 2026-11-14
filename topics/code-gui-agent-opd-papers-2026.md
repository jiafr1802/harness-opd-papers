# Code / GUI Agent × OPD：在线蒸馏、特权 Harness 与可执行 Benchmark

> 截止：2026-11-14。目标不是罗列所有 agent training，而是筛选：**(a) Code/GUI 交互式任务，(b) Student 实际访问的状态 / prefix 上的蒸馏，(c) 明确的 Teacher privilege 或多教师，(d) 能在 SWE-bench / OSWorld / AndroidWorld 等 benchmark 评测**。  
> 配套专题：[Agent × Harness 文献](code-gui-agent-harness-papers-2026.md)。已有：[OPRD + Harness 研究差距](../notes/Complex-Harness-OPD-and-Code-Harness-landscape-zh.md)、[GUI Harness → OPD 候选](../notes/GUI-Agent-Harness-OPD-candidate-map-zh.md)。

## 0. 核心判断

1. **严格 Code Agent OPD 已经有**：OPRD（reminder-conditioned reverse KL，SWE-bench 家族）、CAMP-II（长程 file memory + frozen teacher OPD，Terminal-Bench 2.1 / SWE-bench Pro V2）、Logics-SWE 的 RRE-MOPD（多类别专家 on-policy 融合，SWE-bench Pro V2 / Multilingual）。
2. **严格 GUI Agent OPSD 已经有**：GUI-SD-v2（多步 GUI reasoning + memory guidance，AndroidWorld / MobileWorld）、ComputerSD（实时 transition feedback → privileged teacher + value gate，OSWorld-Verified）；Gated Hindsight Distillation 以未来截图作 privilege，环境数据来源与完整 online GUI rollout 要区分。
3. **复杂 Harness 的整体内化尚未被这些文章证明**：大多数严格 OPD 只额外给 teacher reminder、future screenshot、step analysis、file-memory guidance 等；并没有把 Manager–Executor–Auditor、GUI+CLI executors、独立审计/权限系统在部署时完全消掉。
4. **最接近完整 Harness 内化的是 Harness-Zero**：teacher 使用演化后的 tool/middleware/skill/memory Harness，转换成学生目标动作并在实时 student trial 里 review；但参数更新是 **SFT**，不是 token-level on-policy KL。
5. **OPCD 词义警戒**：On-Policy Context Distillation 常指训练时 teacher 看到 privileged context、student 看不到并在线对齐。找到 [D-OPCD](https://arxiv.org/abs/2610.07250) 将图像生成 Agent 改写 prompt 的能力蒸馏到 diffusion model，但实验是 **text-to-image（GenEval 等）**，不是 Code/GUI 操作 Agent；本专题不将其算作 Code/GUI 命中。下文强调实际损失而非按缩写归类。

## 1. 纳入标准及标签

- **Strict OPD**：Student 自己生成可执行 trajectory / sampled tokens；Teacher 在相应 Student prefix/state 上重评分；优化明确的 policy/logit/token（例如 reverse KL），而不只是离线成功轨迹 SFT。
- **OPSD**：自蒸馏，通常是同权重或参数共享的 privilege-conditioned teacher；**MOPD**：多个教师/专项策略协同；**Context-distillation-like**：teacher 使用部署不可见信息。几种标签可以重叠。
- **On-policy intervention + SFT**：采样是 live student 状态，teacher 纠错后用行为模仿更新；**不擅自升级为 strict OPD**。
- **Offline distillation**：固定 teacher trajectories 训练，无学生当前状态同步；放在近邻对照。
- Harness **被保留** 与 **部署移除** 必须分别注明。一个 student 没有 reminder，**不代表**其底层 shell/GUI/coding runtime 被内化。
- 分数严格绑定 paper version、benchmark subset 和评测 scaffold，不做跨论文排行榜。

## 2. 严选总表：优先级、loss、benchmark、可蒸馏范围

| 优先 | 论文 | 场景及 benchmark | Student / Teacher 与监督信号 | 部署时保留什么 | 分类 |
| --- | --- | --- | --- | --- | --- |
| **P0** | [OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents](https://arxiv.org/abs/2610.21496) (2026-10) | **SWE-bench Verified / Pro / Multilingual** | Reminder-free coding rollout；成功自模仿，失败由 global/turn-local reminder teacher 同前缀 reverse KL | coding agent tools / runtime；**移除 reminder** | **严格 Code OPD** |
| **P0** | [CAMP-II: Coding Agent Memory Post-Training II](https://arxiv.org/abs/2611.14009) (2026-11-10) | **Terminal-Bench 2.1 / SWE-bench Pro V2**（外部分布测试）+ CAMG-II 五类环境 | Bash/file memory agent online rollout；冻结 Qwen3.8-27B teacher，sampled-token reverse KL、异步训练 | Bash、文件系统、上下文替换机制；学会如何写/取记忆 | **严格 Code/Memory OPD** |
| **P0** | [One to More, More to One: Category-Aware Iterative Expert Training for SWE Agents (RRE-MOPD)](https://arxiv.org/abs/2609.23377) (2026, v2) | **SWE-bench Pro V2 / Pro / Multilingual** | 3 个类别专家，label routing + multi-teacher on-policy distillation；ReLU-gated teacher 改善方向 | coding runtime；移除推理时多专家路由 | **严格 SWE MOPD** |
| **P0** | [GUI-SD-v2: Learn How to Act from Your Own Interactions](https://arxiv.org/abs/2609.27307) (2026) | **AndroidWorld / MobileWorld** | 同 GUI state 有/无 privilege 双分支训练，选取 step-specific reasoning & memory 作为自教师指导 | 原 GUI action DSL；部署无 training-only privilege | **严格多轮 GUI OPSD** |
| **P0** | [ComputerSD: Online Self-Distillation from Real-Time Feedback](https://arxiv.org/abs/2609.40253) (2026) | **OSWorld-Verified** | Student 执行动作后 GUI analyzer 产出 feedback/step value；value-gated token OPSD + trajectory GRPO；异步在线 | GUI runtime；移除训练时 analyzer guidance | **严格 GUI OPSD + RL** |
| **P0** | [When Token Supervision Fails: Segment-Level OPD (CHOP)](https://aclanthology.org/2026.aacl-long.167/) (AACL-IJCNLP 2026) | **SWE-bench Verified，固定 OpenHands scaffold**；另有 AIME | 按 token reliability 路由 reverse KL / segment sliced-Wasserstein；仅在有反事实作用时使用 privilege | OpenHands runtime，部署只可用 deployable PI | **严格 long-horizon OPD；非专属 Harness 蒸馏** |
| **P1** | [The Next Screenshot Knows: Gated Hindsight Distillation](https://arxiv.org/abs/2608.06065) (2026) | **AndroidWorld / AndroidLab** | 自教师多看 next screenshot；只有 student 错且 teacher 恢复动作时才蒸馏 | 原 GUI action；部署不能看 future screenshot | **同前缀 hindsight OPSD；主要基于已有轨迹** |
| **P1（非 GUI/Code 核心）** | [State-Matched Routing and Contextualized Self-Distillation (SMRC-SD)](https://arxiv.org/abs/2608.05219) (2026) | **ALFWorld / WebShop** | 验证 student 当前 state 是否对应 successful reference；仅 matched turns 注入 teacher context | 原交互工具 | **严格多轮特权蒸馏；跨领域重要机制** |
| **P1（强相关非严格）** | [Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974) (2026) | **SpreadsheetBench / AppWorld / USPTO** | 已演化复杂 harness 指导独立 agent 审核与替换 student action；accepted rollout 再 SFT | 极简目标 harness；移除专用源 harness | **live intervention + SFT，非 token OPD** |
| **P2（离线对照）** | [OPEN-SWE-TRACES: Advancing Dual-Mode Multilingual Distillation for SWE Agents](https://aclanthology.org/2026.findings-emnlp.131/) (EMNLP 2026 Findings) | **SWE-bench Verified / Multilingual / Pro** | OpenHands / SWE-agent 大量 expert trajectories；训练 coder 模型的监督微调 | 原 coding scaffold | **离线 trajectory distillation，非 OPD** |

## 3. Code Agent：逐篇评述（优先看 P0）

### 3.1 OPRD — 直接命中 SWE-bench 的严格 OPD

[论文](https://arxiv.org/abs/2610.21496) · [仓库已写的细节与复现风险](../notes/Complex-Harness-OPD-and-Code-Harness-landscape-zh.md)

**核心机制**：Student 在真实 repo 环境中不带 reminder 在线 rollout；可执行 verifier 判定终局成功/失败。成功轨迹做 autoregressive self-imitation；失败轨迹带两类训练期 privilege：
- **Global Reminder** 把最终 test diagnostics 变为全程的约束/纠错提醒。
- **Turn-local Reminder** 仅在特定工具 observation 后作用于下一次 action。
- 同一个 student-generated token prefix 被 teacher 带 reminder 重新评分，使用 **reverse KL（论文的 Top-100 近似）**。
- **重要区分**：原文明确评测使用 Claude Code scaffold；并未充分公开训练期 tool schema、action serialization 或环境代码。Teacher 不另执行一条完整 coding trajectory，而是重评分学生轨迹。

**报告结果**：Qwen3.5-4B 在 Verified / Pro / Multilingual 的 benchmark 平均相对 base **+11.3pp**；诊断 error-event **446→222**。不要把平均值当成某个单独 benchmark 的增益。

**与复杂 Harness 的关系**：OPD 内化的是 reminder 引起的 context use / error recovery，不是删除 shell、editor、test runner。复现优先用 [R2E-Gym](https://arxiv.org/abs/2504.07164) 或 [SWE-smith](https://arxiv.org/abs/2504.21798) 固定动作协议；这只是本仓库的实验建议，不是原文给出的开源 OPRD 训练实现。

### 3.2 CAMP-II — 将长期文件记忆变成学生策略的一部分

[论文](https://arxiv.org/abs/2611.14009) · [项目前身（PPO 不是 OPD）](https://arxiv.org/abs/2609.34422)

- **Motivation 很具体**：定制 memory tool / compaction protocol 不在 teacher、student 的已有接口先验中；Student 不会产生有效 memory action，Teacher 也不会准确评判。
- **Harness/action**：普通 Bash + episode-persistent workspace，让 agent 用 create/edit/search/read 文件在 context reset 后恢复需要的状态。Memory **仍由外部文件保存**，被学习的是写什么、何时读、怎样组织，而不是将每个任务的动态信息固化进参数。
- **训练**：Frozen **Qwen3.8-27B teacher**，学生 Qwen3.5 4B / 9B，多环境完全异步 **sampled-token reverse-KL OPD**；CAMG-II 包括 Shop、Coding、AutoResearch、General、Visual。原文指出可复用的工具接口先验是 teacher 可纠错性的前提。
- **外部迁移**：SWE-bench Pro V2，Qwen3.5-4B **11.2% → CAMP-II-4B 18.1%**；Qwen3.5-9B **23.1% → CAMP-II-9B 30.1%**。同一表的 Terminal-Bench 2.1 也提升。注意 Pro **V2** 不是 Pro 原版。
- **创新价值**：极适合研究 Context/Memory Harness 的 *learnable policy* 与 *non-distillable external storage* 分界；也适合接复杂 Harness 的 audited state 或 episodic checkpoint。

### 3.3 Logics-SWE / RRE–MOPD — 多专家融合，有真实 SWE 基准

[论文 v2](https://arxiv.org/abs/2609.23377) · [阿里官方项目与环境/模型](https://github.com/alibaba/AgenticBigBang)

- **训练多个专家**：基于可执行 SWE 任务，SWE Labeler 产生类别/领域标签；三类 same-origin category experts 通过 Agentic-miniRL + Refresh–Repair–Expand 循环，复用自身验证成功轨迹并重选训练任务。
- **融为单模型**：按类别路由对应 teacher 的 on-policy guidance，使用 **ReLU-gated reward extrapolation** 保留教师相对 reference 的改善方向，避免非专长教师的负向干扰。
- **评测（注意 v2 扩展）**：作者报告 27B single student 在 **SWE-bench Pro V2 86.19%**、**SWE-bench Multilingual 71.78%**，相对相应 base **+4.57pp / +3.89pp**。原论文 Pro-618（618 题子集）报告 58.04%，**不能与 Pro V2 的 86.19% 放在同一列比较**。
- **与 Harness 的关系**：工具环境和可执行 verifier 提供真实 coding 反馈，但被迁移的是 **多个训练后专家的 domain policy**；不是将一个更复杂的执行 Harness 整体压缩。若要做 Harness × MOPD，应同时固定 task-label routing 与每个 teacher 的可用工具/上下文。

### 3.4 CHOP — 长程 Code Agent 上 token KL 不一定有效

[正式会议论文（AACL 2026）](https://aclanthology.org/2026.aacl-long.167/)

- **动机**：长期交互中 Teacher 仍能区分好坏序列，但局部 token-level 蒸馏的梯度信号变弱；不能默认所有状态都该用 KL。
- **机制**：按 state reliability 选择 token-level reverse KL 或 segment-level sliced-Wasserstein；另通过 teacher 有无 PI 时的 counterfactual distribution change 判定是否启用 privileged information。
- **SWE-bench Verified**：相同 OpenHands harness，off-the-shelf Qwen3-4B **11.8%**、标准 OPD **6.5%**、**CHOP-deployable 13.2%**；作者明确说 **13.2% 相对 11.8% 不是显著增益**，但防止了 standard OPD regression。**CHOP-oracle 14.2%** 使用 held-out 测试时不可部署的 oracle PI，仅能作为诊断上限。
- **研究意义**：复杂 Harness 产生的 teacher guidance 可能是 **整段 action / edit / tool plan 层级**的，而非每个 token 的局部梯度。可考虑 action/segment-level 蒸馏与学生状态可靠性路由，而不是盲目强制 token KL。

## 4. GUI Agent：逐篇评述

### 4.1 GUI-SD-v2 — 从 GUI grounding 推进到 multi-turn reasoning + memory

[论文](https://arxiv.org/abs/2609.27307)

- 相比只预测坐标的 GUI-SD，GUI-SD-v2 在 Android app 的真实多步执行中加入 step-specific **reasoning** 和跨步 **memory** guidance。
- **Stage 1**：同一 GUI state 对比有/无 privilege 的 rollouts，先提升 teacher 的 privilege-following 能力（PAPO）；否则「Teacher 多看一点信息却不会正确使用」使 KL supervision 有偏。
- **Stage 2**：选择有效的 step-level guidance，按 Student on-policy execution states 自蒸馏，部署时只保留原 GUI policy。
- **原文 Benchmark**：AndroidWorld、MobileWorld，报告 Pass@1 和 Pass@3 的改进。**不要误说做了 OSWorld 或训练了整套可执行 GUI Harness**。
- **启发**：Harness 提供的状态追踪、历史摘要与计划信息，必须先验证 *teacher actually follows privilege*，再送进 OPD。

### 4.2 ComputerSD — 把真实 GUI transition 转换为可门控的 Teacher privilege

[论文](https://arxiv.org/abs/2609.40253)

- **Student**：在线 CUA rollout，执行 GUI action，得到真实 after-state。
- **辅助反馈系统**：微调的 GUI analyzer 读动作及执行后的 transition，产生 step-specific guidance 和 value-like score；这些是训练期可见、部署不可见的 privileged context。
- **优化**：guided teacher 对 Student sampled tokens 做 OPSD；**step value gate** 控制老师带 guidance 后概率上升/下降是否与该步骤正确性一致；与完整轨迹的 GRPO 联合，使用异步训练。
- **OSWorld-Verified**：相对 outcome-only GRPO，在 Qwen3-VL-8B-Thinking 和 EvoCUA-8B 两个 backbone 上分别 **+1.9pp、+4.1pp**。
- **对 Harness × OPD 的启发**：给 GUI-HARVEST / OS-Symphony 式 transition critic 一个明确的数据通道：**先让 Student 执行，再用证据生成 supervision**。但 after-state 是训练时 hindsight，不能在执行前作为部署 Student 的实时输入；需要处理时序和数据泄漏。

### 4.3 Gated Hindsight Distillation (GHD) — 下一张截图的信息增益

[论文](https://arxiv.org/abs/2608.06065)

- Student 只看执行前 GUI prefix；parameter-sharing Teacher 多看一张 **next screenshot**，在 Student 自己抽到的 response prefix 上重新评分。
- **Gating**：仅在 Student 当前预测失败且 hindsight teacher 能恢复示范 action 时施加蒸馏；避免无意义或错误 teacher signal。
- **评测**：AndroidWorld、AndroidLab；优于 GRPO baseline。原论文主要讨论从已有 interaction trajectories 提炼 future evidence，**不要把它说成完整异步在线环境训练**。
- **界限**：future screenshot 必然含已执行动作后的信息，它可以形成训练 signal，不能当作 Student 部署前的可用观测。

### 4.4 SMRC-SD — 多轮状态漂移时 Reference Guidance 会伤害模型

[论文](https://arxiv.org/abs/2608.05219) · [代码](https://github.com/liujunzhuo/SMRC-SD)

- Student 的某一步可能进入与成功参考轨迹不同的环境状态；完整 gold reference 此时会推荐不合理动作。
- **State-matched routing** 只在 Student state 能被参考轨迹对应状态支持时蒸馏，再构建 state-conditioned guidance。
- 原文在 ALFWorld、WebShop 上验证，不是 GUI screenshot benchmark；重要的是 **state alignment 的方法逻辑**，可迁移到 AndroidWorld/OSWorld 的 fork-point / recovery critic 实验，但那属于待检验研究假设。

## 5. 非严格 OPD 的近邻：不要误报为 OPD

### Harness-Zero（复杂 Harness 内化，但 loss 是 SFT）

[论文](https://arxiv.org/abs/2609.24974)

先演化 tools、middleware、skills、memory 的完整 task-specific Harness；训练期另一个 harnessing agent 在 Student 正在执行的状态上 **pass/replace** 学生 response，要求替换动作属于目标 Harness 的动作空间；只接受可执行 response 后用 **LoRA SFT**。在 SpreadsheetBench / AppWorld / USPTO 上宏平均 **23.3→44.3%**（源演化 Harness attached 时为 41.7%）。部署移除复杂源 Harness，但保留最小目标 Harness。**它是更复杂的 source Harness 与 target Harness action space 差异的关键先例，不能称严格 token-level OPD，也没有 SWE-bench 结果。**

### OPEN-SWE-TRACES（强数据，离线监督）

[EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.131/)

作者基于 OpenHands + SWE-agent 轨迹产出 **207,489** 个 agent trajectories，涵盖 **9 种编程语言**；蒸馏后模型报告 SWE-bench Verified / Multilingual / Pro 结果。这是**离线 trajectory SFT / behavioral distillation**，可以作为 OPD 的 teacher-trajectory baseline，不满足 Student-state on-policy 匹配。

### D-OPCD（名称最像，但 domain 不命中）

[论文](https://arxiv.org/abs/2610.07250) · [作者代码](https://github.com/Yummytanmo/D-OPCD-CoEvolution)

Agent 用 memory / skills / iteration 生成更好的 **text-to-image prompt**，EMA teacher 额外看改善后的 prompt，student 只看原 query，在 student 自己的 **diffusion denoising trajectories** 上完成 context distillation。评测 GenEval / GenEval2 / WISE / R2I；**不是 code-action/GUI-interaction agent，不能拿来声称已经在 SWE-bench/OSWorld 实现 OPCD**。它在「Harness 提示收益→模型参数→Harness 重新演化」的思路上值得借鉴。

## 6. 统一比较：Teacher 到底比 Student 多了什么？

| 方法 | Teacher privilege | Teacher 是单独运行 Agent 吗 | 在学生状态上对齐吗 | 监督对象 | 移除什么 |
| --- | --- | --- | --- | --- | --- |
| OPRD | final test diagnostics → global/turn reminder | **否**，同 token prefix 重评分 | 是 | token reverse KL；成功自模仿 | reminder |
| CAMP-II | frozen model 的 file-operation / memory competence | teacher 重评分 student actions | 是 | sampled-token reverse KL | 外部教师监督；**文件工具保留** |
| RRE–MOPD | domain-specialized expert policy | 多专家提供 on-policy guidance | 是（标签路由） | 多教师相对增益信号 | 在线多专家路由 |
| CHOP | 可部署 call-graph/test info 等 PI；oracle 条件仅作诊断 | 同学生状态的特权评分 | 是 | token KL / segment matching | privileged training signals |
| GUI-SD-v2 | stage/step reasoning、memory guidance | 同状态 guided/unguided 分支 | 是 | token 自蒸馏 | privileged guidance |
| ComputerSD | after-action GUI analysis + step value | analyzer 提供额外反馈，guided rescoring | 是，基于实际 GUI transition | value-gated OPSD + GRPO | analyzer |
| GHD | 下一张 GUI screenshot | 参数共享 hindsight teacher | 学生采样前缀上；数据多来自已有轨迹 | gated on-policy-response distillation | future screenshot |
| Harness-Zero | tool / middleware / skill / memory 全部源 Harness | **是**，独立 harnessing agent 审查学生 action | live student trial | **accepted action 的 SFT** | source Harness + reviewer |
| D-OPCD | Agent 改进后的图像生成 prompt | EMA teacher | diffusion denoising path | diffusion prediction matching | 训练时的 improved prompt |

**最重要的观察**：训练期多给了某些信息，并不表示这部分信息在 Student 的可见历史中可恢复。如果 teacher 的高分来自外部实时检索、新文件状态、另一模型执行/独立审计，应该保留对应工具或明确只研究可迁移决策模式。

## 7. 推荐 benchmark 组合与最小实验路线

### Code：先做状态对齐和执行协议，再扩展复杂 Harness

- **训练环境**：优先 [R2E-Gym](https://arxiv.org/abs/2504.07164) / [SWE-smith](https://arxiv.org/abs/2504.21798) 的隔离容器、shell / editor / test / finish action；具体设置在[仓库笔记](../notes/SWE-bench-training-environments-and-action-spaces-zh.md)。
- **验证**：SWE-bench Verified + Terminal-Bench 2.1（基础）；SWE-bench Pro V2 / Multilingual（更难、检验迁移）；回报必须包含 Pass@1、成功率、token/cost、turns、tool failures、验证/返工次数。
- **teacher ladder**：无 privilege → Global/Turn Reminder（OPRD）→ same-state action verification（Mid-Harness）→ 独立 reviewer → audited state（LongHorizon）；保持 Student 的原动作协议相同，逐层增加 teacher 特权。
- **loss ladder**：offline teacher SFT → student-state reverse KL → outcome-gated KL → segment/action-level routing（CHOP 逻辑）；对比同计算预算，不允许纯粹依赖额外 rollouts 取胜。

### GUI：对准真实执行前/后状态和可验证终局

- **训练/验证环境**：AndroidWorld 是可参数化、可 reset/verifier 的手机任务；OSWorld-Verified 用于 GUI execution；OSWorld 2.0 用于更长的多应用工作流。Web/桌面/移动 task 不是同分布。
- **teacher ladder**：无额外信号 → next screenshot（GHD）→ analyzer + value（ComputerSD）→ same-state candidate selection / critic → cross-turn audited memory。重点衡量 Teacher action 与 Student action 的 **同一 GUI DSL**。
- **关键控制**：同起始快照/同完成 verifier、固定 grounding 模型和 tool schema；分开记录动作是否执行成功（step）、是否完成任务（terminal）、是否进行了真实环境恢复（recovery）。

### 必须做的消融

1. Fixed backbone + fixed harness、只改变 teacher extra information / inference compute。
2. Teacher 有 privilege 但不按 Student state 对齐 vs state-matched；以及正确 vs 打乱的 feedback。
3. Online OPD vs 由 Teacher 成功轨迹得到的 offline SFT；同等 token/rollout/训练 FLOPs。
4. Token KL vs action-level vs segment-level；成功/失败 routing；teacher advantage / value gating。
5. Inference 去掉附加 Harness 后的能力、保留工具后的能力、原始复杂 Harness 的 teacher upper bound。
6. 训练和评测隔离，报告 benchmark 精确版本、subset、scaffold commit、最大步数和 evaluator。

## 8. 严选后的阅读顺序

**做 SWE-bench 的 OPD**：OPRD → CAMP-II → RRE–MOPD → CHOP → Harness-Zero（区分损失）→ [Code Harness 专题](code-gui-agent-harness-papers-2026.md)。

**做 GUI OPSD**：ComputerSD → GUI-SD-v2 → GHD → SMRC-SD（状态对齐思想）→ GUI-HARVEST（权重不变的 Harness 上限）→ LongHorizon（多角色状态、非可蒸馏边界）。

**研究问题**：当高复杂度 Harness 含有审核、外部状态、工具、memory 与恢复策略时，哪些增益属于可重建的 **policy improvement**，哪些来自模型不可访问的 **information / execution / enforcement**？可用 state-matched、value-gated、action-space-aligned 的 OPD 作为识别方法，而不是把所有成功轨迹都当成模型可学知识。

> 本专题是文献证据地图，不把跨论文分数做不公平排名。论文原文链接优先；“代码将公布”不等于已有可复现代码。所有研究延展均明确标为建议，而非论文已验证的结论。
