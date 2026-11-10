# 复杂 Harness-OPD 与 Code Harness 调研

调研日期：2026-11-08

## 1. 直接结论

这个问题有两个不同的“最复杂”，不能只给一个名字。

1. **若问“被内化进模型权重的源 Harness 最复杂到什么程度”**，目前我找到的上限是 **Harness-Zero**（arXiv:2609.24974）。它的源 Harness 不再只是 prompt 或 plan–solve，而是一个经过三轮演化的完整 tool-using Harness，包含 **tools、middleware、skills 和 memory**。训练时再用一个 harnessing agent，把这些组件的作用逐步翻译成学生原生动作，最后通过 SFT 内化到 Qwen3.5-9B。
2. **若坚持严格的 token-level on-policy distillation 定义**，coding 方向的直接命中是 **OPRD**（arXiv:2610.21496）。它在学生自己的 repository-level coding 轨迹上，对失败轨迹使用 Reminder-conditioned teacher 的 reverse KL。但被移除的是 Global/Turn-local Reminder，不是底层 coding agent runtime；论文只在评测设置中明确将该 runtime 实现为共享 Claude Code scaffold。
3. **若问“现实中最复杂的 Harness 能做到什么程度”**，LongHorizon-Harness 明显更复杂：Manager–Executor–Auditor、跨轮持久审计状态、fresh context、独立只读环境核验、GUI/CLI 路由、权限隔离和恢复循环。但它不训练模型，也没有把这套结构蒸馏进权重。

因此，当前前沿不是“已经把 LongHorizon 蒸完了”，而是：

```text
OPHSD：严格 OPD，但 Harness 很窄
OPRD：严格 OPD + 长程 coding trajectory，但 privilege 仍是 Reminder 层
Harness-Zero：完整 tool-using Harness 的行为内化，但训练是 reviewed-trajectory SFT
LongHorizon-Harness：运行时结构最复杂，但尚无 distillation
```

## 2. 如何衡量 Harness 复杂度

这里用十个维度比较，而不是只数 prompt 长度：

| 维度 | 低复杂度 | 高复杂度 |
| --- | --- | --- |
| 模型调用/角色 | 单调用、固定两阶段 | 多角色、动态路由、子代理 |
| 外部状态 | 无状态文本 | 跨轮持久、结构化、可审计状态 |
| 环境执行 | 无工具 | shell、文件、GUI、应用、测试 |
| 验证 | 自报答案 | 独立环境核验、验收契约、完整性检查 |
| 动作空间 | 纯文本 | 多工具、GUI/CLI、异构接口 |
| 上下文管理 | 单个增长上下文 | fresh context、压缩、分段与选择性持久化 |
| 权限边界 | 所有角色同权 | 读写隔离、只读审计、授权路由 |
| 恢复控制 | 固定调用图 | 重试、回滚、重新规划、阻塞与终止 |
| 持续时长 | 单题数次调用 | 数十到数百步、跨 session/多小时 |
| Harness 可修改性 | 人工固定 | 自动演化源码、工具、中间件、记忆 |

复杂度高不等于更容易蒸馏。恰恰相反，外部持久状态、独立审计和权限隔离都是结构属性，不能仅靠参数更新完整复制。

## 3. 当前最强直接案例：Harness-Zero

论文：[Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974)

### 3.1 源 Harness 到底有多复杂

Harness-Zero 先为每个领域演化一个学生侧 Harness `h*`。它采用 DeepAgents 抽象，允许四类组件：

- **Tools**：增加新的可执行能力；
- **Middleware**：观察、修改或阻断 agent loop；
- **Skills**：按需加载的工作流与操作规则；
- **Memory**：跨任务积累的失败模式和实践。

Harness 演化本身运行三轮。每轮从失败任务出发，最多十个分析子代理并行检查轨迹，主 evolution agent 合并建议，修改共享 Harness bank，再重新评测。它已经属于“可执行、可演化的 Code-as-Harness”，不是简单文本脚手架。

论文中的三个实例展示了不同组件深度：

| 领域 | 代表组件 |
| --- | --- |
| SpreadsheetBench | workbook 操作 skill、7 条 memory 规则、预填单元格保护、答案范围检查、输出文件 guard、turn-budget middleware |
| AppWorld | API workflow skill、失败恢复、context guard、submit gate |
| USPTO | 反合成候选生成 tool、RDKit SMILES 检查、答案验证 tool、完整性 guard、化学先验 memory |

这比 OPHSD 的固定 draft–verify/plan–solve 多出了真实工具、状态检查、动作阻断、领域知识和失败记忆。

### 3.2 它怎样处理动作空间不一致

直接模仿 `h*` 的轨迹会失败，因为源 Harness 可能有目标部署 Harness 没有的工具。Harness-Zero 因此引入 **agent-as-harness**：

1. 学生在固定目标 Harness `h` 下提出下一条 response；
2. harnessing agent 私下读取由 `h*` 改写得到的 reference Harness `K`；
3. 它选择 `pass` 或 `replace`；
4. replacement 必须是目标 Harness `h` 中可执行的完整 response；
5. 接受的 response 才被执行并进入学生可见轨迹；
6. 私下 review、被拒绝 proposal 和 `K` 都不进入训练轨迹。

目标 Harness `h` 是一个极简 mini-SWE-agent 风格 runtime：固定 prompt、一个 Bash `execute` 工具、文件系统持久化，以及可选的一层子代理委派。复杂源工具会被翻译成 Bash/Python 脚本；源 middleware 会变成 proposal review 条件；skill/memory 会变成诊断准则和干预规则。

这个设计解决了完整 Harness distillation 最难的一个问题：**教师与学生动作空间不同**。

### 3.3 它是否属于严格 OPD

需要精确措辞：

- 数据采集沿着 live student trial 进行；每次 review 都发生在学生当前到达的状态，所以它具有 on-policy intervention 的性质。
- 但参数更新是对 accepted responses 做 **LoRA SFT**，不是在学生 token prefix 上匹配 teacher logits 的 reverse/forward KL。
- 因此它是目前最强的 **agent harness distillation / on-policy reviewed imitation**，但不是 OPHSD 那种严格定义的 token-level OPD。

部署时移除 `h*`、`K` 和 harnessing agent，但仍保留最小目标 Harness `h`。所以它证明的是“复杂专用 Harness 可以被压进模型，使部署 Harness 大幅简化”，不是“模型完全不需要 Harness”。

### 3.4 结果与边界

- Qwen3.5-9B 在最小 Harness 下的三领域宏平均：`23.3% -> 44.3%`；
- 同一 base model 带演化 Harness `h*`：`41.7%`；
- 28 个 Harness-exclusive 行为的平均 recovery：`82.3%`；
- 训练集规模：SpreadsheetBench 487 条、AppWorld 282 条、USPTO 500 条 reviewed rollouts。

但它还没有达到 LongHorizon：

- 没有独立 Manager–Executor–Auditor 三权分离；
- 没有跨 25 轮的审计账本和 fresh executor contexts；
- context management 被论文明确列为“不容易自然表达成学生 response”的机制；
- USPTO 中深领域知识和可执行化学功能只被部分内化，`30.0%` 仍低于带 `h*` 的 `38.0%`；
- harnessing agent 每步都 review，数据采集延迟约为最小 Harness 的 `2.4x`。

## 4. Code Agent 中的严格 OPD：OPRD

论文：[OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents](https://arxiv.org/abs/2610.21496)

OPRD 是“针对 code 的 OPD 有没有”的最直接回答：**有，而且是在 repository-level coding agent 轨迹上做的。**

### 4.1 方法

- 学生在没有 Reminder 的情况下在线生成 coding trajectory；
- 可执行 verifier 把轨迹分为成功与失败；
- 成功轨迹做 autoregressive self-imitation；
- 失败轨迹构造两类 privileged guidance：
  - Global Reminder：把最终测试诊断转成整条轨迹的持续指导；
  - Turn-local Reminder：只对某个 environment observation 后的下一次决策生效；
- teacher 在同一个 student-generated token prefix 上、带 Reminder 重新评分；
- student 用 Top-100 reverse KL 匹配 teacher。

这是严格的 on-policy state alignment：teacher 不另采样一条漂亮轨迹，而是在学生真正访问的错误状态上给分布监督。

### 4.2 Action space、Tool space 与代码可用性

OPRD 没有把 action space 定义为一个离散工具集合。方法中的 `a_k` 是第 `k` 个 assistant turn 的完整 token sequence；loss 只覆盖 `Ω(τ)` 中的有效 assistant tokens。System messages、user messages 和 tool responses 只提供 generation context，不作为 prediction targets。因此它蒸馏的是共享词表上的 token policy，其中 tool call 只是可被外部 runtime 解析的一类 assistant token sequence。

论文没有公开 allowed-tools 列表、tool schema、tool-call serialization 或训练 scaffold 实现。能够从正文、Figure 1 和附录直接确认的能力范围只有：

- repository 文件搜索与读取；
- 文件编辑，包括按 `old_string` 定位的局部 edit；
- command/tool execution 及错误恢复；
- compilation、test/build、diff、formatting 和 validation；
- 路径、调用者、source value、data flow 与 patch surface 检查。

Figure 1 中的 `Edit connector.py`、`Search: get_running_loop()` 和 `Run narrow regression test` 是行为示例，不足以证明这些就是精确工具名。评测明确使用 Claude Code scaffold，因此评测 action protocol 必须与该 scaffold 兼容；但训练部分只描述通用 coding environment，不能从论文确定训练与评测是否使用完全相同的 Claude Code tool set。

截至 2026-11-10，论文页面、arXiv TeX source 和 GitHub 标题/arXiv-ID 搜索均未发现作者发布的 OPRD 方法代码。论文只说明训练基于 VeOmni；VeOmni 是分布式训练框架，不包含本文未公开的 coding environment、Reminder runtime 或 tool schema。GitHub 上存在第三方 `daiwk/auto-research` 的 [`reminder_distillation.py`](https://github.com/daiwk/auto-research/blob/main/src/auto_research/post_training/reminder_distillation.py)，但它明确只是同前缀评分、Reminder scope、Top-100 reverse KL 和 teacher refresh 的算法组件，不执行 repository task，也没有 coding-agent Harness。

### 4.3 Harness 复杂度

OPRD 的训练方法明确写的是：当前学生在一个 code execution environment 中在线生成 reminder-free trajectory，环境返回 tool responses，并由 executable verifier 判断最终 repository state。论文没有进一步说明训练 rollout 的 agent scaffold 是否就是 Claude Code。它只在 **Evaluation Setup** 中明确写道，所有被评测模型共享 Claude Code agent scaffold，单题最多 200 agent turns，评测覆盖 SWE-bench Verified、Pro 和 Multilingual。

Teacher 不需要独立运行这个 scaffold。它不重新采样 trajectory，也不执行 tool call，而是在同一条学生轨迹上读取原始 interaction history；论文明确说明 system messages、user messages 和 tool responses 都属于 generation context。Teacher 只是在该上下文中注入 Reminder，对学生已经生成的 token prefix 计算 logits。因此，执行 action 的 runtime 与定义 action/tool interface 的序列化上下文需要区分：前者只由 rollout policy 调用，后者会同时进入 student/teacher 的条件输入。

但要分清两个层次：

```text
保留的内层 Harness：coding agent runtime、tools、repository environment、verifier
评测时的具体实现：共享 Claude Code scaffold
被蒸馏并移除的外加层：Global / Turn-local Reminders
```

所以 OPRD 是复杂 **coding policy** 上的 OPD，却不是完整 **coding runtime Harness** 的 OPD。它把“看到测试/工具反馈后该如何纠偏”压进模型，但没有内化 shell、编辑器、测试执行、上下文管理或权限系统本身。

### 4.4 结果

- Qwen3.5-4B 三个 benchmark 宏平均：base `36.6%`，完整 OPRD `47.9%`；
- Qwen3.5-9B：base `44.4%`，完整 OPRD `55.9%`；
- 4B diagnostic error events：`446 -> 222`，下降 `50.2%`。

在“严格 OPD + code agent”这个窄定义下，OPRD 是当前最关键的上限案例。

## 5. 为什么 LongHorizon 仍是更高一级

论文与实现：[LongHorizon-Harness](https://arxiv.org/abs/2608.01964)、[GitHub](https://github.com/AMAP-ML/LongHorizon-Harness)

LongHorizon-Harness 是一个外层 Manage–Execute–Audit 状态机：

- **Manager** 持有原始目标、结构化 task state 和全部 audit reports，但不能操作环境；
- **Executor** 每轮拿 fresh、budget-bounded context，可通过 GUI 或 CLI 修改环境；
- **Auditor** 在独立 fresh context 中用只读工具检查真实环境，不信任 Executor 的完成声明；
- 只有审计后的 requirement/artifact/fact 记录和 audit report 跨轮持久化；
- 记录状态包括 `completed / pending / blocked / untrusted`，并保留证据引用；
- GUI/CLI executor 与 auditor 有不同 capability boundaries；
- 外层最多 25 轮，每轮 Executor 最长 1800 秒，Manager/Auditor 各 300 秒；
- 内层可直接运行 Claude Code、Codex CLI、OpenCode 或 DeepSeek Harness 的原生 loop。

它在复杂度上超过 Harness-Zero 的源 Harness，主要不是工具更多，而是多了四个难以权重化的结构：

1. **持久任务账本**：本次任务的动态事实不能预先存进参数；
2. **独立审计视角**：同一模型在同一上下文自审不等价于隔离的只读 Auditor；
3. **权限与职责分离**：Manager 无环境权，Auditor 无写权，这是 runtime enforcement；
4. **跨上下文恢复**：丢弃 Executor 原始轨迹，只从 verified checkpoint 继续。

LongHorizon 自己明确说明不训练新模型。当前证据中没有工作把这四项一起蒸馏到单个学生并在移除外层 MEA 后保持能力。

## 6. 复杂度排序

下表把“被蒸馏对象的复杂度”和“运行任务本身的复杂度”分开：

| 工作 | 被迁移的 Harness/privilege | 环境交互 | 持久状态 | 独立验证 | 参数更新 | 训练范式 | 部署移除什么 | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| HAD | 状态跟踪与行为监控文本 | 简单 agent 环境 | tracker state | 否 | 是 | harness-aware training | 不移除 Harness | 窄 |
| OPHSD | draft–verify / plan–solve 多调用程序 | 否 | 临时 inference state | 内部 verifier call | 是 | student-prefix reverse KL | 整个推理 Harness | 严格 OPD，但简单 |
| HERD | skill registry + residual memory | 文本任务流 | 跨 stage memory | verified answers | 是 | answer-prefix forward KL | 全部外部 skill/memory | 状态丰富但非 runtime |
| HPD | runtime control document 与状态语义 | replay 环境 | runtime state | harness-verified action | 是 | offline hard-target distillation | privileged control document | runtime 语义较强，非 on-policy |
| OPRD | 全局/局部 coding Reminders | repository + tools + tests | trajectory context | executable verifier | 是 | online reverse KL + self-imitation | Reminders | 最强严格 Code OPD |
| Harness-Zero | tools + middleware + skills + memory | Bash、文件、应用、领域工具 | 文件系统与 Harness memory | middleware/benchmark verifier | 是 | live review + SFT | 专用 Harness、review agent | 最强完整 Harness-to-weight |
| AHD | Claude Code/Hermes/OMP 的多代理编排 | 多代理/tool use | inference-time state | validation loop | 否 | black-box Harness extraction | 不移除重建 Harness | Harness-to-Harness，不是 OPD |
| AHE | coding Harness 源码与配置 | Terminal/SWE | long-term memory | benchmark + edit attribution | 否 | Harness source evolution | 不移除 Harness | 最强 Code Harness 优化之一 |
| LongHorizon | MEA 外层完整 runtime | GUI + CLI | audited cross-round ledger | 独立只读 Auditor | 否 | inference-time loop | 无 | 运行时复杂度上限 |

如果只看“有无完整 Harness 被内化”，排序是：

```text
OPHSD < OPRD < Harness-Zero < [尚未实现的 LongHorizon distillation]
```

其中 `<` 表示源 Harness 的工程/状态复杂度，不代表训练算法严格性。严格 OPD 反而是 OPHSD 和 OPRD 更标准。

## 7. Code Harness 邻近工作：哪些不能算 OPD

### 7.1 Agentic Harness Engineering（AHE）

[AHE](https://arxiv.org/abs/2604.25850) 自动修改 system prompt、tool description/implementation、middleware、skill、subagent config 和 long-term memory。十轮把 Terminal-Bench 2 pass@1 从 `69.7%` 提到 `77.0%`。但 base model 全程冻结，学习发生在 Harness 文件中，因此是强 Code Harness evolution，不是 OPD。

### 7.2 Agent Harness Distillation（AHD）

[AHD](https://arxiv.org/abs/2607.28147) 从 Claude Code、Hermes 和 OMP 的黑盒输出中恢复角色、拓扑、tool use 和 coordination，再让 coding agent 写出一个较小的可执行 Harness。它的 “distillation” 发生在 Harness 到 Harness，模型权重不更新。这个工作说明复杂 Harness 可以被压缩/复制，但不回答能否进单模型权重。

### 7.3 Multi-Harness RL

[Multi-Harness RL](https://arxiv.org/abs/2609.04518) 用 Aider、OpenHands、Qwen Code 和 SWE-agent 轨迹做 GRPO replay。它不是 teacher-conditioned distillation，而且对未见 weak-ReAct Harness 没有显著迁移；训练 Harness 间的性能差 `7.13` 个点，明显大于 RL 增益。这个负结果说明普通多 Harness 训练不会自动得到 Harness invariance。

### 7.4 Patches-to-Trajectories（P2T）

[P2T](https://arxiv.org/abs/2605.21996) 用 reference patch 构造 prerequisite process graph，再策划 grounded trajectories 做 SFT。它的 curator 很复杂，包含 proposer、critic、滑动窗口、多候选 sandbox、groundedness judge 和 verifier，但它是 privileged data curation，不是学生 on-policy Harness distillation。

### 7.5 Anchored Context Distillation（ACD）

[ACD](https://arxiv.org/abs/2609.31430) 把旧 tool observations 压成 soft tokens，并从 full-text teacher 向 latent-view student 做 offline KL。它针对 coding agent context representation，保留 OpenHands tools/runtime，不是完整 Harness 行为内化。

## 8. 现在真正开放的问题

Harness-Zero 已经把 tools、middleware、skills、memory 的**可表达行为**翻译到目标动作空间，下一步不应只是再加一个更长 prompt，而应挑战 LongHorizon 的结构属性。

一个现实的研究阶梯是：

### Stage A：Manager policy distillation

保留持久 task-state API，只蒸馏：

- 如何把原目标分成 bounded contract；
- 如何从 audit report 更新 requirement/artifact/fact；
- 如何选择 `execute / done / blocked / ask`。

这是最容易对齐且风险最小的部分，但部署仍需要外部账本。

### Stage B：Auditor-guided on-policy distillation

学生自己执行；独立 Auditor 在学生访问的真实环境状态上给 privileged guidance。可以借鉴 OPRD：

- 成功 transition 做自模仿；
- 假完成、越界修改和漏验收状态做 auditor-conditioned reverse KL；
- 只蒸馏“何时检查、检查什么、失败后如何恢复”，不声称蒸馏独立性本身。

### Stage C：Harness-Zero 式动作空间翻译

把 GUI/CLI Auditor 或专用工具的结论，翻译成学生原生动作：

- CLI verifier 发现问题后，不直接把 CLI action 当 target；
- harnessing agent 生成 GUI-only 或目标 tool schema 中的下一步；
- 所有新证据必须通过学生动作进入学生可见轨迹。

### Stage D：保留最小可信 runtime

合理终点可能不是 Harness-Zero，而是 **Harness-Minimal**：

- 权重内化分解、核验习惯、恢复策略和工具使用；
- runtime 只保留动态账本、权限边界、真实环境执行和独立审计。

这些组件保留不是蒸馏失败，而是因为它们本质上是状态、权限和独立性，不是静态策略知识。

## 9. 最终判断

用户对 OPHSD “Harness 太简单”的判断是对的。它适合证明 on-policy Harness self-distillation 的机制，但不能代表现实 agent Harness 的复杂度。

当前最值得作为下一阶段基线的是：

- **要严格 OPD 与代码任务：选 OPRD；**
- **要复杂 Harness 内化：选 Harness-Zero；**
- **要研究目标上限：用 LongHorizon-Harness 定义未被蒸馏的结构；**
- **要构造强 Code Harness 教师：参考 AHE，再用 Harness-Zero/OPRD 的方式做迁移。**

最准确的一句话是：

> 复杂 Code Harness 的 OPD 已经从“有没有”进入“能蒸馏哪些组件”的阶段；tools、middleware、skills 和 memory 的行为已有 Harness-Zero 证据，coding trajectory 上的严格 on-policy KL 已有 OPRD 证据，但 LongHorizon 式持久审计状态、角色隔离与权限边界仍没有被完整蒸馏。

## 10. 主要来源

- Harness-Zero: Harness Distillation via Agent-as-Harness, arXiv:2609.24974
- OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents, arXiv:2610.21496
- LongHorizon-Harness: Advancing Long-Horizon Agents for Real-World Tasks, arXiv:2608.01964
- Agentic Harness Engineering, arXiv:2604.25850
- Agent Harness Distillation, arXiv:2607.28147
- Multi-Harness Reinforcement Learning for Coding Agents, arXiv:2609.04518
- From Patches to Trajectories, arXiv:2605.21996
- Anchored Context Distillation for Software Engineering Agents, arXiv:2609.31430
- Code as Agent Harness, arXiv:2605.18747