# Code / GUI Agent × Harness：严选文献与 Benchmark 对照

> 调研截点：2026-11-14。只收录可追溯到论文原文的工作；优先 2026、真实可执行环境、明确 Harness 组件、可验证的 benchmark。本文聚焦 **运行时 Harness**，不把有工具调用的 Agent 都算作 Harness 创新。  
> 配套专题：[Code / GUI Agent × OPD](code-gui-agent-opd-papers-2026.md)。已有深度分析：[复杂 Harness 与 OPD](../notes/Complex-Harness-OPD-and-Code-Harness-landscape-zh.md)、[GUI Harness 候选图谱](../notes/GUI-Agent-Harness-OPD-candidate-map-zh.md)。

## 先给结论

**最值得拆系统复现的复杂 Harness**：LongHorizon-Harness（独立 Manager / Executor / Auditor + 持久审计状态 + GUI/CLI）、openJiuwen（可组合多代理 + 动态运行时控制）、GUI-HARVEST（按 GUI 环境证据自动修改 Harness 源码）、Harness-of-Harness（嵌套原生 coding agent，跨多轮/多日迭代的软件工程）。这些是「Harness 复杂」的代表，但 **没有因为 Harness 变强就证明完成了 OPD**。

**控制变量最有价值的工作**：An Empirical Study of Harness Design、Rethinking Verification、Harness-Bench、Same Model Different Harness；适合回答「同一个模型换 Harness 到底提升在哪里」，而不是依赖排行榜裸分。

**如果目标是未来做 Harness → Student 蒸馏**：优先关注可以产生 *同一执行状态下* teacher action 的 Mid-Harness、独立 reviewer、GUI-HARVEST 的 transition diagnosis，以及 LongHorizon 的 audit report。跨角色权限、外部工具能力和持久文件状态不能默认通过无工具 OPD 完整内化。

## 判定口径

- **Harness**：模型外部的执行循环、动作接口、上下文/记忆、环境状态、验证、恢复、任务编排、权限边界，以及对上述系统的自动优化。
- **复杂度 H3**：多角色/多层执行 + 持久状态或独立验证 + 动态路由/恢复；**H2**：至少两种实质性 Harness 控制机制；**H1**：单一局部机制。此分级是本仓库研究用标签，**不是作者报告的分数**。
- **证据等级 A**：原文有固定模型或匹配设置、组件消融及真实环境评测；**B**：原文有完整任务评测但组件归因较弱；**C**：以系统描述、survey 或基准为主。预印本不等于经过同行评审。
- Benchmark 不能混用：SWE-bench Verified / Pro / Pro V2 / Multilingual、Terminal-Bench 各版本、OSWorld / OSWorld-Verified / OSWorld 2.0 都是不同任务与协议。

## 一、Code / Terminal Agent：Harness 本体

| 优先级 | 论文（原文） | Harness 核心机制 | 实际评测 | 复杂度 / 证据 | 是否训练模型 |
| --- | --- | --- | --- | --- | --- |
| **P0** | [openJiuwen: Beyond Static Harnesses for Long-Horizon Coding Agents](https://arxiv.org/abs/2608.27969) (2026) | Rail 组合能力、单 Agent/子代理/Swarm Flow；根据语义诊断和执行结果动态改变上下文、反馈、控制 | **SWE-bench Verified、Terminal-Bench 2.1** | H3 / B | 否，固定模型策略下改 Harness |
| **P0** | [LongHorizon-Harness: Advancing Long-Horizon Agents for Real-World Tasks](https://arxiv.org/abs/2608.01964) (2026) | Manage–Execute–Audit；fresh-context executor、独立只读 auditor、审计状态跨轮保存；适配原生 agent backend | **Terminal-Bench 2.1、WeaveBench、OSWorld 2.0** | H3 / A-B | 否 |
| **P0** | [Rethinking Verification in Agent Harnesses](https://arxiv.org/abs/2610.34577) (2026) | 将验证分为初始指令、执行中提醒、完成门控、独立审查四层；17 个方案 | **SWE-bench Verified、Terminal-Bench、VisualBench** | H2 / A | 否 |
| **P0** | [An Empirical Study of Harness Design for Coding Agents](https://arxiv.org/abs/2609.20804) (2026) | 固定执行循环，只比较 planning、tool action space、context management / budget | **SWE-bench Verified、Terminal-Bench 2.1** | H2 / A | 否 |
| **P1** | [Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement](https://arxiv.org/abs/2609.01481) (2026) | 元 Harness 运行既有 coding harness，增量 plan–code–test、独立评价、版本化项目、逐步暴露 tools/skills | **GameCraft-Bench、FrontierSWE、ProgramBench**；**不是 SWE-bench** | H3 / B | 否 |
| **P1** | [Mid-Harness: Scaling Actions Between Model and Harness for Terminal Agents](https://arxiv.org/abs/2609.39982) (2026) | 每个 action 在真实执行前多采样、验证、择优；保持 generator 与外层 Harness 不变 | **TerminalBench-Lite**，另有跨模型/其他 benchmark 测试 | H2 / A-B | 有 verifier 蒸馏实验，但**不是 actor 的 on-policy OPD** |
| **P1** | [Same Model, Different Harness: Different Coding-Agent Results](https://arxiv.org/abs/2608.26218) (2026) | 机械性截短旧工具回显、检测停滞，模型和任务固定；考察紧上下文 | **SWE-bench Verified、Pro、FeatureBench** | H1-H2 / A | 否 |
| **P1（诊断基准）** | [Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows](https://arxiv.org/abs/2605.27922) (2026) | 跨 Harness × 模型组合的隔离沙盒与 oracle；记录 trace、validator、成本和结果 | **Harness-Bench 自建任务**；不是 SWE-bench | H2 / A（基准设计） | 否 |

### P0-1. openJiuwen：复杂多代理 coding harness

- **值得看的组件**：共享执行底座、Rail-based capability composition、delegated sub-agents、Swarm Flow、runtime adaptivity。关键不是加一个 prompt，而是让新的 tool feedback、diagnostics、context relevance 改变下一步的系统控制。
- **原文结果**：SWE-bench Verified **82.6%**，Terminal-Bench 2.1 **87.19%**（原文比较选定 leaderboard 点估计；排行榜 Harness/模型/预算未必同控，**不可把分数差直接解释为纯 Harness 因果收益**）。
- **蒸馏机会**：记录每次 context pruning、subagent routing、review / repair 决策并映射回统一 action schema；但「子代理独立执行」和「并行工具结果」不是一个模型的隐藏推理。
- **优先读**：[论文 v2（2026-11-05）](https://arxiv.org/abs/2608.27969)。

### P0-2. LongHorizon-Harness：结构复杂度上限之一

- Manager 维护目标、审计过的 task state 和下一步；Executor 用 fresh context 及真实工具执行；**Auditor 只读环境、独立核对真实状态**。只将验证过的事实跨轮持久化。
- 可插入 Claude Code / Codex 等已有 agent loop：这是 **外层 orchestration Harness + 内层原生 Harness**，不是一个简单 ReAct 提示词。
- 原文同一组实验中 Qwen3.7-Plus：WeaveBench **51.8→80.7%**、Terminal-Bench 2.1 **69.7→77.2%**、OSWorld 2.0 **2.8→8.3%**。不同 benchmark 不能横向平均。
- **OPD 边界**：可蒸馏「何时停止 / 返工 / 需要审计 / 如何维护任务事实」；不能把 read-only 权限、持久数据库、实际 GUI 观察等外部能力在无工具部署中直接替换。
- [原文](https://arxiv.org/abs/2608.01964) · [作者代码](https://github.com/AMAP-ML/LongHorizon-Harness) · [仓库既有可蒸馏性分析](../notes/LongHorizon-Harness-OPD-distillability-analysis-zh.md)。

### P0-3. Rethinking Verification：验证位置和角色的受控实验

- L0 initial instructions；L1 mid-run reminders；L2 completion gate；L3 independent reviewer。跨 **7 个 coding 模型、17 种方案** 比较。
- 原文 L3 在 Terminal-Bench 平均提升 **5.9–9.5pp**、SWE-bench Verified **3.9–6.6pp**；对应成本区间分别约 **1.13–1.25×** 和 **1.00–1.04×**。L2 的提升可能伴随明显更高成本。
- **为什么重要**：对于 OPD，可构造一个和 Student 共享执行状态、但多了独立 review/verification 的 Teacher。对照应包含「无审查」「文字提醒」「独立审查」「同等额外 token 预算」。

### P0-4. An Empirical Study：不要把 harness component 一锅端

- **4 个模型、176 组匹配设置**，固定 agent loop；同时考察 5 种 context strategy、4 种 window budget、planning 和 action space。
- 主要结论：紧上下文下 context management 价值较高；先规则性 elision 再 summarization 更有效；strong bash-capable model 在 bash-only 接口下可能更省成本；planning 对更强模型更多体现为成本收益。
- **设计参考**：若做 OPD，要针对 context、plan、tool routing 各自建 teacher privilege，而不是将整个 Harness 改进归因给某个 KL loss。

### P1. 复杂系统及局部机制

- **Harness-of-Harness**：跨数轮持续开发、独立验收、技能和项目版本史；作者报告三次迭代下平均相对增益 52.25%，另有 70+ 迭代软件项目。评测来自 **GameCraft-Bench / FrontierSWE / ProgramBench**，不应误写 SWE-bench。
- **Mid-Harness**：执行前 action Best-of-N + verifier。在 TerminalBench-Lite 中，TMAX-9B base **50.00%**、8 候选 + 较强 verifier **68.03%**。是很干净的 **可蒸馏 action selection 教师**：关键是共享状态和目标动作协议。
- **Same Model Different Harness**：紧窗口控制实验在一个 169-task SWE-bench Verified cohort 上，完整解题数从 **43→72**；这是 cohort 结果，不是完整 500-task leaderboard。
- **Harness-Bench**：arXiv 摘要版本为 **106 tasks、5,194 trajectories**；[项目官网](https://www.harness-bench.ai/)展示更大规模的后续配置。**以具体论文/网站版本为准，不混报任务数和结果**。

## 二、GUI / Computer-Use：Harness 本体

| 优先级 | 论文 | Harness 的硬机制 | Benchmark | 复杂度 / 证据 | 模型是否冻结 |
| --- | --- | --- | --- | --- | --- |
| **P0** | [GUI-HARVEST: Self-Improving GUI Agents through Evidence-Driven Harness Evolution](https://arxiv.org/abs/2610.00948) (2026) | 前后截图 + action effect；重复 rollout 诊断；跨任务失败模式；受约束源码编辑与回归验证 | **OSWorld-Verified、WindowsAgentArena** | H3 / A-B | **是** |
| **P0** | [LongHorizon-Harness](https://arxiv.org/abs/2608.01964) | GUI/CLI executor + 独立只读 auditor + 外部 verified state + restart/fresh context | **OSWorld 2.0、WeaveBench** | H3 / A-B | 是 |
| **P1** | [PhoneHarness: Harnessing Phone-Use Agents through Mixed GUI, CLI, and Tool Actions](https://arxiv.org/abs/2606.14832) (2026) | GUI / device CLI / host-side tools 路由；受限 GUI delegation；可审计 side-effect 证据 | **PhoneHarness Bench**（专有配套基准） | H3 / B | 非参数训练贡献 |
| **P1** | [WeaveBench: A Long-Horizon, Real-World Benchmark for Computer-Use Agents with Hybrid Interfaces](https://github.com/weavebench/WeaveBench) (EMNLP 2026) | **基准而非新 OPD 方法**；任务强制 GUI + CLI/code 交织、保留真实执行 trace | **114 个跨 8 领域长任务** | H3（环境）/ A-B | 不适用 |

### P0. GUI-HARVEST：固定 VLM，真正优化 Harness 源码

- 诊断不是只看成功/失败：将 **LLM 计划 → 实际 action → before/after screenshots** 对齐；同题多次运行比较随机差异；跨题汇聚稳定失败模式。
- 自动提出 **bounded Harness source-code edits**，预先记录可检验行为预测，再通过重复运行核对预测效果及 held-out task 分数，区别于仅修改 system prompt。
- 原文：Qwen3-VL-32B-Instruct 在 OSWorld-Verified 全套 **+12.33pp**；冻结已优化 Harness 迁移到 WindowsAgentArena（GPT-5，50 steps）**+13.87pp**。请注意这是 **Harness 优化、无权重更新，不属于 OPD**。
- [论文](https://arxiv.org/abs/2610.00948) · [代码](https://github.com/GaryYang12345/GUI-HARVEST) · [本仓库已有方法详解](../notes/GUI-HARVEST-reading-notes.md)。

### P1. PhoneHarness 与 WeaveBench：Hybrid Interface 的价值和代价

- **PhoneHarness** 的核心是执行接口层：选 GUI tap/swipe、设备命令，或结构化工具；终局校验真实 side effects，原文 annotated split **75.0% pass rate**、比最强非 PhoneHarness 配置高 **12.9pp**。但这是作者的 PhoneHarness Bench，**不是 AndroidWorld/OSWorld**。
- **WeaveBench** 明确要求一个任务里同时使用 GUI 观察与 CLI/code 交互。作者报告最强 model×harness pairing **41.2%**；是分析跨动作空间决策和 long-horizon handoff 的重要评测。
- **OPD 难点**：CLI 操作可能一次改变大量文件/后台状态，GUI-only Student 不可能用相同一步 action 做 token KL；应先确定终局部署保留什么工具，再选监督对象。

## 三、Benchmark 的正确选择

| 基准 | 重点 | 如何验证 | 适合哪些研究 | 风险 |
| --- | --- | --- | --- | --- |
| [SWE-bench Verified](https://www.swebench.com/) | 真实 GitHub issue 修复，通常 500 例 | repository patch + executable tests | Code Agent 改 Harness / OPD 的经典统一基线 | 老基准可能饱和；不同 scaffold / budgets 高度影响结果 |
| [SWE-Bench Pro](https://proceedings.mlr.press/v306/deng26f.html) | 更长程、更复杂、多语言 SWE | 代码状态/测试；含 public/private 协议 | 复杂编程、长期记忆、验证 | **Pro / Pro V2 / Pro-618 子集不等价** |
| [Terminal-Bench](https://www.tbench.ai/) | 终端任务及工具调用 | 容器化可执行 grader | Shell/tool action、memory、mid-harness | 2.0、2.1、Lite、不同任务子集分数不能互比 |
| [OSWorld](https://arxiv.org/abs/2404.07972) | 真实桌面/多应用状态，初始 369 tasks | VM/task-specific execution-state checks | GUI harness、动作/状态校验 | OSWorld-Verified 与 2.0 不是同一测试集 |
| [OSWorld 2.0](https://arxiv.org/abs/2606.29537) | **108 个长程真实任务**；比 1.0 更复杂 | VM 工件/状态、binary & partial metrics | Manager–Executor–Auditor、长上下文、动态环境 | 平均 318 tool calls（特定参考系统），成本高；partial 不等于 binary |
| [AndroidWorld](https://arxiv.org/abs/2405.14573) | **116 类程序化 Android tasks**、参数化任务 | init / state verifier / teardown | GUI-OPD 在线环境、step feedback | 不同手机 agent 的动作 DSL 和 horizon 差异 |
| [WeaveBench](https://github.com/weavebench/WeaveBench) | **114 long-horizon GUI+CLI 任务** | 轨迹+产物的 evaluator | 异构 action space、复杂 Harness | 其 judge 协议与纯可执行单元测试不同 |
| [Harness-Bench](https://arxiv.org/abs/2605.27922) | 直接量测 harness 配置影响 | 离线 sandbox 与任务 oracle | 把 Harness 当成实验自变量 | 不是 SWE-bench；论文和网站任务数存在版本差 |

### 读排行榜前必须记录的字段

同一报告必须记录：模型精确 checkpoint、Harness repo/commit、system prompt、工具 schema / action parser、单题 turn/time/token budget、并发与重试、context compaction、是否多候选/critic、测试集具体版本与任务数、训练集去重/污染控制、pass@1 还是 Best-of-N、均值/多 seed。**不能把多代理多 rollout 分数与单策略 pass@1 直接比较**。

## 四、从「复杂 Harness」到「可做 OPD」的路线

1. **最小可验证桥梁**：Mid-Harness 的同状态 action proposals + verifier，先在 Terminal-Bench / SWE-bench 的相同 action schema 上研究候选选择能否进权重。
2. **更强的反馈桥梁**：GUI-HARVEST 的 pre/post screenshot transition evidence 或独立 reviewer 的诊断，仅在状态对齐、action 可执行时为 Student 提供 privileged policy guidance。
3. **高难度**：LongHorizon 的 audited state / context resets / subagent delegation。推荐分层蒸馏 *decision policy*（stop、review、delegate、write memory），而非宣称外部可执行能力被全部内化。
4. **Benchmark 组合**：Code 先 SWE-bench Verified + Terminal-Bench 2.1，再 Pro V2；GUI 先 AndroidWorld / OSWorld-Verified，结构上限再 OSWorld 2.0 和 WeaveBench。所有阶段固定 backbone 与资源预算。

## 相关仓库已有笔记

- [SWE-bench 训练环境、action space、verifier](../notes/SWE-bench-training-environments-and-action-spaces-zh.md)
- [复杂 Harness-OPD 可蒸馏性](../notes/Complex-Harness-OPD-and-Code-Harness-landscape-zh.md)
- [GUI Harness 生态与 scale 对标](../notes/Harness-OPD-GUI-Harness-ecosystem-and-scale-report-zh.md)
- [GUI Harness OPD 候选图谱](../notes/GUI-Agent-Harness-OPD-candidate-map-zh.md)

> 甄别原则：优先论文原文 + 官方代码和协议；不从第三方综述搬运未核验成绩；所有上面的性能数字仅对应各自论文报告的特定实验设置。
