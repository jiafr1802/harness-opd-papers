# 哪些 GUI Agent Harness 适合做 OPD

基于原文方法、动作空间与消融的候选图谱 · 2026-11-08

## 1. 结论先行

如果目标是把 GUI Harness 的收益蒸馏进一个**部署时不带该 Harness、只执行一次策略调用的 Student**，最值得优先做的不是代码动作或统一工具接口，而是以下三类：

1. **同状态多采样与评审：GTA1。** Teacher 和 Student 看同一张截图、共享历史和 GUI Action DSL；Teacher 只多做 $K$ 次候选采样及一次 judge 选择。它最接近 OPHSD 所需的 state-matched privileged teacher，是最干净的首个 GUI Harness-OPD 实验。
2. **动作后 transition critic：OS-Symphony 的 RMA、BJudge 的 Behavior Narrative Generator、ClawGUI 的 PRM。** 这些模块比较 action 前后截图，产生“动作是否生效、是否推进任务”的额外监督。它们比外部知识检索更容易蒸馏，因为监督来自 Student 自己已经经历的 transition。
3. **多 rollout 比较：BJudge。** 它能把 Pass@$N$ 的潜在能力转成被选中的完整行为，但不能直接等同于逐 token OPD。更合理的用法是生成轨迹偏好、fork-point action preference 或成功轨迹上的 state-matched teacher targets。

第二梯队是 Agent S2 的规划/路由与 Syll 的 demonstration replay。它们可提供监督，但 Teacher 的结构状态、专家接口或示范信息未必能由无 Harness Student 重建。

CoAct-1、Agent S3 的 Coder、StateAct、CLI-Anything 的主要收益来自**不同动作空间或特权程序状态**。它们适合“保留工具，蒸馏如何调用”，不适合作为严格的无工具 GUI OPD 首选。OpenComputer 更适合充当 verifier、数据引擎和 reward source，而不是被蒸馏的部署 Harness。

## 2. 判定标准

对每个候选，用六个问题判断其收益能否进入 Student 权重：

| 维度 | 通过条件 | 不通过时的后果 |
| --- | --- | --- |
| 状态对齐 | Teacher 在 Student 实际到达的 GUI state 上给监督 | 只能做离线 imitation，容易产生 state distribution shift |
| 信息可重建 | Teacher 使用的信息来自 Student 可见历史或可执行动作 | 蒸馏不能恢复部署时不存在的信息 |
| 动作空间一致 | Teacher 最终 action 属于 Student 的 GUI Action DSL | token/action KL 无定义，需做轨迹翻译或结果监督 |
| 同骨干可控 | 固定 base model，只改变 Harness | 才能把提升归因给 Harness 而不是更强模型 |
| 增益可分解 | 有组件消融或能关闭单一模块 | 无法知道蒸馏目标究竟是什么 |
| 无 Harness 部署闭合 | Student 单次策略能执行目标行为 | 否则应改成 harness-aware distillation |

这里要特别区分三种研究目标：

- **Harness removal：** 训练时有 Teacher Harness，部署时完全移除；这是本文排名的主标准。
- **Harness retention：** 部署保留 code/API/memory，Student 学会何时调用；更接近 HPD/HAD。
- **Environment/reward distillation：** verifier 只在训练期打分，不构成 Teacher policy；这是 OpenComputer、PRM 的常见角色。

## 3. 候选图谱

| 候选 | Harness 的关键增益 | 状态与动作对齐 | 无 Harness OPD 潜力 | 最合适的监督 |
| --- | --- | --- | --- | --- |
| **GTA1** | 每步采样 $K$ 个 action proposal，由 MLLM judge 选一个，再交给 grounder | 高：同一当前截图、任务和历史；最终仍是同一 GUI action | **高，首选** | selected action 的 token KL、semantic-action KL、候选偏好 |
| **BJudge** | 多个完整 rollout 经 transition facts 压缩成 behavior narratives，再比较选优 | 中：初态相同，但候选后续 state 分叉；Agent S3 还含 code action | **高到中** | trajectory preference、fork-point preference、选中轨迹上的 hindsight targets |
| **OS-Symphony RMA** | 根据前后截图验证动作，维护 milestone memory，输出 On-track/Completed/Infeasible/Off-track | RMA 高；Searcher/Coder 低 | **RMA 高，整套中低** | transition label、recovery action、stop/continue label |
| **ClawGUI PRM** | PRM 对前后截图和历史给 step reward，GiGPO 做细粒度 credit assignment | 高：可直接评价 Student rollout；它本身是 reward 而非 action teacher | **高，但更像 RL/OPD 混合** | step value、pairwise preference、value-gated OPD |
| **Agent S2** | Manager/Worker 主动重规划；视觉、OCR、结构化 grounding 专家路由 | 规划 action 较高；UNO `set_cell_values` 等结构动作低 | **中** | subgoal/action proposal 蒸馏；限制 Teacher 只用 GUI-compatible experts |
| **Syll demo replay** | demonstration phase index、Action ICL、Wait/Recover 与 post-state expectation | 中：示范可提供可见 cue，但 phase index 是外置状态 | **中** | phase/action imitation、expectation prediction、课程式撤除 demo |
| **StateAct** | 主 Agent 直接读写文件、backend、DOM；独立 finish gate 检查持久状态 | 低：主动作空间是 code/state，GUI 只是少量 fallback | **低（严格 GUI-only）** | 保留 state tools；蒸馏 routing、何时 GUI fallback、验证习惯 |
| **CoAct-1** | Orchestrator 在 Programmer 与 GUI Operator 间动态委派 | 低：Python/Bash 一步对应多步 GUI，且模型异构 | **低** | tool-use policy distillation 或结果级训练，不做直接 action KL |
| **Agent S3** | flat GUI policy 加内嵌 coding agent | GUI 部分高，code 部分低 | **低到中** | 只蒸馏 flat replanning；Coder 保留或单独做轨迹翻译 |
| **CLI-Anything** | typed command、显式 JSON state、backend truth、程序验证 | 低：它刻意替代 pixel/GUI action surface | **低** | 用作 privileged verifier/data generator；不宣称 GUI action 已被内化 |
| **OpenComputer** | app-specific verifier、任务生成、partial-credit reward | policy 不适用；reward 与 Student rollout 完全可对齐 | **不是 Teacher Harness；训练基础设施价值极高** | verifier-gated rejection sampling、RL、成功/近失轨迹配对 |

## 4. 第一优先：GTA1 的 Step-wise Best-of-$K$ OPD

### 4.1 为什么它最干净

GTA1 在每个真实 GUI step 给 planner 相同的 instruction、history 和 current screenshot，采样 $K$ 个 action proposals，再由 judge 选出最合适的 proposal。论文明确指出 judge 可以就是 planner 本身；默认 $K=8$，并在 UI-TARS-1.5-7B 上测试 $K\in\{1,8,16,32\}$。因此可以固定同一开源 backbone，唯一变化是 Teacher 多了采样和比较计算。

这满足四个关键条件：

1. Student state 是真实 on-policy state，不需重放或猜测；
2. Teacher 没有额外环境信息，只用了更多 inference compute；
3. Teacher 选出的 action 仍在同一 GUI DSL；
4. 部署时可退回 $K=1$，研究目标闭合。

论文结果也给出了 Harness 增益量级：在 OSWorld 上，$K=8$ 时 50-step 为 43.4%，而 $K=1$ 为 41.3%；100-step 为 45.2% 对 43.4%。该差距不算巨大，反而适合做归因清晰的第一实验：若 Student 能追回其中大部分，说明 GUI test-time compute 确实可内化；若追不回，也能较明确地否定假设。

### 4.2 最小实验

设 Student policy 为 $\pi_\theta$，在 state $s_t$ 采样并执行 $a_t$。Teacher Harness 在**同一个 $s_t$** 上：

1. 从冻结参考策略采样 $K$ 个语义 action proposals $\{c_t^k\}_{k=1}^K$；
2. judge 选择 $c_t^*$；
3. 若 action 需要坐标，由 Teacher 与 Student 共用同一 grounder，避免把 grounding 模型变化混入实验；
4. 在 Student rollout 上优化：

$$
\mathcal{L}=\lambda_{\mathrm{tok}}D_{\mathrm{KL}}\!\left(
\pi_T(\cdot\mid s_t,c_t^*)\;\|\;\pi_\theta(\cdot\mid s_t)
\right)
+\lambda_{\mathrm{pref}}\mathcal{L}_{\mathrm{pref}}(c_t^*,c_t^-).
$$

如果不同候选的文本写法不同但语义 action 相同，应优先采用 ActionOPD 式的 semantic-action canonicalization，再做 action-level distillation，而不是强迫自然语言 proposal 逐 token 对齐。

### 4.3 必须有的对照

- $K=1$ Student baseline；
- $K=8$ Teacher upper bound；
- 只 SFT Teacher 选中轨迹的离线 baseline；
- Student-state OPD 对 Teacher-state imitation；
- token KL 对 semantic-action KL 对 pairwise preference；
- same-backbone judge 对更强 judge，区分“额外计算”与“额外模型能力”；
- 固定 grounder，另做 grounding distillation，避免混淆 planning 和 localization。

## 5. 第二优先：Transition-Critic OPD

三个系统提供了高度相似、可组合的训练信号：

- **BJudge narrative generator：** 输入 action 前截图、action、3 秒后的截图，生成 action-effect facts；行为叙事比 screenshot-only、trajectory summary、naive captioning 分别更适合轨迹选择，10 个 GPT-5 Mini rollouts 下达到 60.2%，最佳对照为 56.8%。
- **OS-Symphony step summary/RMA：** 辅助 VLM比较前后截图并判断 GUI action 是否成功，再将轨迹分类为 On-track、Completed、Infeasible 或 Off-track。论文人工检查 100 个 RMA 的 GUI Error 报警，约 90 个确有错误；但也报告细粒度视觉误报和漏报。
- **ClawGUI PRM：** 每步读取 previous screenshot、current screenshot 和 action history，输出该 action 是否推进任务的 dense reward；GiGPO 相对 episode-level GRPO 在 MobileWorld GUI-Only 上从 14.5% 提到 17.1%。

它们共享的优点是：critic 的证据来自 Student 自己已经产生的 $(o_t,a_t,o_{t+1})$，不存在反事实 state 重建问题。可做两条路线：

1. **Value-gated OPD：** 只在 critic 高置信判定 Teacher proposal 优于 Student action 时施加 KL；
2. **Fork-point recovery：** 在失败 transition 处恢复到 $s_t$，采样替代 action，执行后由 critic 或 verifier 比较，形成局部 preference。

风险也很明确：critic 会继承视觉盲点。OpenComputer 的 120-task 对照中，hard-coded verifier 的 task-level human alignment 为 94.1%，LLM judge 为 79.2%；因此能访问程序状态时，应以 verifier 作为最终 reward，以 visual critic 负责无法程序化的局部 action-effect 判断。

## 6. 第三优先：BJudge 的 Wide-to-Single Distillation

BJudge 的 72.6% 不是 Agent S3 单次 rollout 的成绩。其最佳设置从 GPT-5 与 Claude Opus 4.5 各 5 条 rollout 中选择；Agent S3 + GPT-5 单次平均为 62.6%，同模型 10-rollout BJudge 为 69.9%，混合模型达到 72.6%。因此它证明了“多轨迹存在互补成功”，但不自动证明这些收益能被单策略内化。

直接对被选中轨迹做 SFT 有两个问题：

- 只覆盖 Teacher 成功轨迹的 state，未覆盖 Student 部署时会到达的错误 state；
- 多条 rollout 在第一步后环境 state 已分叉，不能在不同 state 的 action 间直接做 token preference。

更合理的研究设计是：

1. Student 生成 rollout；
2. 在其若干 prefix state 建快照；
3. 从每个快照并行生成短分支，而不是只比较完整、已分叉的轨迹；
4. 用 transition facts 加最终 verifier 对分支排序；
5. 在共同 prefix 上做 action preference 或局部 OPD。

这相当于把 BJudge 从“终局选轨迹”改造成“state-matched branch teacher”。它比 GTA1 更昂贵，但能监督 delayed consequence，是后续更强的选题。

## 7. 哪些组件只应部分蒸馏

### OS-Symphony

整套 OS-Symphony 不应被当作一个统一 Teacher。其组件的可蒸馏性不同：

- RMA 的 step verification、错误类型、stop/continue：高；
- milestone selection 和 history compression：中，Student 仍需保留上下文或可写 memory；
- Multimodal Searcher：低，收益来自实时外部教程，知识在部署时可能不可重建；
- Coder：低，动作空间不一致。论文的 Coder 消融从 63.61% 降到 57.42%，说明约 6.2 点增益不能误算成纯 GUI reasoning 提升。

其 RMA 消融提供了很好的 Teacher 组件证据：GPT-5-Mini 设置中，完整 LTM reflection 为 58.05%，无 reflection 为 54.38%，只用短期 memory reflection 为 54.01%。这也说明“加反思文本”本身不是目标，关键是 state-grounded transition evidence 与正确的长期状态选择。

### Agent S2

可蒸馏的是 Manager 的主动重规划、Worker 的 action proposal，以及“何时需要 OCR/visual grounding”的路由。结构化 `set_cell_values` 通过 UNO 直接改表格，不应假装成 GUI action。若要做严格 GUI-only 实验，应关闭 Structural Grounder，并固定 Visual/OCR Grounder；否则改成 harness-retention 设定。

### Syll

Syll 的 demo-guided replay 在单个 Stardew Valley 配对实验中从 instruction-only 的 2/3 phases 提升到 3/3，并将 off-trace actions 从 3 降到 0。但论文明确说明该消融是 $n=1$。它适合研究“课程式撤除 demonstration”：先给完整 keyframes/phase expectations，再逐步遮掉示范，让 Student 学会 phase tracking。现有证据不足以把它排在大规模 benchmark 已验证的方法之前。

## 8. 不适合严格 GUI-only OPD 的候选

### CoAct-1、Agent S3 与 StateAct

三者都表明 code/state action 很有价值，但这正是严格 OPD 的动作不对齐来源：

- CoAct-1 的 full hybrid 为 60.76%，GUI-only 为 50.68%，Programmer-only 为 35.73%；code 一步可以替代多步 GUI。
- Agent S3 将 coding agent 嵌入 GUI policy 的 action space；BJudge 的部分成功因此也依赖 code-GUI handoff，论文 12 个 judge failure 中有 4 个属于该类。
- StateAct 的主 Agent 不暴露鼠标键盘，只用 bash/file editor/state queries，GUI 是少量委派 fallback。其主要收益明确来自 act-on-state：移除该组件后 partial score 从 61.6% 降到 51.3%。

这些结果适合支持“为什么应该保留工具”的论文，不支持“GUI Student 已在权重中学会等价行为”的直接结论。可蒸馏的是 routing、验证习惯、任务分解和 GUI fallback 条件；程序状态与 code action 本身应保留。

### CLI-Anything

CLI-Anything 的论点就是 GUI 对 Agent 是有损、脆弱的控制面，应把软件 backend、typed commands、JSON state、render/export 和 verifier 暴露出来。它最适合：

- 生成高质量 artifact 与验证信号；
- 作为 GUI Student 的 privileged Teacher/verifier；
- 构造 semantic action 与 GUI action 的配对数据。

但从 CLI trajectory 到 GUI trajectory 不是普通蒸馏，而是跨动作空间的 trajectory translation。若部署仍要求 GUI-only，需要环境模型或执行器把一个 CLI semantic action 展开成多步 GUI policy；不能只对 CLI token 做 KL。

### OpenComputer

OpenComputer 不应和 Agent policy Harness 混为一谈。它提供 33 个应用、1000 个任务、平均每应用 17.7 个 verifier endpoints，以及 criterion-level partial reward。它最强的作用是：

- 给 GUI rollout 精确、可审计的终局 reward；
- 从同一 prefix 筛选成功和 near-miss 分支；
- 训练 stop、save、path correctness 与结构完成判断；
- 用于 rejection sampling、RL 和 verifier-gated OPD。

它的 verifier 使用 Student 部署时未必拥有的 backend state，但这不妨碍训练期打 reward；只有当我们要求 Student 从截图预测这些 hidden values 时，才会违反信息可重建性。

## 9. 推荐的研究路线

### Phase A：最小可证伪实验

做 **GTA1 Step-wise BoN → Single-Sample OPD**：同一开源 backbone、同一 grounder、同一 GUI DSL，Teacher $K=8$，Student $K=1$。核心指标不是只看最终成功率，而是：

- Student 回收了多少 Teacher Harness gain；
- action agreement、state coverage 与错误恢复率；
- 每次成功的 token、调用数、墙钟时间；
- 对未见应用、分辨率和更长 horizon 的迁移。

### Phase B：加入 transition critic

用 BJudge-style action-effect facts 或 RMA/PRM 对 Student transition 打分；比较 ungated OPD、value-gated OPD、fork-point preference。若有 OpenComputer verifier，则以程序检查为终局真值，visual critic 只做 dense shaping。

### Phase C：从 step scaling 扩展到 rollout scaling

把 BJudge 改造成共同 prefix 上的短分支比较器，研究 delayed consequence 能否被 single-rollout Student 内化。此阶段再比较单模型多样性与多模型 ensemble，避免一开始把 Harness gain 和更强 Teacher model 混在一起。

### Phase D：边界实验

选择同一批任务，构造三种 Teacher：GUI BoN、GUI+Verifier、Code/State Teacher。测量随着 privileged information 与 action mismatch 增加，OPD gain recovery 如何下降。这能把“可蒸馏性光谱”从分析判断变成实验曲线。

## 10. 最终排序

若只能选一个方向：**GTA1 式同状态、同动作空间的 test-time scaling distillation。**

若做第二个增强：**加入 BJudge/RMA/PRM 式 transition critic，做 value-gated 或 fork-point OPD。**

若追求更强但更难的贡献：**把 BJudge 从完整 rollout 选择改造成 state-matched branch teacher。**

不建议把“普通 function call”“加一个 code agent”或“统一 GUI/CLI 接口”本身包装成无 Harness OPD 创新。它们是重要系统能力，但要么部署仍保留 Harness，要么 Teacher/Student 动作空间不同；研究问题应明确写成 harness-aware distillation、cross-action trajectory translation 或 verifier-grounded learning。

## 11. 主要来源与证据限制

- [GTA1: GUI Test-time Scaling Agent](https://arxiv.org/abs/2507.05791)
- [Scaling Agents for Computer Use / BJudge / Agent S3](https://arxiv.org/abs/2510.02250)
- [Agent S2](https://arxiv.org/abs/2504.00906)
- [CoAct-1](https://arxiv.org/abs/2508.03923)
- [OS-Symphony](https://arxiv.org/abs/2601.07779)
- [StateAct](https://arxiv.org/abs/2607.22798)
- [OpenComputer](https://arxiv.org/abs/2605.19769)
- [ClawGUI](https://arxiv.org/abs/2604.11784)
- [CLI-Anything](https://arxiv.org/abs/2606.03854)
- [Syll](https://arxiv.org/abs/2606.07594)

本文的系统机制、表中数字和消融来自上述论文原文 HTML 版本。候选排序、OPD 适配方式和 Phase A-D 是基于这些证据提出的研究判断，不是论文已验证结论。尤其需要注意：不同论文使用 OSWorld、OSWorld-Verified、OSWorld 2.0、MobileWorld 等不同 benchmark、模型和 step budget，本文不按绝对成功率横向排名，只按 Harness 信息与动作接口对 OPD 的适配程度排名。