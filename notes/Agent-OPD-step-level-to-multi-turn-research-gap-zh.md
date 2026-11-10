# Agent OPD：从局部 Step Matching 到 Multi-turn Temporal Credit 的 Research Gap

调研日期：2026-11-10

## 1. 一句话结论

现有 Agent OPD 已经能够在 multi-turn 学生轨迹的每个步骤上提供 teacher supervision，但多数方法仍然是**把访问到的状态视为固定输入，做局部 action/token matching**；它们尚未完整解决早期动作如何通过环境转移导致后续状态与最终结果的**跨 turn 因果 credit assignment**。

最准确的研究空白是：

> 从“在 multi-turn 轨迹上做 local step matching”，走向“对带环境转移的完整 trajectory distribution 做可靠、低方差、因果一致的 policy distillation”。

## 2. 先澄清术语：Step-wise 不等于 Single-turn

`step-wise` 与 `multi-turn` 描述的是不同维度。

| 术语 | 描述的问题 |
| --- | --- |
| Single-turn / Multi-turn | 一条轨迹是否包含多轮 action、observation 和环境变化 |
| Token-level / Step-level / Trajectory-level | 蒸馏信号或 credit 以什么粒度分配 |

因此，SOD 虽然名为 **Step-wise On-policy Distillation**，本身处理的就是 multi-turn tool-integrated trajectory。它把一条多轮轨迹划分为若干 model response steps，再为每个 step 设置不同蒸馏权重。

不能把研究路线简单写成：

```text
step-level OPD -> multi-step OPD
```

更准确的两条轴是：

```text
交互轴：single-turn response -> multi-turn environment interaction
credit 轴：local matching -> cross-turn temporal credit
```

当前大量工作已经到达“multi-turn + local/step-wise matching”，但还没有完全到达“multi-turn + causal trajectory credit”。

## 3. 为什么普通 OPD 在 Single-turn 中相对简单

在普通自回归生成中，状态通常是 prompt 加学生已经生成的 token prefix：

```math
c_t=(x,y_{<t}).
```

学生生成完整 response，teacher 在学生 prefix 上提供下一 token 分布。典型局部目标是：

```math
\mathcal L_{\mathrm{local}}
=
\mathbb E_{y\sim\pi_\theta}
\sum_t
D_{\mathrm{KL}}\left(
q(\cdot\mid c_t)
\parallel
\pi_\theta(\cdot\mid c_t)
\right).
```

这里的问题虽然也有长程依赖，但每一步主要是在同一条文本序列中追加 token。teacher 与 student 可以在相同 prefix 上比较下一 token 分布。

## 4. Multi-turn Agent 改变了什么

Agent 的轨迹包含模型动作和环境观察：

```math
\tau=(o_0,a_0,o_1,a_1,\ldots,o_T).
```

状态不再只是生成文本，而是完整交互历史：

```math
h_t=(o_0,a_0,o_1,a_1,\ldots,o_t).
```

每个动作会通过环境 transition 改变下一状态：

```math
o_{t+1}\sim P(\cdot\mid h_t,a_t).
```

例如：

```text
Turn 1：进入错误目录
Turn 2：在错误目录搜索代码
Turn 3：修改错误文件
Turn 4：运行测试失败
```

Turn 2 到 Turn 4 不是互相独立的错误，而可能都是 Turn 1 的后果。后面的失败信息需要反过来说明 Turn 1 的动作为什么不好。

## 5. 核心理论 Gap：在错误状态上纠正，不等于教会模型别进入该状态

局部 OPD 目标在每个已经访问的状态上匹配 teacher：

```math
\mathcal L_{\mathrm{local}}
=
\mathbb E_{\tau\sim\pi_\theta}
\sum_t
D_{\mathrm{KL}}\left(
q(\cdot\mid s_t^T)
\parallel
\pi_\theta(\cdot\mid s_t)
\right).
```

它回答的是：

> 学生已经来到状态 `s_t` 后，这一步应该采取什么动作？

但 multi-turn agent 还需要回答：

> 前面的哪个动作导致学生后来进入这个错误状态？

OPD survey 在 §2.7 直接总结：

> Learning at student states does not by itself assign credit for reaching them.

即：

```text
在错误状态上教下一步
!=
教会学生不要进入这个错误状态
```

如果对 state distribution 停止梯度，普通 per-state KL 不会自动包含“动作改变未来状态分布”的梯度项。真正的 trajectory objective 还需要考虑早期动作对后续状态和后续 teacher–student discrepancy 的影响。

## 6. 文献明确提出的三个主要 Gap

### 6.1 TCOD：Trajectory-Level KL Instability

论文：[TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](https://arxiv.org/abs/2604.24005)

TCOD 明确指出，已有 OPD 主要为 static、single-turn reasoning 设计，而 multi-turn agent setting 仍缺少研究。它观察到：

1. 随着 turn 增加，student–teacher KL 持续升高；
2. KL escalation 与 success-rate collapse 同时出现；
3. 学生早期动作和环境 observation 被追加到后续 history，形成跨 turn 的 causal coupling；
4. 学生逐渐进入 teacher effective support 之外的状态，teacher supervision 变得不可靠。

TCOD 将这一失败模式称为：

> **Trajectory-Level KL Instability**

它的解决方法是控制训练时暴露给学生的 trajectory depth：

- TCOD-F2B：学生先只学前几个 turn，再逐渐延长完整 rollout；
- TCOD-B2F：teacher 先把环境导航到接近成功的中间状态，学生从后半段开始学习，再逐渐把起点向前移动。

TCOD 主要解决：

> 学生尚未掌握短 horizon 时，直接暴露完整 horizon 会不会造成 KL collapse？

它没有完整解决“最终失败应归因到哪个早期动作”。其每个状态上的基本监督仍是 KL matching。

### 6.2 SOD：Tool-induced State Drift 与错误级联

论文：[SOD: Step-wise On-policy Distillation for Small Language Model Agents](https://arxiv.org/abs/2605.07725)

SOD 进一步区分 text-only reasoning 与 tool-integrated reasoning：

- 文本推理通常逐 token 渐进偏离；
- tool call 会返回几十到几百 token 的 observation；
- 一个错误 tool call 会造成不连续的 state jump；
- 后续推理从被污染的 observation 开始，可能继续产生错误调用；
- 连续错误会把 student 推进 teacher 极少见的组合状态。

其错误链可以概括为：

```text
错误工具调用
-> 错误或异常 observation
-> 下一步上下文被污染
-> 再次错误调用
-> student-teacher divergence 加速扩大
-> teacher 熵和梯度方差升高
-> supervision SNR 下降
```

SOD 为每个 reasoning step 计算 student–teacher divergence proxy `d_k`，再设置 step weight `w_k`：

- divergence 升高：降低当前及后续不可靠区域的蒸馏权重；
- 学生恢复：重新提高权重；
- 避免一次错误后把全部后续 supervision 永久 mask 掉。

SOD 主要解决：

> 轨迹中哪些 step 的 teacher supervision 仍可信？

但 divergence 是 teacher–student disagreement proxy，不是动作正确性或因果责任的直接证明。SOD 调整局部 loss 权重，并没有计算完整的 future return-to-go。

### 6.3 OPRD：反馈具有不同 Temporal Scope

论文：[OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents](https://arxiv.org/abs/2610.21496)

OPRD 指出，long-horizon coding 中的 corrective information 存在不同时间范围：

- “不要修改测试文件”“保持 API 兼容”是整条轨迹持续有效的 constraint；
- “刚才的 `str_replace` 没有匹配”“当前 pytest 报 TypeError”通常只应影响紧接着的下一步。

如果所有信息都作为 trajectory-wide prompt：

- 局部错误可能在长上下文中被稀释；
- 已经过时的局部信息可能影响无关的后续决策。

如果所有信息都只作用一次：

- 长期任务约束又可能很快被忘记。

因此 OPRD 区分：

- **Global Reminder**：跨 assistant turns 持续可见；
- **Turn-local Reminder**：只在特定 observation 后的下一次决策中可见。

OPRD 还按最终 executable verifier 结果分流：

- 成功 rollout：autoregressive self-imitation；
- 失败 rollout：Reminder-conditioned reverse KL。

OPRD 主要解决：

> 哪条 corrective information 应在什么时间范围内影响 teacher target？

它改善了 feedback localization，但仍主要是在学生实际 prefix 上做局部分布匹配。Global Reminder 对多个位置生效，也不等同于识别哪个早期 action 对最终失败负有因果责任。

## 7. Multi-turn Agent OPD 的主要难点

| 难点 | 具体问题 |
| --- | --- |
| State identity | action 必须在学生当时真实看到的 observation、memory 和 context 下被评分 |
| Environment transition | action 会改变真实环境，而不仅是追加一个 token |
| Error compounding | 早期错误会改变所有后续状态，导致错误级联 |
| Teacher reliability | teacher 不一定能从学生制造的异常状态中可靠恢复 |
| Cross-turn credit | 后面出现的失败应如何归因到早期 action |
| Delayed feedback | 最终测试结果可能在几十或几百步后才出现 |
| Temporal scope | 哪些反馈应持续，哪些只纠正下一步 |
| Intervention shift | teacher 替换 action 后，后续 history 就不再是纯 student rollout |
| History reconstruction | replay 时必须恢复原始可见状态，不能偷偷加回已被 memory 丢弃的信息 |
| Horizon coverage | 短 rollout 省成本，但可能永远训练不到后期关键状态 |
| Staleness | student 更新后，旧轨迹不再来自当前 policy |
| Reward–teacher conflict | teacher 局部偏好未必提高最终 environment reward |
| Compute cost | 完整环境 rollout、teacher scoring 和 verifier 都很昂贵 |

## 8. 为什么 Multi-turn 比 Long-CoT 更难

Long-CoT 主要是在同一个环境状态上生成更长文本。Multi-turn agent 则每一轮都可能改变环境：

```text
Long-CoT:
prefix -> more model tokens -> more model tokens

Multi-turn agent:
history -> action -> environment transition -> observation
        -> new action -> another transition -> new observation
```

因此，即使两者 token 长度相同，multi-turn agent 仍多出：

- 外部环境随机性；
- 不可逆副作用；
- observation reconstruction；
- 工具权限与 action validity；
- delayed terminal outcome；
- teacher intervention 对后续 state distribution 的改变。

## 9. 当前方法各自覆盖的部分

| 方法 | 主要解决的问题 | 尚未解决的核心 |
| --- | --- | --- |
| SOD | 按 step reliability 调整 supervision 强度 | divergence 不等于因果 credit |
| TCOD | 从短到长暴露 horizon，缓解 KL collapse | curriculum 不定位早期责任动作 |
| OPRD | Global/Turn-local temporal scope；成功/失败 routing | reminder scope 不等于 trajectory gradient |
| HINT-SD | 用 hindsight feedback 监督选定 action spans | span localization 仍依赖 heuristic/judge |
| Prefix Replay | 在存储 history 上采样新 student action，降低交互成本 | replay state 未必代表当前学生分布 |
| MemOPD | 保持 memory-compressed history 的 state identity | 解决对齐，不自动解决 delayed credit |
| TurnOPD/STRIDE | completion-aware horizon control、停止与重启 | 省成本与 later-state coverage 存在权衡 |
| Teacher intervention / handoff | 在 divergence 大时接管或桥接 | 改变后续 state distribution，重要性修正困难 |

## 10. 当前仍然开放的核心问题

### 10.1 Local correction 与 temporal credit 的统一

如何同时保留 teacher 在当前状态上的 dense distribution guidance，并将后续结果反向归因给造成这些状态的早期动作？

### 10.2 Teacher 在异常学生状态上的可靠性

Teacher standalone success 很高，不代表它能从学生连续制造的错误 tool states 中恢复。需要判断：

- teacher 是否理解当前异常状态；
- teacher 建议是否真的让后续轨迹恢复；
- 高 disagreement 是有价值纠正，还是 teacher 自己失去校准。

### 10.3 Counterfactual consequence

一个 teacher-preferred action 是否更好，通常必须真的执行才能看到 observation。只比较 next-token logits 无法知道 action 的长期环境后果。

### 10.4 Intervention 后还算不算 On-policy

Teacher bridge、替换动作或 B2F teacher prefix 会改变后续状态。被修改后的轨迹可用于学习，但不能简单宣称整条 history 仍来自当前 student policy。

### 10.5 Horizon 与训练成本的联合分配

有限预算应当花在：

- 更多 fresh student rollouts；
- 更深的 trajectory；
- 更多 teacher queries；
- 对旧状态进行更多 optimizer updates；
- 更昂贵的 verifier 与 counterfactual execution。

这些资源的边际价值相互依赖，目前还没有统一的最优分配规则。

### 10.6 从短交互扩展到 LongHorizon Harness

ALFWorld、WebShop、Python interpreter 或 SWE-bench agent 已经提供 multi-turn evidence，但 LongHorizon-Harness 还包含：

- 跨轮持久审计状态；
- fresh executor context；
- Manager/Executor/Auditor 权限隔离；
- GUI/CLI 异构 action space；
- 独立环境审计和恢复。

如何在保留必要 runtime state 与独立验证的前提下，把可学习的规划、核验和恢复策略蒸馏进权重，仍是更高一级的开放问题。

## 11. 对未来研究的一个可操作 formulation

一个更完整的 multi-turn OPD 目标需要区分三类信号：

```math
\mathcal L
=
\mathcal L_{\mathrm{local\ teacher}}
+
\lambda_{\mathrm{temporal}}\mathcal L_{\mathrm{future\ credit}}
+
\lambda_{\mathrm{outcome}}\mathcal L_{\mathrm{verifier}}.
```

其中：

1. `local teacher`：在学生访问状态上提供下一 action/token 分布；
2. `future credit`：把后续 discrepancy、recovery 或 failure 归因给更早 action；
3. `verifier`：保证 teacher preference 与真实任务成功方向一致。

还需要 state-level 约束：

- teacher 与 student 针对同一个真实决策状态；
- observation 和 memory history 可忠实重建；
- teacher-only hindsight 明确标为 privileged view，不能偷偷进入 student state；
- 若 teacher action 改写 history，应显式记录 mixed-policy distribution。

## 12. 最终判断

从 single-turn OPD 到 multi-turn Agent OPD 的 gap 已经被 TCOD、SOD、OPRD 和 Agentic OPD survey 明确提出，核心包括 error compounding、teacher support decay、trajectory-level KL instability、temporal scope 和 cross-turn credit。

其中最本质的缺口仍然是：

> 现有方法擅长判断“当前 step 应不应该听 teacher”，也开始控制“训练到多深”和“哪条反馈影响哪些 turn”；但它们尚未普遍解决“后续环境结果如何以因果正确、低方差的方式更新早期动作”。

## 13. 主要来源

- A Survey of On-Policy Distillation for Large Language Models, arXiv:2604.00626v5，尤其 §2.7、§7、§8、§11
- TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents, arXiv:2604.24005v3
- SOD: Step-wise On-policy Distillation for Small Language Model Agents, arXiv:2605.07725v3
- OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents, arXiv:2610.21496v2
- HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents, arXiv:2605.17873
- Multi-Turn On-Policy Distillation with Prefix Replay, arXiv:2607.04763
- MemOPD: On-Policy Distillation through Memory State Alignment for Long-Horizon Agents, arXiv:2608.07068
- Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation, arXiv:2609.16937