# Harness 定义与 Formulation 对照

## 一句话结论

这几篇论文中的 **harness 不是同一个严格定义的数学对象**，而是同一类模型外部机制的不同实例：它在不直接改变 backbone 参数的前提下，读取任务、交互历史、外部信息或执行证据，并通过附加上下文、组织多次模型调用、维护记忆或控制 agent runtime 来改变模型的行为。

HAD、OPHSD、HERD 和 GUI-HARVEST 分别强调 harness 的四个层次：**文本状态提示、推理编排程序、有限容量外部记忆、完整可执行 agent runtime**。因此，讨论“蒸馏 harness”时必须说明被迁移的是哪一层，以及部署时 harness 是否仍然存在。

## 跨论文统一抽象

下面不是任何一篇论文原封不动给出的定义，而是便于比较的统一 formulation。设：

- `x` 是任务输入；
- `τ_<t` 是第 `t` 步之前的 observation、reasoning、action 和 tool result 历史；
- `e_t` 是可选的外部信息，如 memory、检索结果、verified answer 或 GUI 执行证据；
- `θ` 是 backbone 参数；
- `H` 是不属于 `θ` 的外部机制。

最小的 context-level harness 可以写成：

```math
c_t^H = H(x, \tau_{<t}, e_t),
\qquad
(z_t,a_t) \sim \pi_\theta(\cdot \mid x,\tau_{<t},c_t^H).
```

这里 `H` 产生额外上下文 `c_t^H`，模型仍负责生成 reasoning `z_t` 和 action `a_t`。更一般地，harness 可以维护内部状态并组织多次模型调用：

```math
q_t = T_H(q_{t-1}, o_t, u_t),
\qquad
u_t \sim \pi_\theta(\cdot \mid P_H(q_{t-1},o_t)),
\qquad
y = R_H(q_T).
```

其中 `P_H` 构造每次调用的 prompt/context，`T_H` 更新 harness 状态并执行工具或控制流，`R_H` 从终态读出答案。按这个抽象，harness 可以包含：

1. **Context**：prompt、状态摘要、规则、retrieved memory；
2. **Orchestration**：plan–solve、draft–verify、多轮调用和路由；
3. **Interface**：action schema、tool wrapper、环境 observation；
4. **Control**：验证、重试、错误恢复和终止；
5. **Persistent state**：skill、案例、residual memory 或优化 ledger。

并非每篇论文都包含以上全部组成。把所有 harness 都简称为“prompt”会低估 OPHSD 和 GUI-HARVEST；把它都称为“完整 agent runtime”又会夸大 HAD 中 harness 的权限。

## 逐篇定义

### HAD：固定的文本状态跟踪与行为监控

HAD 没有把 harness 定义成一个独立的通用程序 `H`，而是把它实现为一组**任务无关、跨环境共享的代码与文本模板**。它在每一步向 observation 增加两类信息：

- state tracking：当前位置、持有物、已访问位置等状态记录；
- behavior monitoring：重复动作、停滞、近期失败等条件触发提示。

可将其写为：

```math
h_t = H_{\mathrm{track}}(\tau_{\le t})
\oplus H_{\mathrm{monitor}}(\tau_{\le t}),
\qquad
x_t^h = x_t^0 \oplus h_t.
```

这里 `x_t^0` 是移除 harness 文本但保留同一学生交互历史后的输入，`x_t^h` 是带 harness 的输入。重要边界是：HAD 的 harness **只添加文本，不替模型选择或屏蔽 action**。训练和部署使用同一个固定 harness；HAD 的目标是让小模型学会正确利用它，而不是要求最终移除它。

HAD 用同一教师在 `x_t^h` 与 `x_t^0` 下的行动差异识别 harness 改变决策的位置：

```math
a_t^+ \sim \pi_\phi(\cdot \mid x_t^h),
\qquad
a_t^- \sim \pi_\phi(\cdot \mid x_t^0).
```

因此 HAD 中的核心问题是 **harness-conditioned action use**，而不是 harness 本身如何生成或演化。

### OPHSD：任务特定的多调用推理程序

OPHSD 把 inference harness 形式化为一个由模型调用和确定性状态转移组成的动态程序。它维护状态 `s_t`，第 `t` 次调用基于当前状态产生内容 `c_t`，确定性转移函数 `τ_H` 更新状态，最终 readout `ρ_H` 从终态抽取答案：

```math
c_t \sim p_\theta(\cdot \mid s_{t-1},x),
\qquad
s_t = \tau_H(s_{t-1},c_t),
\qquad
y = \rho_H(s_T).
```

若 harness 还读取 privileged input `z(x)`，例如 Memory Bank 或参考答案，则得到 `H_θ(y | x,z(x))`。论文的两个具体实例是：

- draft–verify：检索相似案例，先 draft，再用 confirmers/challengers verify；
- plan–solve：先构造计划，再按计划求解。

训练时用 `C[H_θ(x,z)]` 表示 harness 运行后的 terminal context。它只作为 teacher conditioning；裸 student 只看 `x`。所以 OPHSD 要内化的是 harness 诱导的**程序性推理分布**，不是复制整个 harness trace：

```math
D_{\mathrm{KL}}\!\left(
p_S(\cdot \mid x,y_{<n})
\,\Vert\,
p_T(\cdot \mid C[H_\theta(x,z)],y_{<n})
\right).
```

部署时移除 harness。只有能由 `x` 和参数知识重建的 procedure 可以完整蒸馏；实时网页、代码执行结果和动态环境反馈等新信息不能由无 harness student 凭空恢复。

### HERD：skill registry 与 high-residual memory

HERD 给出了最明确的集合式定义：

```math
H_t = (K_t,R_t).
```

其中 `K_t` 是有容量预算的 skill registry；每个 skill 包含 routing prefix 和 instructions，输入 `x` 被路由到匹配的 `K_t(x)`，后者成为 system message。`R_t` 是同样有容量上限的 residual memory，逐字保存完整的 `(prompt, verified answer)` 记录。

执行 harness 被写成：

```math
f(x;\theta,K_t),
```

即模型 `θ` 在 routed skill `K_t(x)` 作为 system message 时回答 `x`。`K_t` 保存可复用规律，`R_t` 保存 skill 与当前权重仍解释不好的高 residual 例外。需要注意，`R_t` 主要用于后续阶段的 drift probes，并不是直接 replay target。

HERD 中 harness 是 continual learning 的**快速、选择性、容量有限的外部层**，模型权重是缓慢、持久的内部层。每个 stage 先演化 `H_t`，再把 skill 提供的分布优势蒸馏进新权重 `θ_t`。最终部署只保留权重，不携带 `H_t`。

### GUI-HARVEST：可编辑的完整 agent runtime

GUI-HARVEST 使用最宽的工程定义。Harness 是冻结视觉语言 backbone 外部的**可执行源码系统**，包括 prompt/context construction、memory、action/tool interface、control flow、verification、error recovery 和 termination logic。它不是单个 prompt，也不是只有 memory。

若冻结 backbone 为 `π_θ`，当前 runtime 为 `H_i`，其 GUI 轨迹可抽象为：

```math
\tau \sim \mathrm{Run}(\pi_\theta,H_i,x,E),
```

其中环境 `E` 返回截图、tool output 和 action effect。GUI-HARVEST 根据多次运行证据提出受限源码 patch：

```math
H_{i+1} = \mathrm{Promote}(H_i + \Delta_i)
```

仅当代码检查、Search/Validation utility gate 和预注册 behavior gate 全部通过时才 promote，否则 rollback。这里学习发生在 `H` 的源码，而不是 `θ`；优化结束后部署的是冻结 backbone 加优化后的 harness `H_*`。

## 横向对照

| 论文 | Harness 的直接对象 | 是否更新 Harness | 是否更新模型 | 部署时 Harness |
|---|---|---:|---:|---:|
| HAD | 状态跟踪与行为监控产生的文本 | 否，人工固定 | 是 | 保留 |
| OPHSD | 多次模型调用、状态转移和 readout 组成的推理程序 | 否，任务特定固定程序 | 是 | 移除 |
| HERD | `H_t=(K_t,R_t)`：skill registry + residual memory | 是，每 stage 演化 | 是，每 stage 蒸馏 | 移除 |
| GUI-HARVEST | prompt、memory、tools、控制流、验证和恢复组成的 runtime 源码 | 是，证据驱动 patch | 否，backbone 冻结 | 保留优化后的版本 |

从窄到宽，可以把四篇的对象关系概括为：

```text
HAD text augmentation
  ⊂ OPHSD inference orchestration
  ⊂ GUI-HARVEST executable agent runtime

HERD 位于另一条轴上：它重点规定 harness 中“可积累的外部知识状态”如何表示、演化并进入权重。
```

这个包含关系只描述**系统能力范围**，不表示后一篇的方法或目标优于前一篇。

## 使用术语时的建议

后续写作中，可以把 harness 的通用定义固定为：

> Harness 是位于模型参数之外、在推理或交互过程中组织上下文、外部状态、模型调用、工具接口与控制逻辑，从而改变模型可观察行为的可执行机制。

随后必须补充三项限定：

1. **包含什么**：只是 context，还是 memory、tools 与 control flow；
2. **谁被学习**：更新模型参数、更新 harness，还是两者交替更新；
3. **部署形态**：harness 被保留、被移除，还是以优化后的版本保留。

在这套术语下：HAD 是 **learning to use a retained harness**；OPHSD 是 **internalizing a fixed harness**；HERD 是 **evolving then recursively internalizing a bounded harness**；GUI-HARVEST 是 **optimizing the executable harness while freezing the model**。

## 原文定位

- [HAD](../HAD-Harness-Aware%20Distillation%20for%20Small%20Language%20Model%20Agents.pdf)：§3.1–3.3、§4.1、Appendix B.2；尤其是 harness 只向 observation 添加文本、不选择或屏蔽 action 的实现边界。
- [OPHSD](../OPHSD.pdf)：§3.1–3.3、Figure 1、Appendix C；尤其是动态 harness 的状态转移定义和 `C[H_θ(x,z)]`。
- [HERD](../HERD-Harness-Evolving%20Recursive%20Distillation%20for%20Continual%20Learning.pdf)：§2.1–2.4；尤其是 `H_t=(K_t,R_t)` 与 `f(x;θ,K)`。
- [GUI-HARVEST](../GUI-Agent-Harness-Evolution.pdf)：§2、§3.1–3.5、Figure 2；尤其是 harness scope、源码 patch 和 promotion gates。
