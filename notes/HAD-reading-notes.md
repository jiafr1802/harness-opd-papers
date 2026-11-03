# HAD 阅读笔记：Harness-Aware Distillation

## Summary

HAD 的核心不是把 harness 蒸馏进模型，而是在部署时保留同一个 harness，并显式训练小模型“如何依据 harness 改变行动”。它以学生 rollout 到达的状态为训练分布，用教师完整回复做基础蒸馏，再比较同一教师在有/无 harness 时的行动，形成 action-level 偏好。规则型有效性检查器只过滤与当前环境事实明确冲突的正行动。该方法在三个结构化文本 benchmark 上有效，但尚未证明能直接推广到开放式软件 Agent。

## 从 OPD 到 HAD

Agent 的轨迹天然以环境 step 组织：学生生成 reasoning 与 action，环境执行完整 action 后才返回新状态。所谓 on-policy，首先指训练状态来自当前学生的轨迹，而不规定 step 内必须使用哪种 token loss。

论文的标准 OPD baseline 在学生生成的 token 前缀上计算 student-to-teacher reverse KL。HAD 自身没有沿用这个 loss：在学生访问的状态 $\tilde{x}_t^\theta$ 上，带 harness 的教师采样完整回复 $\tilde{y}_t^\phi=(\tilde{z}_t^\phi,\tilde{a}_t^\phi)$，学生以 teacher forcing 做交叉熵。对教师样本取期望时，这相当于局部 forward KL 的 Monte Carlo 估计。因此 HAD 的基础项可概括为“on-policy 状态上的动态 SFT”，并与本文实现的 SOPD 基线基本一致。

普通响应蒸馏只告诉学生“带 harness 的教师做了什么”，不能单独识别哪些行动概率是因为 harness 才上升的。HAD 因此在同一学生状态上再次移除 harness 信息，得到教师行动 $a^-$，并把带 harness 的行动 $a^+$ 作为偏好项：

$$
\mathcal L_{\mathrm{pref}}=-\log\sigma\left(\beta[\log\pi_\theta(a^+\mid\tilde{x},z^\theta)-\log\pi_\theta(a^-\mid\tilde{x},z^\theta)]\right).
$$

它形式上类似 reference-free、action-level DPO，但偏好对来自同一教师的有/无 harness 条件，而非人类标签。更重要的是，两个 action 都接在学生自己的 reasoning $z^\theta$ 后评分；基础 response loss 则学习教师 reasoning 与 action。前者保证学生沿着实际生成的 reasoning 走时仍偏向 harness 所引导的行动，后者提供完整任务能力。区别不只在 $z$ 来自学生还是教师：response loss 模仿完整回复，preference loss 只比较 action，并专门提取 harness 引起的相对变化。若 $a^+=a^-$，偏好项没有梯度。

这也解释了图 1 的第二类错误：学生可能在语言上认识到“盘子未清洗”，却仍给出 `go to cabinet`。这不是教师必然错误，也不是 OPD 理论上无法学会；它说明普通模仿对少量 harness 关键决策只给出隐式信号。HAD 加入了针对这种差异的归纳偏置，并不构成对所有 OPD 的理论否定。

## 有效性检查器 $V$

$V(a\mid\tilde{x})\in\{0,1\}$ 不是模型、奖励函数或最优行动 ground truth，而是人工编写的确定性规则。它使用环境公开的 admissible commands、当前持有物和同状态失败历史，拒绝三类正行动：不在可用命令中、操作未持有物体、重复已经无效的动作。它不使用任务奖励、成功标签、未来状态，也不能识别“合法但策略错误”的行动。

最终目标为

$$
\mathcal L_{\mathrm{HAD}}=\mathcal L_{\mathrm{dist}}+\lambda V(a^+\mid\tilde{x})\mathcal L_{\mathrm{pref}}.
$$

过滤只关闭 preference loss；教师错误仍可能进入 response distillation。因此 $V$ 的作用是避免额外强化明显矛盾，而非验证完整正确性。图 1 第三类错误才由 $V$ 处理，第二类错误主要由学生 reasoning 上的 action preference 处理。

## 证据与边界

论文评测 ALFWorld、WebShop、ScienceWorld 三个 benchmark，并在 ALFWorld 上补充 Gemma 跨模型实验。ALFWorld 未见任务中，HAD 的成功率为 63.43%，最佳 OPD 基线为 47.01%；其停滞恢复率为 59.7%，其他模型最高为 50.0%。消融中，只有 response distillation 的成功率为 43.5%，去掉有效性过滤为 50.8%，完整 HAD 为 57.4%，说明行动对比贡献主要增益，$V$ 负责抑制错误偏好。

局限同样明确：三个环境都具有结构化动作、显式 inventory 和可靠失败反馈，便于手写 $V$。论文只测试不超过约 2B 的学生及文本环境。真实软件 Agent 可以检查工具 schema、文件存在性、编译错误和重复失败，但难以仅靠规则判断修改是否真正满足用户目标。

## 原文定位

主要依据：[HAD 论文](../HAD-Harness-Aware%20Distillation%20for%20Small%20Language%20Model%20Agents.pdf) 的 §3.1–3.3（目标与设计）、§4.1–4.5（实验和消融）、Appendix A.1（loss 与 validity checker）、Appendix B.2–B.4（harness、规则和 baseline 实现）。SOPD 对照来源为 Sun et al., *Step-level On-Policy Distillation: Interpolating between On-Policy Distillation and Supervised Fine-Tuning*, arXiv:2608.16333。
