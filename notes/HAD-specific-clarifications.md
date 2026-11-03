# HAD 针对性理解：从轨迹、前缀到两个 Loss

## Summary

理解 HAD 的关键是把三个维度分开：**状态从谁的轨迹来、当前回复使用谁的 token 前缀、loss 在模仿完整回复还是比较 action**。HAD 不是传统 reverse-KL OPD，而是“学生轨迹上的 SOPD 式 response distillation + 学生 reasoning 上的 action preference”。有无 harness 的教师对比不是因为普通 OPD 原理上学不会，而是给小模型显式标出 harness 真正改变决策的少量位置；$V$ 则只处理明显无效的教师正行动。

## 一次训练记录究竟怎样产生

在第 $t$ 个环境 step，历史 $\tilde{x}_t^\theta$ 包含任务、此前所有 observation、harness 信息，以及学生过去生成的 reasoning/action。它是**学生行为造成的状态**，所以数据是 on-policy。学生在这里采样当前回复 $y_t^\theta=(z_t^\theta,a_t^\theta)$，环境只执行学生的 $a_t^\theta$，再进入下一状态。教师的行动只是监督标签，不会改写轨迹。

同一 $\tilde{x}_t^\theta$ 还被查询两次：带 harness 的教师生成 $(z^+,a^+)$；去掉历史中的 harness 文本后，同一教师生成 $(z^-,a^-)$. “去掉 harness”并不会重新跑一条环境轨迹，过去由学生造成的 observation 和 action 都保留。

因此，**step-level 是环境数据的组织方式，token-level 是 step 内的训练方式**。环境只有执行完整 action 后才变化，但一个 step 的 Thought/Action 仍由多个 token 构成。Agent OPD 通常需要学生轨迹，却不代表所有方法必须使用相同 token loss，也不一定每一步都查询教师。

## 三种训练方式不要混在一起

论文的标准 OPD baseline 在学生当前回复的 token 前缀 $y_{1:j-1}^\theta$ 上计算

$$
\mathrm{KL}\!\left(
  \pi_\theta\!\left(\cdot \mid \tilde{x}, y_{1:j-1}^\theta\right)
  \,\middle\|\,
  \pi_\phi\!\left(\cdot \mid \tilde{x}, y_{1:j-1}^\theta\right)
\right)
$$

即 token-level reverse KL。

HAD 的 response loss 不用这个 KL。教师先采样完整回复 $y^+=(z^+,a^+)$，学生再以 teacher forcing 逐 token 拟合它：预测第 $j$ 个 token 时，输入的是教师目标的前 $j-1$ 个 token，而非学生自由生成的前缀。它在优化形式上是交叉熵/SFT；因状态来自当前学生，仍是 on-policy distillation。对教师样本取期望时，它是 $D_{KL}(\pi_\phi\|\pi_\theta)$ 的 Monte Carlo 估计，也就是 forward-KL-like。本文实现中，这一项与 SOPD 基线的核心做法相同。

Preference loss 则把 $a^+$、$a^-$ 都接到学生当前 reasoning $z_t^\theta$ 后评分：

$$
-\log \sigma\!\left(
  \beta\!\left[
    \log \pi_\theta\!\left(a^+ \mid \tilde{x}, z^\theta\right)
    - \log \pi_\theta\!\left(a^- \mid \tilde{x}, z^\theta\right)
  \right]
\right)
$$

它像无 reference 的 action-level DPO：不模仿教师 reasoning，只要求学生沿自己的 reasoning 生成时更偏好 $a^+$。action 若包含多个 token，分数仍是这些 token 的平均 log-probability。因此两个 loss 的区别不只是 $z$ 来自教师还是学生：response loss 学完整 $(z^+,a^+)$；preference loss 只学习 $a^+>a^-$，专门提取 harness 引起的相对变化。

## 为什么学生和教师都有 harness 还需要对比

普通蒸馏只观察带 harness 的教师分布 $P^+(a)$。它理论上可以学会利用 harness，但无法单独标出“这个 action 是本来就常见，还是因为 harness 才变得更可能”。HAD 的有/无 harness 对比强调近似 $\log P^+(a)-\log P^-(a)$；只有教师行动发生变化时才产生有效梯度。这是一种面向有限数据和小模型的归纳偏置，不是对 OPD 的不可能性证明。

这正好区分图 1 的两类疑惑。第二行中，带 harness 的教师可以完全正确；学生甚至说出“盘子没洗”，却在自己的 reasoning 后选了 `go to cabinet`。Preference loss 直接在该学生 reasoning 后提升 `go to sink` 相对 `go to cabinet` 的概率。若教师有无 harness 都给出同一 action，HAD 在这里没有额外帮助。

第三行才是教师错误：harness 记录 `holding nothing`，教师却给出 `clean plate`。$V(a^+\mid\tilde{x})$ 不是模型、reward 或最优 action GT，而是手写布尔规则；它检查 action 是否在 admissible commands 中、是否操作未持有物体、是否重复同状态下已失败的动作。$V=0$ 只删除该 preference pair，错误回复仍进入 response loss，所以它避免“额外强化”明显矛盾，并不能保证正确性，也识别不了合法但策略错误的 action。

## 证据边界

HAD 只在 ALFWorld、WebShop、ScienceWorld 三个结构化文本 benchmark 上验证；Gemma 实验仍是 ALFWorld，不是第四个 benchmark。消融中，仅 response distillation 的成功率为 43.5%，去掉 $V$ 为 50.8%，完整 HAD 为 57.4%。这些环境提供动作列表、inventory 和可靠失败反馈，适合规则型 $V$；论文尚未证明该设计能直接覆盖开放式软件 Agent 的语义正确性。

## 原文依据

以上对应 [HAD 原文](../HAD-Harness-Aware%20Distillation%20for%20Small%20Language%20Model%20Agents.pdf) §3.1（式 2–7）、§3.2–3.3、§4.1–4.5、Appendix A.1 与 B.2–B.4；SOPD 题名为 Sun et al., *Step-level On-Policy Distillation: Interpolating between On-Policy Distillation and Supervised Fine-Tuning*, arXiv:2608.16333。
