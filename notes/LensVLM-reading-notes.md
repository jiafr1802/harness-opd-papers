# LensVLM 阅读笔记：压缩视觉上下文中的选择性展开

> **论文**：Roy Xie et al., *LensVLM: Selective Context Expansion for Compressed Visual Representation of Text*（NeurIPS 2026）  
> [arXiv v4（2026-10-09）](https://arxiv.org/abs/2605.07019) · [原版 PDF（33 页，高清矢量图）](https://arxiv.org/pdf/2605.07019v4) · [arXiv HTML（附独立论文原图）](https://arxiv.org/html/2605.07019v4) · [Apple 官方代码](https://github.com/apple-aiml-research/ml-lensvlm) · [模型](https://huggingface.co/apple/LensVLM-9B)  
> 本笔记区分**论文方法和实验事实**与**对 Harness OPD 的个人研究思考**。数据/章节以 v4 为准。

## 一句话总结

**先把整篇长文档渲染成多页高度压缩的图像，让模型用这些缩略图“找可能相关的页”；找到了再通过 Expand 工具读这一页的原始文本、OCR 或高清图，继续推理并回答。** 模型不需要从完全模糊的字符中“猜出”细节：原始内容保存在外部，按需读取。SFT + DAPO 的重点是让 9B VLM 真正学会选择、调用工具和使用证据。

## 论文核心图（优先使用作者原图，不用模糊截图）

### Figure 2：方法总览（**必看**）

![LensVLM Figure 2：多轮页面选择、Expand 与 SFT/RL 训练流程；作者原图](https://arxiv.org/html/2605.07019v4/f2.png)

[单独查看 Figure 2 原始 PNG](https://arxiv.org/html/2605.07019v4/f2.png) · [PDF 第 3 页高清版本](https://arxiv.org/pdf/2605.07019v4#page=3) · [论文原始图集（HTML）](https://arxiv.org/html/2605.07019v4)

*左边：(a) 压缩页面、THINK、EXPAND、回答；右边：(b) Grounded trajectory generation → SFT → RL。图片直接引用 arXiv v4 的原始 Figure 2，不是二次生成示意图。*

其他重点图（原论文 PDF 中可缩放查看原始高分辨率 / 矢量版本）：

- [Figure 1：问答准确率与 Effective Compression Rate 的权衡（PDF 原版）](https://arxiv.org/pdf/2605.07019v4#page=2)
- [Figure 3：多数据集准确率—压缩率比较（原论文全文查找 Figure 3）](https://arxiv.org/html/2605.07019v4#S4.SS2)
- [Figure 5：Original Text / OCR / Zoom 模态对照（原论文 Section 5.2）](https://arxiv.org/html/2605.07019v4#S5.SS2)
- [Figure 6：Evidence selection 和 attention 可视化（原论文 Section 5.3）](https://arxiv.org/html/2605.07019v4#S5.SS3)

> **图像存储说明**：Figure 2 为文中嵌入的作者原始 PNG 的外部直链；其他图提供论文原始矢量 PDF / HTML 入口，并**未上传成本仓库的本地图片副本**。若日后需要离线保管，应从论文 PDF 或作者源文件导出，不能把当前链接当作已上传的本地 assets。

## 1. 为什么要这么做？压缩与可读性矛盾

传统长上下文直接输入原文要占用很多文本 token。LensVLM 把原始文本 $\mathcal D$ 排版成 $K$ 张页面图像 $\mathcal V=(v_1,\ldots,v_K)$；视觉编码器将其变成较短的 visual token 序列。压缩越大，输入越省，但字越小，文字级信息可能在渲染或视觉编码过程中丢失。

作者的关键观察不是“强视觉模型总能读清压缩字符”，而是：

- **定位证据在哪页**可能只需要章节标题、少量关键词、内容布局等粗粒度线索；
- **精确提取答案**可能需要姓名、数字、日期等完整字符；
- 因此在推理时，先用便宜的全局图像覆盖文档，再对少数页面进行高保真展开。

这不是无损压缩。**没有 Expand 工具时，已经不可分辨的任意字符不可能凭空恢复。**

## 2. Expand(k) 究竟是什么？是否读取原始文本？

论文 §3.2 把工具写为 $\operatorname{Expand}(k)$：输入第 $k$ 张压缩页面的索引，返回对应源内容的高保真表示 $\tau_k$。**不是三个互相竞争的独立 Harness，而是同一种“按页展开”的操作采用不同返回模态。**

| 返回形式 | 工具实际给模型什么 | 是否要求可访问原文 | 适合场景 / 限制 |
| --- | --- | --- | --- |
| **Original Text** | 第 $k$ 页对应的**源文本字符串** | **是**，必须保存来源文本及页码映射 | 原始文本渲染图像的主实验；精确字词最可靠；不能还原视觉布局 |
| **OCR Text** | 对原页或高清图 OCR 得到的文本 | **否**，有可识别的页面图像即可 | 扫描 PDF / 只有图像时；识别错误和布局丢失会影响结果 |
| **High-resolution Image / Zoom** | 第 $k$ 页的**高清页面图像** | **否**，但需保留该页的高清图像 | 原生文档、表格、排版、图文位置关系重要时更合适；仍消耗视觉 token |

**特别澄清**：

- **Original Text 不等于 OCR**：前者直接查源文本；后者识别图像中的字。
- **Zoom 不等于 OCR**：模型直接看到更多像素以及布局，不一定转成文本。
- 对于已经压缩得不可读的页面，模型并不是“自发恢复”这些字。它通过索引从外部保存的源数据中拿到内容。
- 选择的是**页面**（每次一页），不是原论文中任意大小的区域 crop；若要读取多页，需要连续多个回合展开。
- 官方文本演示使用 `read_page(page=10)`，返回 Page 10 对应的文字内容（见 [官方 README 示例](https://github.com/apple-aiml-research/ml-lensvlm#quick-start)）。

### 工具选择是模型怎么做出来的？

模型一次接收全部压缩页面和问题，自己生成推理及 `Expand(page=k)` 工具调用；论文没有在主方法里额外训练一个独立的 page-retriever。工具返回后，模型结合新证据继续选择或回答。

**页面选择正确率不是天然很高**：Appendix E 中未经训练的 Base VLM 选页准确率仅 25.8%（5×）、18.0%（10×）；SFT+RL 以后是 76.8%（5×）、71.0%（10×）、52.1%（15×）。这是论文“教会模型用压缩页面定位”的关键证据。

### 输入压缩率与有效压缩率不是一回事

$$
C_{\mathrm{in}}=\frac{N}{\sum_{k=1}^{K}n_k},
\qquad
C_{\mathrm{eff}}=\frac{N}{T_{\mathrm{reader}}}.
$$

- $N$：原文文本 token 数；$n_k$：第 $k$ 页对应的 visual tokens。
- **ICR** 只比较初始原文和压缩页面的 token 数；
- **ECR** 包括推理时展开返回的内容等实际 reader 输入 token，因此通常 **ECR ≤ ICR**。评估压缩收益要看 ECR，不能只看图片一开始有多小。

## 3. 训练：先 Hard Filtering → 合成工具轨迹 SFT → DAPO

### 3.1 Hard Samples 为什么要筛选？

先让 **Base Qwen3.5-9B VLM** 只看压缩图像和问题，**不给 Expand 工具**。回答由 LLM judge 判断正确与否。错了的样本定义为 hard，正确的是 easy。

这里的 Hard 的含义是：**在当前模型和压缩输入条件下，无工具模式答错**；不是证明问题从信息理论上不可解。

Appendix G（v4 Table 17 / Table 18）：

| Dataset | 样本数 | Hard | Hard 比例 |
| --- | ---: | ---: | ---: |
| Natural Questions | 36,600 | 28,877 | 78.9% |
| HotpotQA | 89,364 | 61,482 | 68.8% |
| MuSiQue | 14,503 | 12,920 | 89.1% |
| HELMET | 2,678 | 1,923 | 71.8% |
| **合计** | **143,145** | **105,202** | **73.5%** |

作者将 **80% hard** 用于 SFT，剩余 **20% hard 与 easy 混合后再子采样**用于 RL（实际 RL 训练约 15K），不是所有 143K 样本都进入两个阶段。

### 3.2 Teacher 到底在做什么？不是训练 397B

论文的 synthetic trace generator 是 **Qwen3.5-397B**。它获得：

- 问题 $q$；
- 已标注的 **正确证据原文** $\mathcal D_{E^*}$；
- **标准答案** $a^*$。

然后合成「推理 → Expand 相关页面 → 证据 → 最后回答」的训练示范：

$$
y^*\sim\pi_{\mathrm{synth}}(\cdot\mid q,\mathcal D_{E^*},a^*).
$$

**397B 本身不是被 SFT/RL 更新的目标**；Teacher 甚至提前知道答案，因此这里是 *answer-conditioned trajectory synthesis*，不能解释为 397B 自主从模糊图像中找对证据。被训练的是 **Qwen3.5-9B-Base → LensVLM-9B**。

作者解释：直接让强 Teacher 看高度模糊的页面自主产出轨迹容易出现错误与幻觉，所以先给正确证据和答案，合成相对可靠的工具轨迹。

### 3.3 SFT 哪些 token 有 loss？

SFT 对 **模型自己生成**的推理文本、工具调用、回答等计算交叉熵；`Expand` 返回的 `tool_response` 是供模型读取的输入，**不对工具返回文本计算 loss**。

$$
\mathcal L_{\mathrm{SFT}}
=-\sum_{i\in\mathcal A}
\log\pi_\theta(y_i^*\mid\mathcal V,q,y_{<i}^*),
$$

其中 $\mathcal A$ 仅包含 VLM 自己生成的 token 位置。这不等于工具返回内容对训练毫无影响：它仍会作为上下文改变后续生成分布。

### 3.4 为什么 SFT 后还需要 RL？

SFT 轨迹来自**已知正确证据的合成 Teacher**；推理时的小模型却必须根据**自己的模糊页面判断**究竟该展开哪一页。SFT 可能只学到工具调用的形式，RL 则让模型在自己采样的页面选择和回答轨迹上优化。

使用 **DAPO**；奖励为：

$$
R=0.7c+0.3c\,u,
$$

其中 $c=1$ 表示最终答案正确，$u=1$ 表示曾调用 Expand。

请注意：奖励 **没有直接把选中 GT evidence page 作为逐步监督**，也**没有对多次展开显式成本惩罚**。只要回答正确且用过工具就有额外奖励，所以确实鼓励工具调用。最大 6 回合，平均每条轨迹 1.3 次 Expand。

实现：SFT/RL 均冻结视觉编码器和 projector，只更新语言模型部分；SFT 约 2 epochs，RL 使用 VERL/DAPO，约 15K 数据（论文 Appendix D.2）。

## 4. 结果到底证明什么？

### 主结果（7 个文本 QA benchmark 平均）

- Full Text（直接读全原文）约 **72.4%**；
- **5× 输入压缩**时，无工具 Compressed Image 只有 **31.3%**；
- Base + Expand（未经后训练的工具使用）约 **39.2%**；
- SFT only 约 **63.7%**；
- **LensVLM 经过 SFT + RL 约 68.9%**，实际 **ECR 4.3×**。

作者还报告在不同压缩设置下依然优于各种文本/视觉压缩和检索 baseline，且对 code 和原生文档有迁移。

**解读**：性能提升主要来自「按需获得可读的高保真证据 + 学会定位和使用工具」，而不是训练出能从无限模糊图像里识别任意字符的视觉模型。

### Original Text、OCR 和 Zoom 的实验证据

论文 §5.2 对**渲染文本**的对照发现：

1. No-tool 纯 reasoning trace 不能弥补根本看不清字的信息缺失；
2. **Original Text 最好**；没有源文本时 OCR 是可行替代；
3. 对同样的渲染文本，**Zoom 通常落后于 Text/OCR**，因为文字直接以 token 给模型的保真度更高。

但在 **MMLongBench-Doc 原生多模态 PDF**，顺序可能相反：

| 模式 | 5× | 10× | 15× |
| --- | ---: | ---: | ---: |
| 仅压缩图像 | 51.2 | 40.3 | 31.1 |
| LensVLM + OCR | 45.8 | 38.7 | 32.1 |
| LensVLM + Zoom | 50.5 | 46.6 | 41.9 |

原因是原生文档里**图表、布局、空间关系**等很重要，OCR 变成纯文本会损失这些信息；高分辨率原图反而更有用。5× 时很多字已足够可读，因此额外工具不一定带来收益。

### 选页指标的限制

论文的 Selection Accuracy 定义为：**整个轨迹里至少有一次 Expand 命中某张 GT 证据页**。对于 MuSiQue 这种可能需要 2–4 张证据页的多跳问题，它只是 *any-hit*，并不是“全部证据页成功召回”。不能拿它直接代表多跳证据完整率。

## 5. 与 RAG、Visual Zoom Agent、Harness 的区别

- **RAG**：一般先构建检索索引，然后按 query 获取片段；LensVLM 先保留一份完整的压缩视觉概览，在**在线多轮上下文**中决定展开哪一页。并非说所有 RAG 都不能多轮/多模态；这是论文强调的基线流程差别。
- **普通 Zoom Agent**：对原本存在的高清图做放大以消除歧义；LensVLM 刻意把长文本先压成低分辨率页面，并把 Expand 作为高保真访问接口。
- **Harness**：从工程角度看，原始数据存储、页面映射、tool executor、多轮工具调用、上下文管理已可组成一个最简 Harness；但“把 `read_page` 包成 Harness”本身不是论文新贡献。

## 6. 对 Harness-OPD 的启发（**以下不是 LensVLM 原论文方法**）

LensVLM 的 **Student 在部署时仍然有 Expand 工具**。若改成“Teacher 在训练时可 Expand，Student 推理时无工具”，这就变成另一个研究问题，与 [OPHSD](OPHSD-reading-notes.md) 的 privileged-teacher 设置有关。

值得验证的是：

1. **定位能力 vs 事实读取能力**：低清页面可能仍支持定位，但不支持读出随机数字。蒸馏页面定位/推理过程可能有价值；直接蒸馏不可见事实可能只是记忆化。
2. **不能混淆上下文与 loss**：Teacher 私有原文和工具响应不直接算 Student response loss，但其存在会改变 Teacher 对 Student rollout 输出 token 的概率，所以信息不对称仍影响 KL 梯度。
3. **最小对照**：无工具 Visual Teacher、完整 Text Teacher、GT Evidence（Oracle）Teacher、真实 Select–Expand Harness Teacher；统一 Student 的压缩输入及训练预算。
4. **先验证是否有可迁移收益**：如果 Student 在压缩输入中压根看不到答案，加入更复杂的 Harness 也无法保证无工具问答提升。不要把多轮 Verify、OCR/Zoom routing 或增加中分辨率直接当成核心研究贡献。
5. **当前较聚焦的研究问题**：在固定 Select–Expand Harness 下，Teacher 看到的正确证据中，哪些监督能让无工具 Student 在 held-out 文档上泛化？

## 7. 实现与复现入口

- [官方仓库](https://github.com/apple-aiml-research/ml-lensvlm) 的 `lensvlm/rendering.py`：将源文本分成可索引页面并渲染；
- `lensvlm/rendering_config.py`：5×、10×、15× 压缩配置；
- `lensvlm/evaluate.py`：模型生成 → 解析调用 → 工具返回 → 再生成的多轮循环；
- `scripts/prepare_data.py`：构建文档、distractor、证据页标签 `gt_pages`；
- `eval/evaluate.py`：问答 judge 与页面选择评测；
- [官方 README 的 run_demo 示例](https://github.com/apple-aiml-research/ml-lensvlm#quick-start)：15 张图中选择 Page 10 并读取原文，最终回答 Chief of Protocol。

### 原文定位速查

| 问题 | 原文位置 |
| --- | --- |
| 主框架图 | Figure 2，PDF p.3 / §3 |
| 三种 Expand 返回形式、逐页调用 | §3.2 |
| ECR vs ICR | §3.1–3.2、Appendix A |
| Hard filtering、数据量与比例 | §3.3、Appendix G.1–G.2 |
| 397B 合成 Teacher 和标准答案 | §3.3、Eq. (3.4) |
| SFT mask、冻结视觉部分 | §3.4、Appendix D.2 |
| DAPO、reward、工具调用次数 | §3.4、Appendix C.7 / D.2 |
| 文本/OCR/Zoom 对照 | §5.2、Figure 5、Table 2 |
| 压缩信息不可恢复的限制 | Appendix E |
| 选页是否天然更容易？ | Appendix E、Table 10 |

---

**研究判断：** 这篇文章真正好的地方是把“全局概览”与“局部高保真读取”拆开，并通过 RL 教会小模型选择正确页面；它**并没有证明**训练期看到过原文的 Teacher 能把那些已经不可见的具体事实蒸馏给不使用工具的 Student。
