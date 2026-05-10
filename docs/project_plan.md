# 项目计划：Quasar 吸收谱后训练

## 1. 项目题目

面向已知谱线假设的 quasar 吸收谱 LLM 后训练：局域 mask、连续谱标定、transition velocity-frame 拟合与 agent 结构分析。

## 2. 背景与动机

Quasar 吸收谱里包含大量关于星际介质、星系际介质和晕气体的信息。实际分析时，研究者通常不是从整条光谱里完全盲猜谱线身份，而是先有一个候选系统红移、catalog redshift 或已知谱线假设，然后围绕目标谱线做局域检查：看局部谱窗、mask 掉污染区域、估计连续谱、对每条 transition 建立自己的 velocity frame、拟合该 frame 里的多个吸收峰，并判断拟合结果是否可信。

EGENT 这类工作说明，LLM 可以和传统光谱拟合工具结合起来：物理拟合仍由确定性工具完成，LLM 负责边界情况的视觉检查和决策。本项目仍在开发中，现阶段沿袭 EGENT 的第一部分思想：拟合阶段先由传统工具执行，遇到连续谱不稳、峰数歧义、残差异常、blend 或多解时，可以申请大模型介入辅助处理。除此之外，本项目还有第二部分：大模型基于拟合结果分析不同 transition 里的峰是否可能同源、谱线结构是否自洽、是否需要人工复核。

因此，本项目强调的是开发中的工作流：

- 数据构造
- 弱监督标注
- GPT-5.4 teacher 辅助标注
- SFT
- 规则奖励 RL
- 可程序化评测

而不是做一个大而全的多模态 agent。

底层物理工具不从零造轮子。线表、速度换算、Voigt/profile 和多成分拟合优先复用 `srt` 里的 `astro` / `ABSpec` 成熟实现，本项目重点放在 adapter、DESI 接入、审查包、agent 介入点和训练数据 schema。

## 3. 问题定义

输入包括：

- `line_id`：目标谱线身份，例如 `CIV_1548` 或 `CIV_doublet`
- `z_sys`：系统红移
- quasar 光谱：`wavelength / flux / ivar / pipeline_mask`

本项目只讨论 quasar 光谱中的吸收系统。即使后续扩展到 H I，也默认指 quasar 背景光中的 Ly alpha / DLA 吸收，而不是普通恒星或星系光谱里的谱线任务。

这里的“已知谱线假设”不是说目标吸收线必然存在。Quasar 吸收系统沿 sightline 不规则出现，同一个 quasar 光谱里可能有多个红移系统、blend、sky residual、噪声峰和错误 catalog 候选。因此 agent 必须判断局域吸收峰是否和候选吸收体假设自洽，而不是机械地把预期波长附近的任何凹陷都当成真实吸收。

这里有三个必须严格区分的概念：

- `transition`：一条谱线，例如 `CIV_1548` 或 `CIV_1550`。每条 transition 根据自己的静止波长和 `z_sys` 建立一个局部 velocity frame。
- `peak`：某个 transition velocity frame 里的数据吸收谷。一个 transition frame 里可以有多个 peak，它们对应不同速度。
- `absorber component`：物理吸收体成分。一个 component 可以在多条 transition 中产生对应峰；同一条 transition 里也可能出现多个 component 的峰。

`line_id` 和 `z_sys` 只负责切粗窗口、列出候选 transition family 和保存 provenance；它们不能在后续拟合中把峰中心钉死。第一层基础任务只做 per-transition-frame 的峰检测和拟合；跨 transition 的同源关系、吸收体环境和联合解释留给第二层 agent 分析。当前仍在开发阶段，不做联合分布。

对 doublet 或多 transition ion，不能把不同 transition 的观测波长简单压到同一个共享速度轴里当成一条线处理。每个 transition 都有自己的观测波长窗口、velocity frame、mask、残差和拟合贡献。

`configs/line_catalog.json` 还提供 `line_family_context` 这类软背景知识：ion 名称、doublet/multiplet 成员、oscillator strength 比例、预期强弱关系和 outlier policy。它只用于帮助 agent 理解图和组织证据，不是硬过滤器。doublet sibling 在相近 velocity 的一致结构是支持证据；单成员特征、强弱比异常或 sibling 不一致必须保留并报告，不能仅因“不符合 doublet”就自动删除、mask 或 reject。

LSF 也按这个原则处理。DESI 等数据源可以提供真实 resolution matrix / LSF；但在真实 LSF 未接入时，不猜测一个 Gaussian 或常数分辨率去改主拟合参数。当前实现只在 metadata 提供 per-transition LSF/resolution matrix 时输出并列诊断：同一组 intrinsic Voigt posterior 参数经过 LSF 后的观测空间模型和 residual。`instrument_lsf_applied_to_fit_likelihood` 当前为 `false`，表示主 likelihood、公开参数和 gate 指标仍来自 intrinsic Voigt。

本项目不要求 agent 独立完成所有科学裁决。遇到吸收速度大于 halo virial 速度、不同元素交叉验证不一致、多个吸收体解释竞争、或 catalog/拟合结果互相矛盾时，agent 的职责是完整整理候选结果和冲突证据，并显式输出 `human_review_required`。最终是否接受该系统、如何解释其物理来源，应留给人工/QC。

工具层根据：

```text
lambda_obs = lambda_rest * (1 + z_sys)
```

自动截取目标线附近的局域谱窗。第一部分 LLM/agent 读取局域谱窗、velocity-frame 拟合图、工具摘要和候选拟合结果，通过工具调用修正拟合：加/删源、调源参数、改 fit window、改 mask、改连续谱 anchors，并请求 refit。第二部分 LLM/agent 才负责结构化复核、同源关系和人工复核建议。

LLM 不负责：

- 从整条光谱中任意识别未知元素
- 绕过 deterministic fitter 直接给最终拟合参数
- 不经工具验证就直接给最终科学参数

LLM 负责：

- 第一部分：多模态拟合控制，直接看拟合图并调用工具修正连续谱、mask、源和窗口
- 第二部分：复核第一部分 per-transition-frame 候选拟合
- 判断不同 transition 中的峰是否可能来自同一个 absorber component，以及谱线结构是否自洽
- 汇总所有候选测量结果和冲突 flags，辅助人工最终裁决
- 输出合法、可验证的 JSON

## 4. MVP 范围

第一条可交付验证任务优先做 **C IV doublet absorption**，但工程接口不绑定 doublet，也不把 doublet 当成唯一拟合单位。

原因：

- C IV 在 quasar 吸收谱研究中常见。
- C IV 双线结构有明确物理约束，适合作为第一组 line-family 测试。
- 正样本、负样本和 hard cases 都比较容易程序化构造。
- `present / fit_ok / add_component` 等决策比开放式谱线识别更容易评测。
- C IV doublet 可以检查两条 transition frame 中的峰是否支持同一组 absorber components，适合作为“第一层拟合 + 第二层分析”的第一组测试。

未来规划：

- H I / DLA
- N V
- O VI
- 更复杂的多成分吸收系统
- 小图或局部图像辅助输入
- 基于同一天区、相近 sightline 的吸收体环境分析，判断是否存在更大的吸收体结构或群环境

## 5. 系统设计

系统分成两层：确定性工具层和后训练 LLM 层。

### 5.1 工具层

工具层负责：

- 读取光谱和 catalog
- 根据 `line_id` 和 `z_sys` 计算目标观测波长
- 自动截取局域谱窗
- 保留原始采样、gap 和 pipeline mask
- 根据 pipeline flag、gap、sky residual、telluric 区域生成 hard mask
- 根据 anchor points 拟合局部连续谱
- 为每条 transition 建立局部 velocity frame
- 在每个 transition frame 里检测和拟合多个吸收峰
- 对每条 transition 保留自己的窗口、mask 和拟合贡献
- 计算第一层拟合质量摘要，例如残差、RMS、reduced chi-square、峰中心、峰数

### 5.2 LLM 层

下面这层现在在代码里已经有一个有界多轮 agent/tool/refit loop。它仍是开发脚手架，长期策略、评测和 RL reward 还没定稿。

LLM/agent 负责：

- 第一层拟合控制：看 velocity-frame 图，改 continuum anchors/masks，增删源，调源参数，调窗口，请求 refit
- 第二层结构分析：吸收成分之间的同源关系判断、谱线结构和 blend 判断、人工复核建议
- 简短 rationale 生成

当前一轮 loop 不是单次 LLM 调用，而是：agent 先基于当前 record 和图做工具决策；工具层执行 refit 和 deterministic gate；随后 agent 在同一轮看到刚生成的 refit feedback、图和 gate 信号，写简短 assessment，并可通过 `request_more_budget` 申请进入下一轮。`--max-rounds` 是初始实验预算，`--hard-max-rounds` 是绝对上限。

模型输出必须符合固定 JSON schema。自然语言 rationale 只作为辅助检查，不作为主评测目标；核心指标都来自结构化字段。

## 6. Task A：局域 mask 与连续谱 anchor 预测

### 6.1 输入

Task A 的输入包括：

- `task = local_mask_continuum`
- `line_id`
- `z_sys`
- 局域观测波长窗口
- `wavelength / flux / ivar / pipeline_mask`
- 可选工具摘要，例如 SNR、gap 位置、hard mask 区间

局域窗口由工具自动切出，不需要人工手动切窗。

### 6.2 输出 schema

```json
{
  "task": "local_mask_continuum",
  "line_id": "CIV_1548",
  "window_action": "keep",
  "analysis_mask_intervals_A": [[5578.2, 5584.6], [5601.1, 5603.0]],
  "continuum_anchor_points_A": [5568.0, 5572.5, 5594.0, 5608.0],
  "quality": "medium",
  "rationale": "Mask contaminated pixels and absorption cores before continuum fitting."
}
```

### 6.3 标签来源

Task A 的标签不要求全人工标注，可以来自：

- pipeline mask 和坏点标记
- 低 ivar 或 ivar 为 0 的像素
- gap 和异常采样检测
- 已知 sky/telluric 污染区间
- 吸收核心自动检测，用来避免连续谱拟合被吸收谷拉低
- 高 flux、低残差的局部区域，用来构造 continuum anchors
- GPT-5.4 teacher 对 ambiguous 样本给出的候选 mask 和 anchor
- 少量人工/QC 修正

目标不是每个点都完美人工标注，而是构造规模可控、规则一致、可审计的训练信号。

### 6.4 Task A-control：第一层多模态拟合控制

这个任务是强化学习优先优化的对象。输入包括 review packet 的结构化字段和 `*.plot.png` velocity-frame 拟合图。模型不直接改数组，而是调用工具：

- `add_absorption_source`
- `remove_absorption_source`
- `update_absorption_source`
- `set_fit_mask_interval`
- `set_fit_window`
- `add_continuum_anchor`
- `remove_continuum_anchor`
- `update_continuum_mask`
- `request_refit`

连续谱 anchor 采用 ABSpec 的节点思路：节点由 `wavelength_A` 和可选 `continuum_flux` 组成，工具层根据节点重建 continuum 和归一化谱；agent 可以添加节点，也可以按节点序号删除偏在吸收谷里的节点。

工具执行后 deterministic fitter 重新生成 continuum、normalized flux、Voigt fit 和诊断图。RL 奖励评估的是这轮工具动作是否让拟合图、残差、连续谱和物理约束变好。

## 7. Task B：第一层拟合复核与第二层分析

### 7.1 输入

Task B 的输入包括：

- `task = fit_review`
- `line_id`
- `z_sys`
- 归一化后的局部光谱
- 各 transition 定义的局部 velocity frame，以及 frame 内检测到的峰
- 第一层拟合参数和残差统计
- 拟合质量摘要
- 各 transition 的局部贡献
- 可选 absorber hypothesis 特征，例如每条预期线的局部 depth、中心偏移和污染标记
- `line_family_context`：来自线表的软物理背景和 outlier policy
- 可选 `lsf_diagnostic`：同一组 intrinsic Voigt 参数经过真实 LSF/resolution matrix 后的并列 residual 诊断
- 可选环境约束，例如 halo virial velocity、impact parameter、host redshift、其他元素的候选测量

### 7.2 输出 schema

```json
{
  "task": "fit_review",
  "line_id": "CIV_doublet",
  "present": true,
  "system_plausible": true,
  "consistency_status": "plausible",
  "human_review_required": true,
  "review_flags": [
    "velocity_exceeds_virial_expectation",
    "cross_element_inconsistency"
  ],
  "matched_expected_lines": ["CIV_1548", "CIV_1550"],
  "missing_expected_lines": [],
  "fit_results": [
    {
      "fit_id": "fit_001",
      "line_family": "CIV",
      "z_abs": 2.6012,
      "center_shift_kms": -38.0,
      "n_components": 2,
      "equivalent_width_A": 0.42,
      "supported_by": ["CIV_1548", "CIV_1550"],
      "conflicts": []
    },
    {
      "fit_id": "fit_002",
      "line_family": "CIV",
      "z_abs": 2.5989,
      "center_shift_kms": -230.0,
      "n_components": 1,
      "equivalent_width_A": 0.21,
      "supported_by": ["CIV_1548"],
      "conflicts": ["missing_CIV_1550"]
    }
  ],
  "fit_ok": false,
  "next_action": "add_component",
  "preferred_n_components": 2,
  "final_center_shift_kms": -38.0,
  "quality": "medium",
  "final_confidence": 0.78,
  "rationale": "The residual is asymmetric near line center, suggesting an additional component."
}
```

`present` 只回答目标吸收信号是否存在；`system_plausible` 回答第二层分析后的系统解释是否自洽。第一层拟合只负责在每条 transition 定义的 velocity frame 里检测和拟合多个吸收峰；第二层分析再判断这些峰之间是否可能同源、谱线结构是否自洽、是否需要人工复核。若只出现单条疑似吸收、另一些 transition 落在坏点或 sky residual 区域，输出不应盲目接受，而应选择 `inspect`、`refit` 或 `reject`。

对于超过模型可靠边界的判断，`next_action` 应优先选择 `inspect`，并保留 `fit_results`。典型触发条件包括：

- 吸收速度超过 halo virial velocity 或环境约束给出的合理范围。
- C IV、Mg II、H I、N V、O VI 等元素之间的红移、速度结构或强度关系互相不支持。
- 一个凹陷可以被多个 redshift / line_id 解释。
- doublet 只有一条线明显，另一条落在 gap、bad pixels 或 sky residual 上。
- 拟合统计量看起来好，但物理解释不可信。

这类样本的训练目标不是让模型“猜最终答案”，而是让模型稳定输出完整候选、冲突来源、需要人工看的图/字段和保守建议。

### 7.3 动作空间

MVP 阶段只保留小动作空间：

- `accept`
- `refit`
- `add_component`
- `reject`
- `inspect`

动作空间要小一点，这样标签、奖励和评测都更可靠。

## 8. 数据集设计

### 8.1 数据来源

暂定数据来源：

- DESI quasar spectra
- 相关 quasar absorption 或 DLA catalog
- 可用的 sky/telluric 区间参考

数据源细节单独维护在 [数据源说明](data_sources.md)。关键边界是：训练和评测样本按 quasar object / sightline 组织，C IV 与 H I 都来自 quasar spectra；H I 路线优先使用 DESI DR1 DLA catalog。

我们不假设 DESI 已经提供完整任务标签。DESI 谱提供真实数据分布，任务标签由弱监督规则、候选拟合器、GPT-5.4 teacher、validator 和小规模人工/QC gold set 共同构造。

### 8.2 样本类型

数据集应包含：

- C IV 正样本窗口
- 明确无目标吸收的负样本窗口
- 低 SNR hard cases
- 有 gap 或不连续采样的窗口
- 受 sky residual 或 telluric 污染的窗口
- blend 或多成分吸收系统
- 候选拟合失败或拟合质量差的样本

### 8.3 样本规模与划分

目标规模：

- 约 1000 条真实局域窗口
- 每个窗口生成 1 到 2 条训练记录
- `train / valid / test = 70 / 15 / 15`

尽量按 quasar object 划分，避免同一个 object 的窗口同时出现在训练集和测试集里。

### 8.4 标签分层

数据标签分成四层：

- Raw windows：自动切出来的 DESI 局域谱窗，包含 `line_id`、`z_sys`、wavelength、flux、ivar、pipeline mask。
- Silver labels：规则、候选拟合、GPT-5.4 teacher 和物理 validator 共同生成的弱监督标签。
- Gold labels：约 100 到 150 条 hard 或代表性样本，由人工/QC 确认，用于最终评测和误差分析。
- Hard cases：低 SNR、gap、sky residual、blend、拟合失败或指标矛盾的样本，用来压力测试 SFT 和 RL。

### 8.5 弱监督标签策略

Task A 标签可以由以下信息生成：

- pipeline mask
- 低 ivar / 零 ivar 像素
- gap 和异常采样
- sky/telluric 污染区间
- 吸收核心检测
- 高 flux、低残差区域作为 continuum anchors
- GPT-5.4 teacher 对 ambiguous 样本的候选判断

Task B 标签可以从候选拟合构造：

- 如果残差、reduced chi-square、RMS、center shift 和 doublet consistency 都好，标 `accept`。
- 如果拟合大致可信但窗口或连续谱不稳定，标 `refit`。
- 如果目标附近残差不对称或有 W-shaped pattern，标 `add_component`。
- 如果目标不存在、污染严重或物理不一致，标 `reject`。
- 如果指标互相矛盾，自动规则不能可靠判断，标 `inspect`。
- 如果速度、环境或多元素证据冲突，标 `inspect`，并保存所有候选结果，不把冲突样本压成单一最终答案。
- 如果 doublet/multiplet 只有单成员支持或 sibling 关系异常，不直接删除该候选；保留为 outlier/ambiguous case，并由 agent/人工报告可能原因。

如果正负标签不够可靠，`present` 字段要谨慎处理。MVP 可以先把重点放在 `fit_ok`、`next_action`、`preferred_n_components`、mask 和 anchors 上；`present` 只在明确构造的随机负样本上评测。

### 8.6 GPT-5.4 Teacher 标注

GPT-5.4 用作固定 schema 下的 teacher，不作为绝对真值。

teacher 输出必须经过 validator 才能进入 SFT 或 RL 数据。

validator 要过滤掉：

- JSON schema 不合法
- mask 掉目标线核心，但没有选择 `reject` 或 `inspect`
- continuum anchor 落在深吸收谷里
- 与 pipeline hard mask 明显冲突
- reduced chi-square、RMS 或残差结构很差却标 `fit_ok=true`
- 谱线干净、对齐且物理一致时却标 `reject`

测试集不能只用 GPT-5.4 生成标签。最终报告的指标应主要依赖规则可验证标签和小规模 gold/QC set。

## 9. 训练计划

### 9.1 Base Model

使用 7B 或 8B 级开源 instruction model 作为学生模型。具体模型根据显存、LLaMA-Factory 兼容性、JSON 输出稳定性来选。

### 9.2 SFT

使用 LLaMA-Factory 做监督微调。

SFT 版本包括：

- 只用规则生成的 silver labels
- 使用规则 + GPT-5.4 teacher 过滤后的 silver labels
- 如果数据量允许，单独加入 hard cases 做增强

SFT 目标：

- 学会两个任务格式
- 输出合法 JSON
- 学会局域谱窗判读
- 在污染或模糊样本上保持保守
- rationale 不乱编、不和结构化字段矛盾

### 9.3 RL

使用 OpenRLHF 或 VeRL 做规则奖励 RL。

RL 阶段优先优化第一层多模态拟合控制 agent，而不是第二层解释总结。训练样本包含 review packet、velocity-frame 拟合图、可用工具列表和工具执行后的新拟合结果。

RL 阶段先从小规模开始，重点优化容易验证的能力：

- JSON 合法性
- 工具调用合法性
- 拟合控制动作选择正确性
- refit 后残差/RMS/reduced chi-square 改善
- 连续谱 anchor/mask 是否更合理
- 源数量是否更接近人工/QC 或 teacher 结果
- rationale 与结构化字段一致

MVP 不需要先训练 reward model。

RL 奖励可以加入 teacher agreement 项，但 teacher 信号必须低于物理规则和 validator：

```text
reward = rule_reward
       + alpha * teacher_agreement
       - beta * over_mask_penalty
       - gamma * invalid_json_penalty
       - delta * unnecessary_tool_penalty
```

teacher agreement 只比较结构化字段，不比较自由文本，例如：

- mask IoU
- anchor 位置接近程度
- 工具调用序列是否接近 teacher/QC
- source 增删是否合理
- source center / width / optical-depth 初值是否在容差内
- refit 后的拟合质量是否改善

## 10. Reward 设计

### 10.1 Task A Reward

候选 reward 组成：

| 组成项 | 权重 |
| --- | ---: |
| JSON 合法性 | 0.10 |
| analysis mask interval IoU / point-wise F1 | 0.30 |
| continuum anchor 对连续谱误差的改善 | 0.25 |
| window_action 正确性 | 0.10 |
| quality 一致性 | 0.10 |
| rationale 一致性 | 0.05 |
| hard cases 上的保守性 | 0.10 |
| ambiguous 样本上的 teacher agreement | 可选 |

### 10.2 Task A-control Reward

候选 reward 组成：

| 组成项 | 权重 |
| --- | ---: |
| 工具调用合法性 | 0.10 |
| refit 后残差/RMS 改善 | 0.25 |
| continuum anchor/mask 改善 | 0.20 |
| source 增删正确性 | 0.20 |
| source 参数初值接近度 | 0.10 |
| 不必要工具调用惩罚 | 0.05 |
| 多模态 hard case 上的保守性 | 0.10 |

### 10.3 Task B Reward

候选 reward 组成：

| 组成项 | 权重 |
| --- | ---: |
| JSON 合法性 | 0.10 |
| `present` 正确性 | 0.20 |
| `fit_ok` 正确性 | 0.15 |
| `next_action` 正确性 | 0.20 |
| `preferred_n_components` 正确性 | 0.10 |
| `final_center_shift_kms` 容差命中 | 0.15 |
| rationale 一致性 | 0.10 |
| ambiguous 样本上的 teacher agreement | 可选 |

权重后续可以调。第一版要保证奖励清晰、可解释、容易 debug。

## 11. 评测计划

### 11.1 主对比

至少比较：

- Rule baseline
- Base model
- SFT model
- 使用 GPT-5.4 teacher labels 的 SFT model
- SFT + RL model
- SFT + RL + teacher-agreement reward model

### 11.2 指标

Task A 指标：

- JSON 合法率
- mask interval IoU
- mask point-wise F1
- predicted anchors 带来的连续谱拟合误差改善
- `window_action` accuracy

Task B 指标：

- JSON 合法率
- `present` F1 / precision / recall
- `fit_ok` accuracy
- `next_action` accuracy
- `preferred_n_components` accuracy
- center shift MAE
- center shift 容差命中率

通用指标：

- hard-case 表现
- rationale 与结构化字段一致性
- 拒答或无效输出率
- Base -> SFT -> RL 的提升

### 11.3 Ablation

建议做这些消融：

- 纯规则工具 baseline
- 不给 fit-summary 的 LLM 输入版本
- 不给 `line_id` 的退化输入版本
- 不给 `z_sys` 的退化输入版本
- 不加入 hard cases 的 SFT
- 只用规则标签 SFT vs 规则 + GPT-5.4 teacher 标签 SFT
- 纯规则 reward RL vs 规则 + teacher-agreement reward RL

## 12. 与 EGENT 的关系

EGENT 是参考系统，不是本项目主体。

我们借鉴的原则：

- 物理拟合仍由确定性工具完成
- LLM 负责判断、复核和结构化决策
- 保留中间结果和 provenance
- 宁可保守，也不要幻觉式测量

关键区别：

- EGENT 研究恒星谱 EW 测量。
- 本项目研究 quasar 吸收谱。
- EGENT 重点是 agent workflow。
- 本项目重点是 post-training 和评测。
- EGENT 用视觉检查处理边界拟合。
- 本项目训练模型输出固定 schema 的局域决策。
- EGENT 当前没有显式 analysis-mask 工具。
- 本项目把 analysis mask 预测作为核心输出之一。

## 13. 交付物

最小交付：

- Task A 和 Task B 的 dataset schema
- 局域窗口自动截取工具
- 样本构造 pipeline
- LLaMA-Factory 兼容的 SFT 数据
- GPT-5.4 teacher 标注协议和 validator
- SFT 模型评测
- 规则 reward 评测代码
- 小规模 OpenRLHF 或 VeRL 实验
- Rule / Base / SFT / teacher-SFT / RL 对比表
- 最终报告：数据、训练、reward、评测、失败案例和局限性

可选交付：

- DLA 或 H I 扩展
- N V 或 O VI 扩展
- 更复杂的多成分拟合决策
- 小图辅助输入
- 交互式 demo

## 14. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 标签有噪声 | 程序化标签 + hard cases 小规模人工/QC |
| 任务太大 | MVP 只做 C IV doublet |
| RL 提升不明显 | 先确保 SFT 和离线 reward evaluation 扎实 |
| 模型不利用物理先验 | 做去掉 `line_id`、`z_sys`、fit summary 的消融 |
| JSON 不稳定 | 严格 schema、validator、JSON 合法性 reward |
| 数据太简单 | 主动加入低 SNR、gap、blend、拟合失败样本 |
| teacher 标签错 | 用 validator 过滤，并在独立 gold/QC set 上评测 |
| student 追不上 GPT-5.4 | 把 GPT-5.4 定位成 teacher / upper reference，不要求 student 绝对打败它 |

## 15. 实施路线

### Phase 1：规格与 schema

- 固定 Task A 和 Task B schema
- 定义 C IV 谱线表
- 定义局域窗口表示
- 定义评测指标

### Phase 2：数据构造

- 读取 DESI 光谱和 catalog
- 根据 `line_id` 和 `z_sys` 自动切局域窗口
- 生成 hard masks
- 生成初始弱标签
- 对 ambiguous / hard cases 调用 GPT-5.4 teacher
- 验证并过滤 teacher labels
- 构造小规模 gold/QC set
- 划分 train / valid / test

### Phase 3：工具 baseline

- 实现基于 anchors 的连续谱拟合
- 实现速度空间转换
- 实现候选拟合生成
- 实现规则 baseline

### Phase 4：SFT

- 导出 LLaMA-Factory 训练文件
- 用规则 silver labels 训练
- 用规则 + GPT-5.4 teacher silver labels 训练
- 在 validation 和 gold/QC set 上评测 SFT 版本

### Phase 5：RL

- 实现规则 rewards
- 实现可选 teacher-agreement reward
- 跑小规模 OpenRLHF 或 VeRL
- 比较 SFT、teacher-SFT、rule-RL、teacher-agreement RL

### Phase 6：报告与分析

- 生成指标表
- 分析 hard cases
- 总结失败模式
- 说明与 EGENT 的关系
- 写最终课程报告

## 16. 成功标准

项目成功的标志：

- 有可复现的数据构造 pipeline
- 模型能稳定输出符合 schema 的 JSON
- Base -> SFT 有可测提升
- SFT -> RL 至少在 JSON 合法性、动作选择或 hard-case 稳健性上有明确提升
- teacher-assisted labels 能改善 student 决策，同时不会绕过物理 validator
- `line_id`、`z_sys` 和速度空间摘要确实能提升模型判断
- 清楚地区分工具层物理拟合和 LLM 决策支持

## 17. 当前开发切片状态

当前代码里已经有的开发切片：

- `review/packet.py`：审查包编排入口
- `review/continuum.py`：连续谱与局域窗口处理
- `review/plot.py`：velocity-frame 图和平滑 Voigt 曲线
- `spectra/voigt_fit.py`：第一层 transition-frame peak fit
- `agent/llm.py`：provider-agnostic LLM client skeleton，包含第一层 `fit_control` 和第二层 `fit_review`
- `agent/fit_control.py`：归一化第一层工具调用，生成可审计 `fit_control_patch`，应用回拟合输入，并生成 deterministic refit gate
- `agent/loop.py`：有界 agent/tool/refit loop，负责跨轮状态、图像输入和候选结果选择
- `agent/policy.py`：fit-control 与图像/评分共享的小常量

还没形成稳定方案的东西：

- 可长期依赖的批量 refit 策略和质量门限
- review JSON -> LLM control/review JSON 的真实批处理评测
- 生产级 provider 配置
- SFT / RL 数据导出器
