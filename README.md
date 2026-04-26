# Quasar 吸收谱后训练项目

面向已知谱线假设的 quasar 吸收谱 LLM 后训练项目。

本项目不是直接复现 `Egent/`，也不是构建一个通用多模态光谱 agent。项目目标是借鉴 EGENT 的“传统光谱工具 + LLM 决策”路线，构造 quasar 吸收谱领域数据集，并通过 SFT 与规则奖励 RL 后训练，使开源 LLM 学会在已知 `line_id` 与系统红移 `z_sys` 的条件下完成局域谱窗判读、分析 mask 预测、连续谱锚点辅助标定，以及速度空间拟合结果复核。

当前训练路线采用 **teacher-assisted post-training**：真实 DESI 谱窗提供数据分布，规则工具、GPT-5.4 teacher 与少量人工/QC 共同构造标签；最终学生模型通过 SFT 与规则/teacher 混合奖励 RL 学习固定 schema 下的窄域结构化决策。

## 项目范围

### 核心假设

输入中已经给定目标谱线身份与系统红移：

- `line_id`
- `z_sys`
- quasar 光谱：`wavelength / flux / ivar / pipeline_mask`

系统根据

```text
lambda_obs = lambda_rest * (1 + z_sys)
```

自动切取目标谱线附近的局域谱窗。LLM 不负责从整条光谱中任意识别元素，也不直接替代物理拟合器，而是在工具层结果基础上做结构化分析决策。

重要约束：quasar 吸收线不是按固定规律必然出现。即使给定 `line_id` 和 `z_sys`，模型也必须判断局域谱窗中的吸收谷是否和候选吸收体自洽，例如 doublet 两条线是否同时支持、相对强度是否合理、是否可能只是噪声、sky residual、blend 或错误红移。

更高层的科学解释必须保留人工裁决。例如吸收速度超过 halo virial 速度、不同元素的交叉验证不一致、多个红移系统都能解释同一凹陷时，agent 不直接给最终物理结论，而是输出所有候选测量、冲突 flags、证据摘要和推荐检查项，让人根据完整结果判断。

### MVP 目标

最小可交付版本优先聚焦 **C IV doublet**，原因是其双线物理约束较明确，适合构造可验证标签与规则奖励。DLA / H I、N V、O VI 可作为后续扩展。

## 任务定义

### Task A：局域 mask 与连续谱 anchor

给定目标谱线的局域观测波长窗口，模型输出：

- 是否需要扩展或缩小窗口
- analysis mask 区间
- continuum anchor points
- 质量等级
- 简短 rationale

示例输出：

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

### Task B：速度空间拟合复核

工具根据 Task A 输出完成局域 continuum fitting，并把归一化谱转换到目标线速度空间。传统拟合器生成候选拟合结果后，模型复核拟合质量并选择下一步动作。

示例输出：

```json
{
  "task": "fit_review",
  "line_id": "CIV_doublet",
  "present": true,
  "absorber_reasonable": true,
  "consistency_status": "plausible",
  "human_review_required": true,
  "review_flags": ["cross_element_inconsistency"],
  "candidate_results": [
    {
      "candidate_id": "fit_001",
      "center_shift_kms": -38.0,
      "n_components": 2,
      "supported_by": ["CIV_1548", "CIV_1550"]
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

## 数据计划

数据来源暂定为 DESI quasar spectra 及相关 DLA / absorption catalog。详细数据源边界见 [docs/data_sources.md](/Users/mac/Desktop/BDMI/astro/docs/data_sources.md)。

本项目的 C IV 与 H I 路线都限定在 **quasar 光谱 / quasar sightline** 上。这里的 H I 默认指背景 quasar 光中的 Ly alpha / DLA 吸收系统，不混入普通恒星或星系光谱任务。

样本构造策略：

- 不假设 DESI 已经提供完整任务标签；真实谱窗负责提供真实数据分布
- 从已知 C IV / DLA catalog 抽取正样本窗口
- 从无目标吸收线区域抽取可验证负样本窗口
- 加入 hard cases：低 SNR、sky residual、telluric contamination、gap、不连续采样、pipeline bad pixels、拟合失败、多成分吸收
- 同一局域窗口可分别构造成 Task A 与 Task B 训练记录
- 同一窗口可生成不同质量候选拟合结果，用于训练 `accept / refit / add_component / reject / inspect`
- 使用规则标签、GPT-5.4 teacher 标签与 validator 过滤构造 silver labels
- 保留约 100 到 150 条 hard/gold 样本，由人工/QC 确认，用于最终评测和误差分析

目标规模：

- 约 1000 个真实局域窗口
- 每个窗口可生成 1 到 2 条训练记录
- 数据划分：`train / valid / test = 70 / 15 / 15`

## 训练计划

### SFT

使用 LLaMA-Factory 对 7B 或 8B 级开源模型进行监督微调。

训练目标：

- 遵守固定 JSON schema
- 学会局域观测波长空间中的 mask 与 continuum anchor 输出
- 学会读取速度空间拟合摘要并做结构化决策
- 学会判断候选吸收峰是否和吸收体假设自洽
- 遇到物理解释冲突时保留所有候选结果并触发人工审查
- 输出保守、可验证、低幻觉的领域判断

### RL

使用 OpenRLHF 或 VeRL 进行规则奖励 RL。优先采用可程序化验证的 reward，不先训练 reward model。RL 奖励可加入 teacher agreement 项，但 teacher 不能覆盖物理 validator。

Task A 奖励包括：

- JSON 合法性
- analysis mask 的 interval IoU 或 point-wise F1
- anchor points 对 continuum 拟合误差的改善
- window_action 正确性
- rationale 与结构化输出一致性

Task B 奖励包括：

- JSON 合法性
- `present` 判断正确
- `fit_ok` 判断正确
- `next_action` 判断正确
- `preferred_n_components` 正确
- `final_center_shift_kms` 容差命中
- rationale 与结构化结论一致性

## 评测计划

必须比较：

- Rule baseline
- Base model
- SFT model
- SFT with GPT-5.4 teacher silver labels
- SFT + RL model
- SFT + RL with teacher-agreement reward

主指标：

- JSON 合法率
- mask interval IoU / point-wise F1
- continuum anchor 带来的拟合误差改善
- `present` F1 / precision / recall
- `fit_ok` accuracy
- `next_action` accuracy
- `preferred_n_components` accuracy
- center shift MAE 或容差命中率

对比实验：

- 规则方法 baseline
- 不带拟合结果摘要的 LLM 输入版本
- 不带 `line_id` 的退化输入版本
- 不带 `z_sys` 的退化输入版本

这些对比用于验证 post-training 是否真正提升了局域谱线分析决策能力，以及模型是否利用了谱线先验与速度空间信息。

## 与 EGENT 的关系

`Egent/` 是参考文献 EGENT 的代码基线，不是本项目主体。

本项目借鉴 EGENT 的部分思想：

- 使用传统工具执行物理拟合
- LLM 负责局域判读与拟合决策
- 保留完整中间结果与可解释 provenance

关键区别：

- EGENT 面向恒星谱 equivalent width measurement
- 本项目面向 quasar absorption spectra
- EGENT 重点是 agent workflow
- 本项目重点是 post-training：数据集构造、SFT、RL 与可验证评测
- EGENT 当前没有显式 analysis mask 工具；本项目把 LLM 输出 mask 作为核心能力之一

## 仓库结构

```text
.
├── BDMI 大作业.pdf          # 原始课程项目设想
├── Egent/                  # EGENT 参考代码
├── configs/                # 配置文件占位，后续放谱线表与训练配置
├── data/                   # 数据目录占位，不提交真实大数据
├── docs/
│   ├── egent_reference.md  # EGENT 架构与最小复现说明
│   └── project_plan.md     # 正式项目计划
├── outputs/                # 实验输出占位，不提交生成结果
├── scripts/                # 后续放数据构造与训练辅助脚本
├── src/astroagent/         # 本项目 Python 包占位
├── tests/                  # 后续放单元测试
├── pyproject.toml          # Python 项目基础配置
└── README.md               # 本项目入口文档
```

## 当前状态

- 已明确项目边界：不做通用整谱 agent，不直接复现 EGENT
- 已确定 MVP：优先 C IV doublet
- 已拆分两个训练任务：Task A 与 Task B
- 已确定训练路线：Rule labels / GPT-5.4 teacher -> SFT -> RL
- 已确定评测路线：结构化指标 + ablation
- 已加入最小人工审查包实现：谱线表、局域切窗、rule baseline、JSON/CSV 输出与单元测试
- 已实测 EGENT 参考代码：direct fitting 正常，CherryIN `openai/gpt-5.4` 可跑通 LLM 视觉复核

最小实现说明见 [docs/minimal_implementation.md](/Users/mac/Desktop/BDMI/astro/docs/minimal_implementation.md)。可以先运行：

```bash
python3 scripts/make_review_packet.py
```

下一步应接入 DESI 小批量 catalog，验证能否从 `TARGETID` 找回 quasar 光谱并生成真实 review packets。
