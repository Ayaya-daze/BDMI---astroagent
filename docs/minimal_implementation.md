# 最小实现说明：人工审查包

这个文档描述的是开发中的最小切片，不是完成版。目标不是直接训练模型，而是先搭出一个可以审查、可以运行、可以逐步扩展的项目骨架。

它回答三个问题：

- 已知 `line_id` 和 `z_sys` 时，代码能不能切出局域 quasar 吸收谱窗？
- 局域谱窗里能不能先为每条 transition 建立自己的 velocity frame，再在这个空间里拟合多个吸收峰？
- 工具层能不能给出一个非常朴素的 rule baseline？
- 结果能不能保存成人类能检查和修改的文件？

## 文件职责

- `configs/line_catalog.json`：谱线常量表。现在包含 C IV、H I Ly alpha、Mg II。
- `src/astroagent/spectra/line_catalog.py`：读取谱线表，负责 `line_id -> transition definitions`。
- `src/astroagent/review/packet.py`：审查包编排入口，负责切窗、摘要、fit summary 组装、rule baseline、写出审查包。
- `src/astroagent/review/continuum.py`：连续谱、归一化谱、局域速度坐标和吸收核心 exclusion。
- `src/astroagent/review/plot.py`：平滑 Voigt 曲线、per-transition velocity-frame 诊断图。
- `src/astroagent/spectra/voigt_fit.py`：第一层 per-transition-frame peak fit。
- `src/astroagent/agent/llm.py`：两层 LLM 接口骨架；第一层 `fit_control` 看图并输出拟合工具调用，第二层 `fit_review` 做结构复核。
- `src/astroagent/agent/fit_control.py`：把第一层 LLM 工具调用归一化成可审计 patch，应用到 fitter overrides，并执行 refit gate。
- `src/astroagent/agent/loop.py`：有界 agent/tool/refit loop，只负责编排每轮 LLM 决策、refit 和状态传递。
- `src/astroagent/agent/policy.py`：fit-control 与图像/评分共享的小常量。
- `src/astroagent/cli/make_review_packet.py`：命令行入口，只负责参数解析和调用核心函数。
- `scripts/make_review_packet.py`：本地开发用包装脚本，不要求先安装包。
- `tests/test_review_packet.py`：最小单元测试，保证核心流程能跑通。

## 运行方式

从仓库根目录运行：

```bash
.venv/bin/python scripts/make_review_packet.py
```

默认会生成一个合成的 C IV doublet quasar 吸收谱窗口。输出只用于开发检查，不作为长期结果保留。

```text
outputs/review_packet/
```

运行脚本时通常会生成这些开发检查文件：

- `*.review.json`：结构化审查记录。
- `*.window.csv`：局域谱窗数组，包含 `wavelength / flux / ivar / pipeline_mask / velocity_kms`。
- `*.plot.csv`：画图和拟合专用数组，包含连续谱、归一化谱和第一层拟合列。
- `*.model.csv`：速度空间平滑 Voigt 曲线，`component` 是单峰曲线，`combined` 是同一 transition 内所有峰的合成曲线。
- `*.plot.png`：ABSpec 风格的拟合诊断图；每条 transition 一个局部速度坐标子图，不输出波长空间总览图。
- `*.llm_control.json`：运行 LLM control 入口后得到的第一层多模态拟合控制输出。
- `*.fit_control_patch.json`：运行 LLM control 入口后得到的归一化 patch，可用于下一轮 refit。
- `README.md`：输出文件说明。

也可以传入自己的 CSV：

```bash
.venv/bin/python scripts/make_review_packet.py \
  --input-csv data/interim/example_spectrum.csv \
  --line-id CIV_doublet \
  --z-sys 2.6
```

输入 CSV 至少需要四列：

- `wavelength`
- `flux`
- `ivar`
- `pipeline_mask`

## 人工审查什么

先打开生成的 `*.review.json`，重点看：

- `input`：谱线假设、系统红移、观测中心波长、窗口范围。
- `window_summary`：像素数、坏像素比例、flux 分位数、粗略 SNR。
- `absorber_hypothesis_check`：粗窗口里是否有数据驱动的吸收谷，只负责告诉你有没有值得进入第一层拟合的结构。
- `human_adjudication`：明确哪些科学裁决必须由人完成，例如 virial velocity 冲突或多元素不一致。
- `fit_results`：保留第一层拟合结果，后续 agent 再分析同源关系、blend 和多成分结构。
- `task_a_rule_suggestion`：规则方法建议的 mask 和 continuum anchors。
- `human_review`：留给人工填写接受、拒绝或修正。

这个格式故意很朴素。后续接入 DESI 真数据、teacher labels、SFT 数据导出时，仍需要继续审计和调整；不要把当前输出当作科学可用结果。

## 第一层 refit loop

最小 refit loop 已经有开发版链路：

1. `scripts/run_fit_review.py --mode fit_control` 生成 `*.fit_control_patch.json`。
2. `scripts/apply_fit_control_patch.py` 读取原始 `*.review.json`、`*.window.csv` 和 patch。
3. 工具层根据 patch 更新 continuum anchors/masks、fit windows、fit masks 和 source seeds。
4. 确定性 fitter 重新生成 `*_refit.review.json`、`*_refit.plot.csv`、`*_refit.model.csv` 和 `*_refit.plot.png`。
5. `fit_control_evaluation` 比较原始 fit 和 refit：失败、RMS 明显变差、组件数暴涨、fit pixel 被 mask 掉太多或 quality 退化时，patch 用内部兼容字段标记为 `rejected`，含义是不进入主状态、效果可能变差；仍有 high residual / inspect 信号时标记为 `needs_human_review`；只有无新增警告的改善才自动 `accepted`。
6. `scripts/run_fit_control_loop.py` 在同一套 patch/refit/gate 上做多轮编排。每轮会继承上一轮已保留的 overrides，LLM 可以一次提出多个 source、window、mask 或 continuum edits。

这还不是定稿拟合方案。它只用于验证第一层大模型能通过工具调用参与拟合输入修正，并为后续 RL 记录 action -> refit result。

## LLM 接口边界

当前开发版的 LLM/agent 层由 `src/astroagent/agent/llm.py`、`src/astroagent/agent/fit_control.py` 和 `src/astroagent/agent/loop.py` 组成。它们负责：

- 把 review packet 压缩成 `fit_control` prompt messages，并可附带 `*.plot.png`。
- 暴露第一层拟合控制工具：增删/更新 source，改 fit window，改吸收 mask，改 continuum anchors/masks，请求 refit。
- 把第一层工具调用保存成 `fit_control_patch`。
- 把 patch 变成确定性的 fitter overrides，执行 refit，并记录 deterministic gate 决策。
- 多轮 loop 只传递状态和图像，不重新实现工具 schema 或 refit 逻辑。
- 把 review packet 压缩成 `fit_review` prompt messages，用于第二层结构复核。
- 定义 provider-agnostic 的 `LLMClient` 协议。
- 提供离线 client 和 OpenAI-compatible client。

它不做三件事：

- 不从整条光谱盲识别谱线。
- 不绕过第一层物理拟合器直接给最终拟合参数；第一层只能通过工具调用修改拟合输入并请求 refit。
- 不给最终科学结论；第二层只输出结构化复核、冲突证据和人工复核建议。

## 为什么这样写

项目开发不要一开始写成巨大 pipeline。比较稳的顺序是：

1. 先有一个小输入。
2. 用纯函数做一个确定转换。
3. 写出人能检查的中间产物。
4. 加一个测试保证行为不乱变。
5. 再接真实数据和模型。

这就是这个最小实现的边界。
