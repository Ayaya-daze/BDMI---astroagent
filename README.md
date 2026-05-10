# Quasar 吸收谱后训练项目

这个仓库还在开发中。主线很短：给定 `line_id + z_sys` 和一段 quasar 光谱，工具层切出局域谱窗，先做连续谱和峰锚定，再在每条 `transition` 自己的局部 velocity frame 里拟合多个吸收峰，最后把结果交给 LLM/agent 做结构分析和人工审查。

详细项目地图见 [docs/repo_map.md](docs/repo_map.md)。完整研究计划见 [docs/project_plan.md](docs/project_plan.md)。

## 项目定位

本项目不是重新写一套光谱物理工具。底层线表、速度换算、Voigt/profile 和多成分拟合优先复用 `srt` 里的 `astro` / `ABSpec` 成熟代码，本仓库只写薄 adapter、DESI 数据接入、审查包、LLM 接口和训练数据 schema。

创新点在两层：

- 第一层沿袭 EGENT 思路：传统工具先拟合；遇到连续谱不稳、成分数歧义、残差异常、blend 或多解时，可以申请大模型介入辅助拟合处理。
- 第二层是本项目自己的大模型分析阶段：基于第一层输出分析吸收成分是否同源、谱线结构是否自洽、是否有 blend/污染、是否需要人工复核。

## 先看什么

1. [docs/repo_map.md](docs/repo_map.md)：仓库导航，说明哪些文件是主线、哪些只是参考或产物。
2. [docs/minimal_implementation.md](docs/minimal_implementation.md)：当前可运行的最小实现。
3. [src/astroagent/review/packet.py](src/astroagent/review/packet.py)：审查包编排入口。
4. [src/astroagent/review/continuum.py](src/astroagent/review/continuum.py)：连续谱与局域窗口处理。
5. [src/astroagent/review/plot.py](src/astroagent/review/plot.py)：拟合模型和诊断图。
6. [src/astroagent/agent/llm.py](src/astroagent/agent/llm.py)：LLM 接口骨架。
7. [tests/test_review_packet.py](tests/test_review_packet.py)：当前行为边界。

## 当前阶段

当前仍处于开发和重构阶段，不是定稿版本。现在可运行的是一个最小审查包切片，拟合链路正在按“第一层拟合、第二层分析”拆分，并且已经接入有界的 agent/tool/refit loop。

目前可用的开发切片有：

- 谱线常量表：`configs/line_catalog.json`
- 局域观测波长切窗
- 速度坐标转换
- 窗口质量摘要
- line-family 元数据和审查字段；C IV / Mg II 只是第一批验证样例，不代表项目只做 doublet
- 第一层拟合草案：每条 transition 自己建 velocity frame，拟合多个吸收峰
- 第二层分析接口骨架：后续让 agent 判断同源关系、blend、结构和人工复核需求
- Task A 的 rule baseline：analysis mask 和 continuum anchors
- 有界 agent/tool/refit loop：每轮复用同一套 patch schema，输出下一轮是否继续、是否转人工审查
- 诊断图同时保留观测总览和残差视图，便于 agent 看到全局和局部退化
- 人工裁决字段：保留最终科学判断给人
- JSON/CSV 审查包输出
- ABSpec 风格诊断图：每条 transition 一个速度空间子图，观测谱用 step，模型用平滑 Voigt component/combined 曲线
- LLM 接口骨架：支持离线 client 和 OpenAI-compatible chat completions；有界多轮 agent loop 已接入，策略和评测仍在迭代

最近已经用真实脏数据跑过一批小实验，包括本地 `srt/data` 里的 COS/HI 样本。结论是：数据处理层和 loop 脚手架已经能工作，但具体样本是否继续探索、是否转人工，仍然依赖 gate 和后续模型判断。

下一步主线还是接入 DESI 小批量真实数据，从 `TARGETID` 或 catalog 记录生成可审查的 review packets；实验输出不作为长期资产保留。

## 运行

从仓库根目录运行 demo：

```bash
.venv/bin/python -m astroagent.cli.main packet
```

默认会生成一个合成样本用于本地检查。生成物只用于调试，不作为长期结果保留。

使用自己的 CSV：

```bash
.venv/bin/python -m astroagent.cli.main packet \
  --input-csv data/interim/example_spectrum.csv \
  --line-id CIV_doublet \
  --z-sys 2.6
```

输入 CSV 至少需要四列：

- `wavelength`
- `flux`
- `ivar`
- `pipeline_mask`

运行第一层 LLM 拟合控制的最小接口：

```bash
.venv/bin/python -m astroagent.cli.main llm \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --client offline \
  --mode fit_control
```

`offline` client 不联网，只用于验证 schema 和文件链路。真实 provider 使用 `--client openai-compatible`，并通过 `ASTROAGENT_LLM_API_KEY`、`ASTROAGENT_LLM_MODEL`、`ASTROAGENT_LLM_BASE_URL` 或 `ASTROAGENT_LLM_API_URL` 配置。
`fit_control` 模式会同时输出 `*.fit_control_patch.json`，供后续 refit loop 和 RL 使用。
如果同目录存在同名 `*.plot.png`，入口会自动作为图像输入；也可以用 `--plot-image` 显式指定。

例如使用 Paratera GLM-4V 兼容接口时，可以在 shell 里临时设置：

```bash
export ASTROAGENT_LLM_API_KEY='...'
export ASTROAGENT_LLM_MODEL='GLM-4V-Plus-0111'
export ASTROAGENT_LLM_BASE_URL='https://llmapi.paratera.com'
```

运行有界 agent/tool/refit loop：

```bash
.venv/bin/python -m astroagent.cli.main fit-loop \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --client offline \
  --max-rounds 2 \
  --hard-max-rounds 6
```

loop 每轮复用同一套 `fit_control` tool schema、patch 记录和 deterministic refit gate。`offline` client 默认不发工具调用，只验证编排；真实多模态 provider 才会提出 add/update/remove source、mask/window/continuum edits，并触发下一轮 refit。
如果同目录存在同名 `*.window.csv`、`*.overview.png` 和 `*.plot.png`，loop 入口会自动使用它们；也可以用参数显式覆盖。

loop 不是单次 LLM 调用。每个实验轮包含两段 agent 交互：先决策 source/window/mask/continuum edits，工具层执行 refit 和 gate；随后 agent 在同一轮看到刚生成的 refit 结果和图，写出简要 assessment，并决定是否通过 `request_more_budget` 申请进入下一轮。gate 只负责防止明显退化和记录人工复核状态，不负责最终科学裁决。loop 结束后还会写出 `<sample_id>.audit/audit_report.md` 和 `audit_report.json`，方便人工快速审查本轮工具动作、gate 结果和后续检查项。

## 目录分层

```text
.
├── src/astroagent/          # 本项目主线代码
├── scripts/                 # 本地运行入口
├── configs/                 # 谱线表和后续配置
├── tests/                   # 最小行为测试
├── docs/                    # 项目地图、计划、数据源和参考笔记
├── data/                    # 数据占位，不提交大数据
├── outputs/                 # 临时生成产物，只保留 .gitkeep
├── Egent/                   # 外部参考代码，不是本项目主体
└── BDMI 大作业.pdf          # 原始课程设想
```

## 重要边界

- `Egent/` 是参考实现，不是要直接维护的主线。
- `srt` 里的 `astro` / `ABSpec` 是优先复用的成熟工具来源，主线代码只做 adapter 和流程编排。
- 当前第一条验证任务用 C IV / Mg II 双线，但核心接口按通用 line-family / component / system 写。
- LLM 不做整谱盲识别，也不替代物理拟合器。
- LLM 接口现在只是一层薄 client，不绑定某一家 provider。
- 更高层科学解释必须保留人工裁决。
- 当前代码先保证中间产物可审查，再扩展到 DESI、teacher labels、SFT 和 RL。
- `outputs/` 里的实验产物默认可清理，不作为长期结果或数据集版本管理位置。
- `reports/fit_control_real_model_report/` 是本地技术报告工作区，包含 PDF、图片和真实模型验证摘要副本；默认 ignore，不作为代码仓库资产提交。
