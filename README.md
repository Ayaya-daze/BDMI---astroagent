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
3. [src/astroagent/review_packet.py](src/astroagent/review_packet.py)：审查包编排入口。
4. [src/astroagent/review_continuum.py](src/astroagent/review_continuum.py)：连续谱与局域窗口处理。
5. [src/astroagent/review_plot.py](src/astroagent/review_plot.py)：拟合模型和诊断图。
6. [src/astroagent/llm_interface.py](src/astroagent/llm_interface.py)：LLM 接口骨架。
7. [tests/test_review_packet.py](tests/test_review_packet.py)：当前行为边界。

## 当前阶段

当前仍处于开发和重构阶段，不是定稿版本。现在可运行的是一个最小审查包切片，拟合链路正在按“第一层拟合、第二层分析”拆分。

目前可用的开发切片有：

- 谱线常量表：`configs/line_catalog.json`
- 局域观测波长切窗
- 速度坐标转换
- 窗口质量摘要
- line-family 元数据和审查字段；C IV / Mg II 只是第一批验证样例，不代表项目只做 doublet
- 第一层拟合草案：每条 transition 自己建 velocity frame，拟合多个吸收峰
- 第二层分析接口骨架：后续让 agent 判断同源关系、blend、结构和人工复核需求
- Task A 的 rule baseline：analysis mask 和 continuum anchors
- 人工裁决字段：保留最终科学判断给人
- JSON/CSV 审查包输出
- ABSpec 风格诊断图：每条 transition 一个速度空间子图，观测谱用 step，模型用平滑 Voigt component/combined 曲线
- LLM 接口骨架：支持离线 client 和 OpenAI-compatible chat completions；完整多轮 agent loop 仍在开发中

下一步主线还是接入 DESI 小批量真实数据，从 `TARGETID` 或 catalog 记录生成可审查的 review packets；实验输出不作为长期资产保留。

## 运行

从仓库根目录运行 demo：

```bash
.venv/bin/python scripts/make_review_packet.py
```

默认会生成一个合成样本用于本地检查。生成物只用于调试，不作为长期结果保留。

使用自己的 CSV：

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

运行第一层 LLM 拟合控制的最小接口：

```bash
.venv/bin/python scripts/run_fit_review.py \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --client offline \
  --mode fit_control \
  --plot-image outputs/review_packet/demo_CIV_doublet_z2p6000.plot.png
```

`offline` client 不联网，只用于验证 schema 和文件链路。真实 provider 使用 `--client openai-compatible`，并通过 `ASTROAGENT_LLM_API_KEY`、`ASTROAGENT_LLM_MODEL`、`ASTROAGENT_LLM_BASE_URL` 配置。
`fit_control` 模式会同时输出 `*.fit_control_patch.json`，供后续 refit loop 和 RL 使用。

## 目录分层

```text
.
├── src/astroagent/          # 本项目主线代码
├── scripts/                 # 本地运行入口
├── configs/                 # 谱线表和后续配置
├── tests/                   # 最小行为测试
├── docs/                    # 项目地图、计划、数据源和参考笔记
├── data/                    # 数据占位，不提交大数据
├── outputs/                 # 生成产物，不作为主线阅读入口
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
