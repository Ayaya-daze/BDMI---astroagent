# 仓库地图

这份文档只解决一个问题：打开仓库时应该先看哪里、哪些文件暂时不用管。

## 一句话主线

本项目要做的是 quasar 吸收谱后训练数据与工具链。当前最小实现先生成人能检查的 review packet；后续拟合链路按两层拆分：第一层在每条 transition 的 velocity frame 里用 physical Voigt posterior 拟合多个吸收峰，第二层由 agent 分析同源关系、blend、结构和人工复核需求。

## 阅读顺序

1. `README.md`：入口页，只保留当前状态、运行方式和目录分层。
2. `docs/minimal_implementation.md`：解释当前最小实现为什么这么切。
3. `src/astroagent/review/packet.py`：审查包编排入口。
4. `src/astroagent/review/continuum.py` / `src/astroagent/review/plot.py` / `src/astroagent/spectra/voigt_fit.py`：连续谱、画图、第一层拟合。
5. `src/astroagent/agent/llm.py`：两层 LLM 接口骨架；第一层 `fit_control` 能看图并输出拟合工具调用，第二层 `fit_review` 做结构复核。
6. `docs/llm_interface.md`：LLM 接口用法和边界。
7. `tests/test_review_packet.py`：当前行为边界。
8. `docs/project_plan.md`：研究路线，明确区分当前开发切片和未来规划。

## 主线文件

这些文件是现在要维护的主体：

```text
configs/line_catalog.json
scripts/make_review_packet.py
src/astroagent/spectra/line_catalog.py
src/astroagent/review/continuum.py
src/astroagent/review/packet.py
src/astroagent/review/plot.py
src/astroagent/spectra/voigt_fit.py
src/astroagent/agent/llm.py
src/astroagent/agent/fit_control.py
src/astroagent/agent/loop.py
src/astroagent/agent/policy.py
src/astroagent/cli/make_review_packet.py
src/astroagent/cli/run_fit_review.py
src/astroagent/cli/apply_fit_control_patch.py
src/astroagent/cli/run_fit_control_loop.py
tests/test_review_packet.py
tests/test_llm_interface.py
tests/test_fit_control_loop.py
docs/minimal_implementation.md
docs/llm_interface.md
docs/project_plan.md
```

## 外部参考

`Egent/` 和用户目录里的 `srt/code` 是参考来源，不是本项目主代码。保留的原则只有两条：

- EGENT 式大模型介入拟合控制，而不是让 LLM 直接给最终科学结论。
- `srt` 里的 `astro` / `ABSpec` 成熟工具优先作为薄 adapter 复用，避免重复造线表、速度换算和 Voigt/profile 基础工具。

## 数据和产物

这些目录是输入或输出，不是代码主线：

```text
data/raw/
data/interim/
data/processed/
outputs/
```

`outputs/review_packet/` 里的 JSON/CSV 是脚本生成的审查样例。它们可以用来检查格式，但不要把它们当成源代码。

## 当前可运行切片

运行：

```bash
.venv/bin/python scripts/make_review_packet.py
```

当前流程：

```text
line catalog
  -> demo or CSV spectrum
  -> local wavelength window
  -> window summary
  -> absorber hypothesis check
  -> continuum / normalized spectrum
  -> transition-frame physical Voigt posterior summary
  -> Task A rule suggestion
  -> human review fields
  -> JSON/CSV review packet
```

## 下一步主线

建议按这个顺序继续：

1. 用小批量 DESI 样本持续检查 review packet 和图像输出。
2. 让第一层 agent 继续通过图像修正 source、window、mask 和 continuum anchors。
3. 用 `astroagent-run-fit-review --mode fit_control` 读一个 review JSON 和 `*.plot.png`，输出 `*.llm_control.json` 与 `*.fit_control_patch.json`。
4. 让 `fit_control_patch + window.csv -> refit` 持续稳定，确保 refit 质量 gate 可靠。
5. 增加第二层 agent 分析：判断同源关系、谱线结构、blend 和人工复核需求。
6. 把人工/agent 修正后的 review packets 导出为 Task A-control / Task B 训练样本。
7. 再进入 SFT、RL 和评测；RL 优先优化第一层多模态拟合控制。
8. 基本任务稳定后，再扩展到同一天区、相近 sightline 的吸收体环境分析，判断是否存在更大的吸收体结构或群环境。

## 暂时不要优先做

- 不要先重构成大型 pipeline。
- 不要把 doublet 的不同 transition 压进同一个共享速度轴里直接拟合。
- 不要在本仓库里重复实现已经稳定的线表、速度换算和 Voigt/profile 基础工具。
- 不要先训练模型。
- 不要把 `Egent/` 当成本项目主代码改。
- 不要让 LLM 直接给最终科学结论。
