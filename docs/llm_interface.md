# LLM 接口说明

这份文档描述当前开发切片，不代表第一部分拟合 agent 已完成。现在的重点是把单轮工具调用、patch 记录和 refit gate 先固定下来，后续再扩展成多轮 agent loop。

当前 LLM 接口分两层：

- 第一层 `fit_control`：多模态拟合控制 agent，能看 `*.plot.png`，可以加/删/调 source，改 fit window，改 continuum anchors，改 continuum mask，改 absorption mask，然后请求 refit。
- 第二层 `fit_review`：对第一层输出做结构化复核和解释。

## 当前开发切片

- `src/astroagent/llm_interface.py`
  - `build_fit_control_messages(record, plot_image_path=None)`：把 review packet 压缩成 fit-control messages，可附带图像。
  - `run_fit_control(record, client, plot_image_path=None)`：调用 client，解析 tool calls / JSON，并做最小 schema 检查。
  - `build_fit_review_messages(record, plot_image_path=None)`：把 review packet 压缩成复核 messages。
  - `run_fit_review(record, client, plot_image_path=None)`：调用 client，解析 JSON，并做最小 schema 检查。
  - `OfflineReviewClient`：不联网，固定输出 `inspect`，用于测试链路。
  - `OpenAICompatibleClient`：调用 OpenAI-compatible `/chat/completions` provider。
- `src/astroagent/cli/run_fit_review.py`
  - 读取 `*.review.json`。
  - 可附带 `*.plot.png`。
  - 输出 `*.llm_review.json` 或 `*.llm_control.json`。
  - `fit_control` 模式额外输出 `*.fit_control_patch.json`，用于后续 refit loop 和 RL reward。
- `src/astroagent/fit_control_apply.py`
  - 把 `fit_control_patch` 转成 deterministic fitter overrides。
  - 支持连续谱 anchors/masks、transition fit windows、fit masks、source seed 增删/更新。
  - source seed 现在可以显式携带 `center_prior_sigma_kms`、`center_prior_half_width_kms`、`logN`、`b_kms` 和上下界。
  - 重新生成 review JSON、plot CSV、model CSV 和 velocity-frame plot PNG。
  - 写入 `fit_control_evaluation`：比较原始拟合和 refit 后拟合，给出 `accepted`、`needs_human_review` 或 `rejected`。LLM 工具调用不会绕过这个 gate。
- `src/astroagent/cli/apply_fit_control_patch.py`
  - 读取原始 `*.review.json`、`*.window.csv` 和 `*.fit_control_patch.json`。
  - 输出 `*_refit.review.json`、`*_refit.plot.csv`、`*_refit.model.csv`、`*_refit.plot.png`。
  - stdout 会打印 `fit-control decision`，便于批处理筛选。
- `scripts/run_fit_review.py`
  - 本地开发包装入口，不要求先安装包。

## 当前边界

- 不做整谱盲识别。
- 不把 LLM 输出当最终科学结论。
- 不直接替代 deterministic fitter，但会参与其拟合循环。
- 这个阶段还不做完整多轮 agent 编排；当前先支持单轮工具调用、patch 记录和一次 refit。

## 环境变量

真实 provider 走 `openai-compatible` client：

```bash
export ASTROAGENT_LLM_API_KEY=...
export ASTROAGENT_LLM_MODEL=...
export ASTROAGENT_LLM_BASE_URL=https://api.openai.com/v1
```

`ASTROAGENT_LLM_BASE_URL` 可选；如果使用兼容接口的本地或代理服务，改这个变量即可。

## 本地验证

先生成 review packet：

```bash
.venv/bin/python scripts/make_review_packet.py
```

再跑离线 LLM control：

```bash
.venv/bin/python scripts/run_fit_review.py \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --client offline \
  --mode fit_control \
  --plot-image outputs/review_packet/demo_CIV_doublet_z2p6000.plot.png
```

输出文件默认是：

```text
outputs/review_packet/demo_CIV_doublet_z2p6000.llm_control.json
outputs/review_packet/demo_CIV_doublet_z2p6000.fit_control_patch.json
```

这个文件只验证接口和 schema，不代表科学判断。

把第一层 patch 应用回拟合器：

```bash
.venv/bin/python scripts/apply_fit_control_patch.py \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --window-csv outputs/review_packet/demo_CIV_doublet_z2p6000.window.csv \
  --patch-json outputs/review_packet/demo_CIV_doublet_z2p6000.fit_control_patch.json
```

离线 client 默认不发工具调用，所以这个 refit 只验证链路。真实多模态 provider 发出 `add_continuum_anchor`、`set_fit_mask_interval`、`add_absorption_source` 等工具调用后，patch 会改变下一轮连续谱和 Voigt 拟合。

`add_continuum_anchor` 支持 ABSpec 节点式用法：只给 `wavelength_A` 时工具会取最近可用像素 flux；同时给 `wavelength_A` 和 `continuum_flux` 时，会把它作为显式连续谱节点插入样条/PCHIP 重建。`remove_continuum_anchor` 用当前 `plot_data.anchor_wavelengths_A` 的节点序号删除节点。

`add_absorption_source` / `update_absorption_source` 现在支持更多先验字段：

- `center_prior_sigma_kms`
- `center_prior_half_width_kms`
- `logN`
- `logN_lower` / `logN_upper`
- `b_kms`
- `b_kms_lower` / `b_kms_upper`

refit 结果不是自动接受。检查 `*_refit.review.json` 里的 `fit_control_evaluation`：

- `accepted`：没有新增警告，质量未退化，RMS/组件数/fit pixels 都通过 deterministic gate。
- `needs_human_review`：可能改善了部分指标，但仍有 inspect/high residual 等风险。
- `rejected`：refit 失败、RMS 明显变差、组件数暴涨、mask/window 破坏了拟合像素，或质量退化。
