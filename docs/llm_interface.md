# LLM 接口说明

这份文档描述当前开发切片，不代表第一部分拟合 agent 已完成。当前已经固定了单轮工具调用、patch 记录、refit gate，并提供一个有界多轮 agent/tool/refit loop runner。

当前 LLM 接口分两层：

- 第一层 `fit_control`：多模态拟合控制 agent，能看 `*.plot.png`，可以加/删/调 source，改 fit window，改 continuum anchors，改 continuum mask，改 absorption mask，然后请求 refit。
- 第二层 `fit_review`：对第一层输出做结构化复核和解释。

当前的设计重点是“多轮探索 + gate + 人工兜底”，不是“模型一票否决”。模型可以在一轮里同时提出多个 source、window 和 mask 编辑；如果 refit 后没有明显退化但仍然不够稳，结果会被保留并转为 `needs_human_review`，而不是直接当作失败丢掉。

## 传给 LLM 的物理上下文

`review.json` 的 `input.line_family_context` 是来自 `configs/line_catalog.json` 的软背景知识，不是硬过滤规则。它包含：

- `ion`、`family`、`multiplet_type`
- `transition_line_ids`
- `oscillator_strengths` / `oscillator_strength_ratio`
- `soft_background.summary`
- `soft_background.expected_relation`
- `soft_background.outlier_policy`
- `soft_background.agent_guidance`

对 C IV / Mg II doublet，LLM 会被告知：相近 velocity 的 sibling transition 结构是 absorber 解释的支持证据；未饱和干净吸收中强线通常更深，饱和时强弱比可能接近 1:1。但这不是自动验收或自动剔除规则。单成员特征、sibling 不一致、强弱比异常都要保留并报告，可能对应 blend、污染、bad pixels、低 S/N、饱和、谱窗边缘效应或真实模糊样本。LLM 不应仅因为另一个 panel 弱或缺失就删除 source 或加 mask。

每个 `transition_frame` 也会带 `ion`、`partner_line_id` 和 `family_context`，帮助模型理解 velocity-frame 图之间的关系。

`fit_control` prompt 不是完整 `review.json` 的逐字段转储。完整 per-pixel residual、diagnostic residual、posterior initializer 细节和 provider raw payload 仍保存在 review JSON/CSV artifact 中；发给 LLM 的版本只保留决策摘要：当前 fit metrics、每条 transition 的窗口/质量/top residual 摘要、核心 component 参数、LSF 诊断摘要、loop history 和上一轮 gate feedback。上一轮 refit feedback 也只注入 summary，避免在第二轮重复塞入完整 fit summary。

## LSF 诊断边界

当前主拟合仍是 intrinsic physical Voigt posterior：公开参数、主 `voigt_model`、主 residual 和 gate 指标来自没有 LSF 卷积的 posterior median 模型。这样避免在没有真实 LSF 时用猜测的 Gaussian/常数分辨率污染参数。

如果输入 metadata 提供 per-transition LSF/resolution matrix：

```json
{
  "lsf_matrices_by_transition": {
    "MGII_2796": [[...], "..."]
  },
  "lsf_source": "desi_resolution_matrix"
}
```

拟合器会额外计算同一组 intrinsic posterior 参数经过 LSF matrix 后的观测空间模型，并写入并列诊断：

- `fit_results[0].lsf`
- `fit_results[0].instrument_lsf_applied`
- `fit_results[0].instrument_lsf_applied_to_fit_likelihood`
- `transition_frames[].lsf`
- `transition_frames[].lsf_diagnostic`
- `plot.csv` 的 `voigt_lsf_model`、`voigt_lsf_residual`、`voigt_lsf_residual_sigma`
- `model.csv` 的 `smooth_lsf_model`

`instrument_lsf_applied_to_fit_likelihood` 当前固定为 `false`。也就是说，LSF 结果是给 LLM 和人工看的对照诊断，不替代主拟合参数。LLM 应比较 intrinsic residual 与 LSF residual 后再决定是否继续实验或转人工。

## 当前开发切片

- `src/astroagent/agent/llm.py`
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
- `src/astroagent/agent/fit_control.py`
  - 把 `fit_control_patch` 转成 deterministic fitter overrides。
  - 支持连续谱 anchors/masks、transition fit windows、fit masks、source seed 增删/更新。
  - source seed 现在可以显式携带 `center_prior_sigma_kms`、`center_prior_half_width_kms`、`logN`、`b_kms` 和上下界。
  - 重新生成 review JSON、plot CSV、model CSV 和 velocity-frame plot PNG。
  - 写入 `fit_control_evaluation`：比较原始拟合和 refit 后拟合，给出 `accepted`、`needs_human_review` 或内部兼容字段 `rejected`。这里的 `rejected` 表示 refit 不进入主状态、效果可能变差，不表示 LLM 介入没有价值。
- `src/astroagent/spectra/voigt_fit.py`
  - transition-frame Voigt 拟合的公开结果必须来自 UltraNest posterior median。
  - least-squares / MAP 只允许作为 initializer 和诊断参考；不能填充 public model、residual、component 参数或 posterior interval。
  - 如果 UltraNest posterior 不可用、不完整，或缺少任一组件参数 median，这个 transition fit 会标记为 posterior unavailable，并进入 review，而不是回退到 initializer。
  - posterior band 只根据实际 q16/median/q84 绘制；不会人为加最小宽度来制造不确定性。
  - 如果 metadata 提供 per-transition LSF matrix，会额外输出 LSF-convolved model/residual 诊断；主 likelihood 和 public parameters 仍保持 intrinsic Voigt。
- `src/astroagent/agent/loop.py`
  - 编排有界多轮 loop：`run_fit_control -> build_fit_control_patch -> refit_record_with_overrides -> write_review_packet`。
  - 不重新实现 provider、tool schema、patch normalization 或 refit gate，只保存每轮 `control_status`、tool 数、gate decision、是否推进到下一轮。
  - `accepted` 和 `needs_human_review` 都可以继续进入下一轮，方便模型继续细化；内部 `rejected` refit 不进入主状态，但会记录非主线 refit 图、patch 和 gate 原因供下一轮参考。
  - 每轮同时把观测波长空间总览图和 velocity/residual 图送给模型，避免只看局部速度窗导致误判。
- `src/astroagent/cli/apply_fit_control_patch.py`
  - 读取原始 `*.review.json`、`*.window.csv` 和 `*.fit_control_patch.json`。
  - 输出 `*_refit.review.json`、`*_refit.plot.csv`、`*_refit.model.csv`、`*_refit.plot.png`。
  - stdout 会打印 `fit-control decision`，便于批处理筛选。
- `src/astroagent/cli/run_fit_control_loop.py`
  - 读取初始 `*.review.json`、`*.window.csv` 和可选 `*.plot.png`。
  - 输出每轮 refit 的 review packet，并写出 `*.fit_control_loop.json` summary。
  - 对外保持和单轮 CLI 一样的 `--client offline|openai-compatible`、`--temperature` 风格。
- `scripts/run_fit_review.py`
  - 本地开发包装入口，不要求先安装包。
- `scripts/run_fit_control_loop.py`
  - 本地开发包装入口，不要求先安装包。

## 当前边界

- 不做整谱盲识别。
- 不把 LLM 输出当最终科学结论。
- 不直接替代 deterministic fitter，但会参与其拟合循环。
- 当前 loop 是有界最小编排，不是长期记忆 agent，也不做最终科学裁决。

## 环境变量

真实 provider 走 `openai-compatible` client：

```bash
export ASTROAGENT_LLM_API_KEY=...
export ASTROAGENT_LLM_MODEL=...
export ASTROAGENT_LLM_BASE_URL=https://api.openai.com/v1
```

`ASTROAGENT_LLM_BASE_URL` 可选；如果使用兼容接口的本地或代理服务，改这个变量即可。也可以直接设置完整 endpoint：

```bash
export ASTROAGENT_LLM_API_URL=https://example.com/v1/chat/completions
```

Paratera GLM-4V 示例：

```bash
export ASTROAGENT_LLM_API_KEY=...
export ASTROAGENT_LLM_MODEL=GLM-4V-Plus-0111
export ASTROAGENT_LLM_BASE_URL=https://llmapi.paratera.com
```

## 本地验证

先生成 review packet：

```bash
.venv/bin/python -m astroagent.cli.main packet
```

再跑离线 LLM control：

```bash
.venv/bin/python -m astroagent.cli.main llm \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --client offline \
  --mode fit_control
```

输出文件默认是：

```text
outputs/review_packet/demo_CIV_doublet_z2p6000.llm_control.json
outputs/review_packet/demo_CIV_doublet_z2p6000.fit_control_patch.json
```

这个文件只验证接口和 schema，不代表科学判断。

把第一层 patch 应用回拟合器：

```bash
.venv/bin/python -m astroagent.cli.main apply-patch \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --patch-json outputs/review_packet/demo_CIV_doublet_z2p6000.fit_control_patch.json
```

离线 client 默认不发工具调用，所以这个 refit 只验证链路。真实多模态 provider 发出 `add_continuum_anchor`、`set_fit_mask_interval`、`add_absorption_source` 等工具调用后，patch 会改变下一轮连续谱和 Voigt 拟合。

运行有界多轮 loop：

```bash
.venv/bin/python -m astroagent.cli.main fit-loop \
  --review-json outputs/review_packet/demo_CIV_doublet_z2p6000.review.json \
  --client offline \
  --max-rounds 2 \
  --hard-max-rounds 6
```

这些统一入口会从 `sample_id` 自动推断同目录的 `*.window.csv`、`*.overview.png` 和 `*.plot.png`；旧的 `scripts/*.py` 包装入口仍然保留给未安装包的本地开发。
`--max-rounds` 是初始完整实验轮预算；每轮包含一次 agent 决策、一次确定性 refit/gate、一次 agent assessment。assessment 会看到本轮刚生成的 refit feedback，并可通过 `request_more_budget` 申请进入下一轮；`--hard-max-rounds` 是绝对上限。loop 结束后会额外生成 `<sample_id>.audit/audit_report.md` 和 `audit_report.json`。报告只汇总已有 loop 历史、fit metrics 和 gate 信号，不替代人工科学裁决。

工具调用会在 LLM 边界和 patch 边界做 required/type/enum 校验；缺字段、非 JSON arguments 或非有限数值会直接失败，不会记录为 `validated` 的 no-op。refit 可以继承累计 controls，但 deterministic gate 的 edit profile 只按当前轮 patch 计数，避免上一轮工具动作污染本轮评估。assessment 阶段只消费 `request_more_budget`；如果模型同时输出 source/mask/window/continuum 编辑，这些编辑不会执行，必须在获批后的下一轮 decision call 中重新提出。

`fit_control_loop` 的停止原因包括：

- `no_refit_requested`：模型没有请求任何会改变拟合输入的工具；单独 `request_refit` 不算有效编辑。
- `needs_human_review`：refit gate 认为没有自动接受条件，应该交给第二层 review 或人工。
- `refit_rejected`：内部兼容名，表示 deterministic gate 判定这轮 patch 不适合进入主状态，效果可能变差或风险过高。
- `max_rounds_reached`：达到有界轮数，保留最后一轮产物供第二层 review 或人工检查。

注意：`request_refit` 本身不算有效编辑。只有真正改了 source、window、mask 或 continuum 才会触发下一轮 refit。

`add_continuum_anchor` 支持 ABSpec 节点式用法：只给 `wavelength_A` 时工具会取最近可用像素 flux；同时给 `wavelength_A` 和 `continuum_flux` 时，会把它作为显式连续谱节点插入样条/PCHIP 重建。`remove_continuum_anchor` 用当前 `plot_data.anchor_wavelengths_A` 的节点序号删除节点。

`add_absorption_source` / `update_absorption_source` 现在支持更多先验字段：

- `center_prior_sigma_kms`
- `center_prior_half_width_kms`
- `logN`
- `logN_lower` / `logN_upper`
- `b_kms`
- `b_kms_lower` / `b_kms_upper`

拟合参数来源有一个硬边界：MAP/least-squares 只用来初始化 posterior 或做诊断对照。`review.json`、`model.csv`、velocity 图、gate 指标和 prompt 摘要里的 component 参数都必须来自 posterior median；posterior 不完整时宁可失败并交给 review，也不使用 initializer 补值。

refit 结果不是自动接受。检查 `*_refit.review.json` 里的 `fit_control_evaluation`：

- `accepted`：没有新增警告，质量未退化，RMS/组件数/fit pixels 都通过 deterministic gate。
- `needs_human_review`：可能改善了部分指标，但仍有 inspect/high residual 等风险。
- `rejected`：内部兼容名；refit 失败、RMS 明显变差、组件数暴涨、mask/window 破坏了拟合像素，或质量退化，因此不进入主状态并交给下一轮/人工复核。

本地真实数据已经跑过一批脏样本，包括 `srt/data` 里的 COS / low-z HI 例子。结论是：loop 脚手架和数据传递已经通，但最终是否继续探索、是否加更多 component、是否改窗口或 mask，仍需要 gate 和人工复核共同决定。
