# Agent 工具层重设计实施文档

> 版本：v1 · 适用分支：`claude/analyze-project-architecture-P0wyI`
> 配套文档：`docs/optimization_proposal.md`（总体架构方案）
> 范围：本文专门处理**让 agent 真正闭环——分析→调整→审计→交付人类**所需的工具层改造。
> 本文档不替代 A1/A2/A3，是它们之后的**第一个交付价值最大的 PR 集合**。

---

## 0. 目标与非目标

### 目标
1. Agent 跑完之后人类拿到的是 **可读的审计报告 (`audit_report.md`)**，而不是一堆 JSON+PNG。
2. Agent 行为是 **inspect → adjust → audit** 的真正闭环，loop 由 agent 主动 `emit_audit_report` 终止，而不是被 `max_rounds` 截断。
3. 发现、疑虑、置信度都是**结构化字段**，可被自动汇总到报告，可被未来的 SFT/RL 直接消费。

### 非目标
- **不**改业务逻辑（Voigt 拟合、gate 阈值、连续谱算法不动）
- **不**做训练管线、不做 DESI 接入
- **不**引入 OpenTelemetry / FastAPI / 数据库等外部依赖
- **不**重写 `loop.py` 全部结构（只在终止条件、round 计费、audit_log 三处插钩子）

---

## 1. 当前工具层根本缺陷（简版回顾）

| # | 缺陷 | 影响 |
|---|---|---|
| C1 | **只有一类工具**（mutate），LLM 没有"读"和"记"的工具 | Agent 瞎改一通看 RMS 怎么变 |
| C2 | **Loop 终止全是被动的**（gate / max_rounds），无 agent 自主"我说完了"信号 | 报告永远是半成品 |
| C3 | **发现/疑虑没有承载体**，全埋在 rationale 文本里 | 人类要从 N 轮 history 里翻 |

---

## 2. 新工具分类

把工具明确分四类，副作用边界清晰、计费规则不同：

| 类别 | 数量 | 副作用 | 计入 round | 阶段 |
|---|---|---|---|---|
| **Inspect** | 4 | 无 | ❌ 不计 | 探索 |
| **Mutate** | 7（保留现有） | 改 candidate state 触发 refit | ✅ 计 | 调整 |
| **Annotate** | 4 | 写 `session.audit_log` | ❌ 不计 | 全程 |
| **Terminal** | 1 | 终止 loop 并触发报告渲染 | — | 收尾 |

**关键规则**：只有 mutate 工具触发的 refit 才消耗 round 配额。这样 agent 敢"先看清楚再动手"——这是行为质量提升的关键开关。

---

## 3. 工具规格（按分类）

### 3.1 Inspect 工具（PR-T2 落地）

#### 3.1.1 `query_residual_at_velocity`

**Schema**
```json
{
  "type": "function",
  "function": {
    "name": "query_residual_at_velocity",
    "description": "Inspect residual statistics in a narrow velocity window without changing fit state.",
    "parameters": {
      "type": "object",
      "properties": {
        "transition_line_id": {"type": "string", "enum": ["CIV_1548", "CIV_1550", "MGII_2796", "MGII_2803", "HI_LYA"]},
        "velocity_kms": {"type": "number", "minimum": -2000, "maximum": 2000},
        "half_width_kms": {"type": "number", "minimum": 5, "maximum": 200, "default": 30}
      },
      "required": ["transition_line_id", "velocity_kms"],
      "additionalProperties": false
    }
  }
}
```

**Tool result 形状**
```json
{
  "n_pixels": 7,
  "max_abs_residual_sigma": 4.2,
  "mean_residual_sigma": -2.8,
  "in_fit_window": true,
  "in_mask": false,
  "wavelength_range_A": [1551.2, 1551.7],
  "shared_with_other_frames": [
    {"transition_line_id": "CIV_1550", "velocity_kms": -318.4}
  ]
}
```

**实现要点**：直接读 `record["fit_results"][0]` 与 `plot_data` 现有字段算切片，**无新计算**。

#### 3.1.2 `compare_doublet_column_density`
检查双线 logN 一致性（C IV / Mg II 各自的 1548/1550、2796/2803 比值物理）。

```json
{
  "args": {
    "primary_transition_line_id": "CIV_1548",
    "sibling_transition_line_id": "CIV_1550",
    "component_index": 0
  }
}
```

返回 `logN_primary`、`logN_sibling`、`expected_ratio_consistency`（`consistent` / `primary_too_low` / `sibling_too_low` / `saturation_limited`）、`saturation_warning`。

#### 3.1.3 `get_component_posterior_band`
返回 v、logN、b 的 16/50/84 分位数 + 是否触上下边界。让 agent 判断"组分是真的 vs 噪声拟合"。

#### 3.1.4 `inspect_continuum_anchor`
按 `anchor_index` 或 `wavelength_A` 查 anchor 是否落在饱和区、邻近吸收线、对当前归一化贡献多大。

---

### 3.2 Mutate 工具（PR-T3 微调）

**保留全部 7 个现有工具**：`add_absorption_source`、`remove_absorption_source`、`update_absorption_source`、`set_fit_window`、`set_fit_mask_interval`、`add_continuum_anchor` / `remove_continuum_anchor` / `update_continuum_mask`、`request_refit`（与 A1 中已纳入原生 tool calling 的 schema 一致）。

**唯一改动**：每个 mutate 工具的 schema **新增必填 `expected_outcome` 字段**：

```json
"expected_outcome": {
  "type": "object",
  "properties": {
    "rms_change_direction": {"type": "string", "enum": ["decrease", "increase", "neutral"]},
    "rms_relative_change_pct": {"type": "number", "minimum": -100, "maximum": 100},
    "n_components_change": {"type": "integer", "minimum": -10, "maximum": 10},
    "narrative": {"type": "string", "minLength": 10, "maxLength": 200}
  },
  "required": ["rms_change_direction", "narrative"],
  "additionalProperties": false
}
```

**Harness 行为**：refit 完成后自动 diff 实际 vs 预期，写入 `audit_log` 一条 `self_calibration` 记录。

---

### 3.3 Annotate 工具（PR-T1 落地）

#### 3.3.1 `record_finding`

**Schema**
```json
{
  "type": "function",
  "function": {
    "name": "record_finding",
    "description": "Record a structured finding into the audit log. Does not affect fit state.",
    "parameters": {
      "type": "object",
      "properties": {
        "category": {"type": "string", "enum": [
          "component_identification", "blend", "saturation",
          "continuum_bias", "projection_overlap", "noise_artifact",
          "doublet_consistency", "other"
        ]},
        "transition_line_id": {"type": "string"},
        "velocity_kms": {"type": "number"},
        "summary": {"type": "string", "minLength": 10, "maxLength": 500},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
      },
      "required": ["category", "summary", "evidence", "confidence"],
      "additionalProperties": false
    }
  }
}
```

#### 3.3.2 `flag_uncertainty`

**Schema**
```json
{
  "args_schema": {
    "topic": "string, 10-200 chars",
    "why_uncertain": "string, 20-500 chars",
    "what_human_should_check": "string, 10-300 chars",
    "severity": "enum: low | medium | high",
    "blocks_finalization": "bool, default false"
  }
}
```

#### 3.3.3 `mark_component_confidence`
对某个已拟合组分打置信标签：`{component_index, confidence: high|medium|low, reason}`。

#### 3.3.4 `request_human_review`
`{scope, specific_question, blocker_for_finalization}`——比 gate 自动 needs_human_review 更具针对性。

**所有 4 个 annotate 工具**：
- 实现 = 把 args 包装成 `AuditEntry` 追加到 `session.audit_log`
- Tool result：返回 `{"ok": true, "audit_entry_id": "..."}`
- **不触发 refit，不计 round 配额**

---

### 3.4 Terminal 工具（PR-T1 落地）

#### `emit_audit_report`

```json
{
  "type": "function",
  "function": {
    "name": "emit_audit_report",
    "description": "Finalize the analysis. This is the LAST tool call; the loop ends after this.",
    "parameters": {
      "type": "object",
      "properties": {
        "overall_status": {"type": "string", "enum": [
          "fit_accepted_clean",
          "fit_acceptable_with_caveats",
          "fit_inconclusive_human_required",
          "data_unfittable"
        ]},
        "executive_summary": {"type": "string", "minLength": 50, "maxLength": 1500},
        "n_components_final": {"type": "integer", "minimum": 0},
        "key_metrics": {
          "type": "object",
          "properties": {
            "fit_rms": {"type": "number"},
            "reduced_chi2": {"type": "number"},
            "max_residual_sigma_in_work_window": {"type": "number"}
          },
          "required": ["fit_rms"]
        },
        "what_was_changed": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "what_was_NOT_changed_and_why": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "human_action_items": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "confidence_overall": {"type": "string", "enum": ["high", "medium", "low"]}
      },
      "required": ["overall_status", "executive_summary", "n_components_final",
                   "key_metrics", "confidence_overall"],
      "additionalProperties": false
    }
  }
}
```

**Harness 行为**：
1. 不再继续 LLM round
2. 把 `audit_log` 中的所有 findings/uncertainties/self_calibration 与本次 args 合并
3. 渲染 `audit_report.md` + `audit_report.json`
4. 把所有相关 plot 拷贝到 `outputs/<sample>/audit/plots/`
5. 写一份 `index.html` 入口

---

## 4. Session 与数据结构改动

### 4.1 `AuditEntry`

```python
# src/astroagent/agent/audit/entries.py
from dataclasses import dataclass
from typing import Literal, Any
from datetime import datetime

@dataclass
class AuditEntry:
    entry_id: str                # uuid4
    timestamp: str               # ISO 8601
    round_index: int
    kind: Literal["finding", "uncertainty", "component_confidence",
                  "human_review_request", "self_calibration"]
    payload: dict[str, Any]      # 原始 tool args（或 self_calibration diff）
    tool_call_id: str | None     # provider 端 tool_call id（self_calibration 为 None）
```

### 4.2 `Session` 字段新增

在 A3 已规划的 `AgentSession` 上增加：
```python
audit_log: list[AuditEntry] = field(default_factory=list)
audit_emitted: bool = False
audit_payload: dict | None = None    # emit_audit_report args 缓存
```

如果 A3 尚未落地，本 PR 在现有 `record["fit_control_loop"]` 旁同级新增 `record["audit_log"]: list`，结构兼容。

---

## 5. Loop 改动

### 5.1 终止条件优先级

```
1. emit_audit_report 被调用       → stop_reason = "audit_emitted"
2. 业务 stop (no_refit_requested) → stop_reason = "no_refit_requested"
3. gate rejected (last round)     → stop_reason = "refit_rejected"
4. needs_human_review (last round)→ stop_reason = "needs_human_review"
5. max_rounds 到达                 → stop_reason = "max_rounds_reached_without_audit"
                                     (此时 harness 自动用兜底字段渲染不完整报告)
```

### 5.2 Round 计费

`loop.py` 中的 `round_index` 自增条件改为：**candidate_patch.requires_refit == True**（即 mutate 工具产生了实际 refit 请求）。

inspect 与 annotate 调用走快路径：直接执行、追加 audit_log、把 tool_result 回灌进 history、**不增加 round_index**、**不调 fit_control 重拟合**。

### 5.3 主循环伪代码

```python
def run_fit_control_loop(...):
    session = AgentSession(...)
    round_used = 0
    while round_used < max_rounds and not session.audit_emitted:
        result = session.step()        # LLM call → 解析 tool_calls

        for call in result.tool_calls:
            kind = classify_tool(call.name)  # inspect | mutate | annotate | terminal
            if kind == "terminal":
                session.audit_payload = call.args
                session.audit_emitted = True
                break
            elif kind == "annotate":
                _execute_annotate(session, call)
            elif kind == "inspect":
                _execute_inspect(session, call)
            elif kind == "mutate":
                _queue_mutate(session, call)

        if session.has_pending_mutations():
            _refit_and_evaluate(session)   # 现有 fit_control 流程
            round_used += 1

    _render_audit(session, output_dir)
    return ...
```

### 5.4 兜底报告（agent 没有 emit）

如果 `audit_emitted == False` 且 `round_used == max_rounds`，harness 用以下字段自动构造一份 `overall_status="fit_inconclusive_human_required"` 的报告：

- `executive_summary`：从最后一轮 LLM rationale 摘录前 800 字符 + 标记 `[auto-generated, agent did not finalize]`
- `key_metrics`：从最后一次 fit_results 取
- `human_action_items`：自动加一条 `"Agent did not finalize within max_rounds; please review manually."`

---

## 6. Audit 渲染层

### 6.1 文件结构

```
src/astroagent/agent/audit/
├── __init__.py
├── entries.py             # AuditEntry dataclass
├── classifier.py          # classify_tool(name) -> 四分类
├── renderer.py            # render_audit_report(session, output_dir)
└── templates/
    ├── audit_report.md.j2
    ├── index.html.j2
    └── components_summary.csv.j2
```

### 6.2 输出目录布局

```
outputs/<sample_id>/audit/
├── audit_report.md          # 人类阅读主入口
├── audit_report.json        # 机器消费版
├── findings.jsonl           # 每条 record_finding 一行
├── uncertainties.jsonl      # 每条 flag_uncertainty 一行
├── components_summary.csv   # 每个组分一行
├── self_calibration.jsonl   # 预期 vs 实际 diff（PR-T3 之后填充）
├── trace/
│   ├── round_1.json
│   ├── round_2.json
│   └── ...
├── plots/
│   ├── final_overview.png
│   ├── final_per_transition.png
│   └── round_3_pre_audit.png
└── index.html
```

### 6.3 Markdown 模板（`audit_report.md.j2`）

```jinja
# Audit Report — {{ sample_id }}

**Status:** `{{ overall_status }}`
**Confidence:** `{{ confidence_overall }}`
**Final components:** {{ n_components_final }}
**Generated:** {{ timestamp }}
**Session:** `{{ session_id }}`

---

## Executive Summary

{{ executive_summary }}

## Key Metrics

| Metric | Value |
|---|---|
| Fit RMS | {{ "%.4f"|format(key_metrics.fit_rms) }} |
| Reduced χ² | {{ "%.3f"|format(key_metrics.reduced_chi2) if key_metrics.reduced_chi2 is not none else "—" }} |
| Max residual (σ, work window) | {{ "%.2f"|format(key_metrics.max_residual_sigma_in_work_window) if key_metrics.max_residual_sigma_in_work_window is not none else "—" }} |

## Final Components

| # | Transition | v (km/s) | logN | b (km/s) | Confidence | Notes |
|---|---|---|---|---|---|---|
{% for c in components -%}
| {{ c.index }} | {{ c.transition_line_id }} | {{ "%.1f"|format(c.center_velocity_kms) }} | {{ "%.2f"|format(c.logN) }} ± {{ "%.2f"|format(c.logN_err) }} | {{ "%.1f"|format(c.b_kms) }} ± {{ "%.1f"|format(c.b_err) }} | `{{ c.confidence }}` | {{ c.notes }} |
{% endfor %}

## What I changed and why

{% for change in what_was_changed %}
- {{ change }}
{% endfor %}

## What I did **not** change and why

{% for hold in what_was_NOT_changed_and_why %}
- {{ hold }}
{% endfor %}

## Findings ({{ findings|length }})

{% for f in findings %}
### `[{{ f.confidence }}]` {{ f.category }} — {{ f.transition_line_id or "(global)" }} @ v={{ "%.0f"|format(f.velocity_kms) if f.velocity_kms is not none else "—" }} km/s

{{ f.summary }}

**Evidence:**
{% for e in f.evidence %}- {{ e }}
{% endfor %}

{% endfor %}

## Uncertainties flagged for human review ({{ uncertainties|length }})

{% for u in uncertainties %}
- **`[{{ u.severity }}]` {{ u.topic }}**
  - Why uncertain: {{ u.why_uncertain }}
  - **Action requested:** {{ u.what_human_should_check }}
  {% if u.blocks_finalization %}- ⚠ This issue blocks finalization.{% endif %}
{% endfor %}

## Human Action Items

{% for item in human_action_items %}- [ ] {{ item }}
{% endfor %}

{% if self_calibration_entries %}
## Self-Calibration (predicted vs actual)

| Round | Tool | Predicted | Actual | Match |
|---|---|---|---|---|
{% for s in self_calibration_entries -%}
| {{ s.round_index }} | `{{ s.tool_name }}` | {{ s.expected_narrative }} | {{ s.actual_narrative }} | {{ "✓" if s.matched else "✗" }} |
{% endfor %}
{% endif %}

---

*Generated by AstroAgent v{{ version }} · stop reason: `{{ stop_reason }}`*
```

### 6.4 `index.html`

最简实现：单页 HTML，左侧目录、右侧渲染好的 markdown（用 `marked.js` 走 CDN 或离线 JS 包）+ 图片缩略图网格。**不引入 React/Vue 等前端框架**。

---

## 7. PR 拆分（实施顺序）

### PR-T1 — Annotate + Terminal + 审计渲染（最大用户价值）

**新增**
- `src/astroagent/agent/audit/{__init__,entries,classifier,renderer}.py`
- `src/astroagent/agent/audit/templates/{audit_report.md.j2,index.html.j2,components_summary.csv.j2}`
- `src/astroagent/agent/tools/annotate.py`（4 个 tool 的 schema + 执行函数）
- `src/astroagent/agent/tools/terminal.py`（`emit_audit_report`）
- `tests/test_audit_renderer.py`
- `tests/test_annotate_terminal_tools.py`

**修改**
- `src/astroagent/agent/loop.py`：
  - 新增 `_execute_annotate` / `_classify_tool_calls` / `_render_audit`
  - 修改终止条件优先级
  - round 计费改为只在 mutate 触发 refit 时累加
- `src/astroagent/agent/llm.py`：
  - prompt 中新增 annotate/terminal 工具的使用指南
  - 把 4+1 个新工具加入传给 LLM 的 tool list
- `pyproject.toml`：加 `jinja2>=3.1` 依赖；`package_data` 加 `astroagent.agent.audit.templates`
- `docs/optimization_proposal.md`：第 4 章追加 "T1 完成" checkbox

**预计代码量**：`+650 / -50` LOC，**2–3 天**

**验收清单**
- [ ] 给定一段 mock LLM 序列（含 inspect → mutate → annotate × 3 → emit_audit_report），跑完后 `outputs/<id>/audit/audit_report.md` 存在且包含所有 finding
- [ ] LLM 不调 emit_audit_report 而被 max_rounds 截断时，自动生成兜底报告，`overall_status == "fit_inconclusive_human_required"`
- [ ] `index.html` 在浏览器里能正确显示报告与缩略图
- [ ] 现有 `tests/test_fit_control_loop.py` 不破坏（可能需要更新 stop_reason 断言）

---

### PR-T2 — Inspect 工具

**新增**
- `src/astroagent/agent/tools/inspect.py`（4 个 inspect 工具 + 执行函数）
- `tests/test_inspect_tools.py`

**修改**
- `src/astroagent/agent/loop.py`：新增 `_execute_inspect` 路径
- `src/astroagent/agent/llm.py`：
  - prompt 中**删除**主动推送 `velocity_frame_wavelength_overlaps`、`sibling_work_window_projections`、`residual_hints` 的大段（约 15–20 条规则）
  - 改为指引 "需要这些信息时调用 inspect 工具"
- `src/astroagent/agent/llm.py:_compact_fit_summary`：进一步精简

**预计代码量**：`+450 / -350` LOC，**2 天**

**验收清单**
- [ ] 单条 fit_control prompt token 比 PR-T1 后再降 30%（用相同样本对比）
- [ ] 离线 mock 测试：LLM 调 `query_residual_at_velocity(CIV_1548, -150, 30)` → tool result 字段齐全且数值与直接读取 `record` 一致
- [ ] inspect 调用不增加 `round_index`

---

### PR-T3 — Mutate 工具加 `expected_outcome` + 自我校准

**修改**
- `src/astroagent/agent/tools/mutate.py`（如 PR-T1 已重构则在此处；否则在 `agent/llm.py` 的 schema 列表中）：7 个 mutate 工具加 `expected_outcome` 必填字段
- `src/astroagent/agent/audit/self_calibration.py`（新增）：refit 后比对 expected vs actual，生成 `AuditEntry(kind="self_calibration")`
- `src/astroagent/agent/loop.py`：refit 完成后调用 self_calibration
- `src/astroagent/agent/audit/templates/audit_report.md.j2`：新增 self-calibration 段（已在模板中预留）
- `tests/test_self_calibration.py`

**预计代码量**：`+250 / -50` LOC，**1 天**

**验收清单**
- [ ] LLM 漏填 `expected_outcome` → schema 校验拒绝（被 provider 拒）
- [ ] expected `rms_change_direction=decrease` 但实际 RMS 上升 → audit_log 中标 `matched=false`
- [ ] 报告 self-calibration 段显示完整对比表

---

## 8. 测试策略

### 8.1 新增 fixture

`tests/fixtures/audit/`：
- `mock_llm_sequence_clean_fit.json`：3 轮，最后一轮 emit_audit_report，overall_status=`fit_accepted_clean`
- `mock_llm_sequence_with_uncertainty.json`：5 轮，含 2 条 flag_uncertainty，emit 时 overall_status=`fit_acceptable_with_caveats`
- `mock_llm_sequence_no_emit.json`：max_rounds 截断，触发兜底报告
- `mock_llm_sequence_inspect_heavy.json`（PR-T2）：第 1 轮先 4 次 inspect 再 1 次 mutate，验证 round 计费

### 8.2 黄金报告测试

- `tests/golden/audit_reports/clean_fit.md`：固化期望输出
- `tests/test_audit_golden.py`：跑 mock 序列 → 生成报告 → 与 golden 文本比对（按行 diff，时间戳字段做字符串替换）

### 8.3 集成测试

- `tests/test_end_to_end_audit.py`：从合成谱开始，跑完整 loop，断言 `outputs/<id>/audit/` 下所有期望文件存在且 markdown 可被 `markdown` 库无错解析

---

## 9. 向后兼容

| 影响面 | 处理 |
|---|---|
| `record["fit_control_loop"]` 字段 | 保留，新字段 `record["audit_log"]` 平行存在 |
| 现有 `stop_reason` 取值 | 保留全部，新增 `"audit_emitted"` 与 `"max_rounds_reached_without_audit"` |
| `tests/test_fit_control_loop.py` 已有断言 | 仅在 stop_reason 检查处需要更新；其他不破坏 |
| 既有 CLI `astroagent-run-fit-control-loop` | 行为相同，**默认输出多一个 `audit/` 目录**；加 `--no-audit` flag 可关闭（默认不关） |
| LLM prompt 模板（如 A2 已落地） | T1 在 system prompt 末尾追加 "Annotate as you go; emit_audit_report when done" 段；T2 删除一批冗余规则 |

---

## 10. 关键风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| LLM 永远不调 `emit_audit_report` | 中 | system prompt 显式指令 + 兜底报告机制 + 错误率写进 metrics 用作回归监控 |
| Annotate 工具被滥用（每轮调几十次） | 中 | schema 字段 minLength 约束 + audit_log 长度软上限（>100 条触发 warning） |
| `expected_outcome` 拉低 LLM 决策质量（多想了反而出错） | 低 | T3 落地后做 N=20 离线 A/B：开/关 `expected_outcome` 比较 audit 准确率 |
| Markdown 模板渲染失败导致 loop 失败 | 低 | 渲染失败时降级输出 raw JSON 并标 `report_render_failed=true`；不阻塞主流程 |
| 模板字段在 schema 演进时漂移 | 中 | 模板里所有访问加 `is defined`/`is not none` 守卫；增加 `tests/test_audit_renderer.py` 的 missing-field 用例 |

---

## 11. 实施顺序与里程碑

```
Day 0:  阅读本文档 + A1/A2/A3 状态对齐
Day 1-3: PR-T1   (annotate + terminal + 渲染)        ← 最大用户价值
Day 4-5: PR-T2   (inspect 工具 + prompt 瘦身)
Day 6:   PR-T3   (expected_outcome + self_calibration)
Day 7:   联调 + 文档更新 + 给 1-2 个真实样本跑通审计报告
```

**与 A 系列的依赖**：
- T1 依赖 A1（原生 tool calling）：必须先有 tool schema 注册机制
- T1 不强依赖 A2（prompt 三段式）：但若 A2 已落地，T1 在规则段追加 4 条新工具指引最自然
- T1 强烈推荐先于 A3：T1 中扩展的 `audit_log` 与 A3 的 `Session` 是天然合一的，A3 直接接管即可

---

## 12. 第一行代码该写哪？

建议从 `src/astroagent/agent/audit/entries.py` 开始——`AuditEntry` 数据类是后续所有改动的"接口契约"，先把它定下来：

```python
# src/astroagent/agent/audit/entries.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

AuditEntryKind = Literal[
    "finding",
    "uncertainty",
    "component_confidence",
    "human_review_request",
    "self_calibration",
]

@dataclass
class AuditEntry:
    entry_id: str
    timestamp: str
    round_index: int
    kind: AuditEntryKind
    payload: dict[str, Any]
    tool_call_id: str | None = None

    @classmethod
    def make(cls, *, kind: AuditEntryKind, round_index: int,
             payload: dict[str, Any], tool_call_id: str | None = None) -> "AuditEntry":
        return cls(
            entry_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            round_index=round_index,
            kind=kind,
            payload=dict(payload),
            tool_call_id=tool_call_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "round_index": self.round_index,
            "kind": self.kind,
            "payload": self.payload,
            "tool_call_id": self.tool_call_id,
        }
```

定下来之后，4 个 annotate 工具的执行函数就是一个 5 行的 wrapper，audit renderer 也只需消费 `list[AuditEntry]`——一次定义，三处使用，PR-T1 的脚手架就立起来了。

---

## 附录 A：完整的 audit_report.json 示例

```json
{
  "schema_version": 1,
  "sample_id": "desi_civ_J0123_z2.34",
  "session_id": "loop_20260509T101523Z_a3f4",
  "timestamp": "2026-05-09T10:18:42Z",
  "stop_reason": "audit_emitted",
  "overall_status": "fit_acceptable_with_caveats",
  "confidence_overall": "medium",
  "n_components_final": 3,
  "key_metrics": {
    "fit_rms": 0.918,
    "reduced_chi2": 1.152,
    "max_residual_sigma_in_work_window": 2.81
  },
  "executive_summary": "C IV doublet at z_abs=2.34 fitted with 3 components ...",
  "what_was_changed": [
    "Removed 1 spurious component at v=+450 km/s",
    "Masked sibling-projection overlap in CIV_1550 frame [-340, -300] km/s"
  ],
  "what_was_NOT_changed_and_why": [
    "Did not remove component 3 despite low confidence — needs human inspection of raw flux"
  ],
  "human_action_items": [
    "Verify whether component 3 is a separate system or Mg II contamination at λ≈1552Å",
    "Confirm continuum normalization on the red side"
  ],
  "components": [
    {"index": 0, "transition_line_id": "CIV_1548", "center_velocity_kms": -148.2,
     "logN": 13.78, "logN_err": 0.04, "b_kms": 8.1, "b_err": 1.2,
     "confidence": "high", "notes": "Doublet-consistent."},
    {"index": 1, "transition_line_id": "CIV_1548", "center_velocity_kms": 22.4,
     "logN": 14.05, "logN_err": 0.03, "b_kms": 11.6, "b_err": 0.8,
     "confidence": "high", "notes": ""},
    {"index": 2, "transition_line_id": "CIV_1548", "center_velocity_kms": 181.7,
     "logN": 12.94, "logN_err": 0.18, "b_kms": 28.4, "b_err": 5.1,
     "confidence": "low", "notes": "May be Mg II blend."}
  ],
  "findings": [...],
  "uncertainties": [...],
  "self_calibration_entries": [...]
}
```

---

*本文档落地完成后，`docs/optimization_proposal.md` 第 4 章追加完成情况；后续若 audit 数据格式演进，更新本文档的"附录 A"与 schema_version 版本号。*
