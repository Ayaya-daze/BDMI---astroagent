# AstroAgent 架构优化建议书

> 版本：v1 · 适用分支：`claude/analyze-project-architecture-P0wyI`
> 范围：本文不引入业务功能，仅就**工程架构、Agent 工作层、可复现性、训练管线衔接**给出按阶段排序的优化方案。
> 阅读对象：项目维护者、未来贡献者、做后训练实验的同学。

---

## 0. TL;DR

当前仓库（`src/astroagent/` 共 ~8 K LOC）实现了 MVP 范围内的 review packet 生成、确定性 Voigt 拟合、LLM 拟合控制多轮闭环。架构思路清晰，文档完备，测试覆盖核心路径。**主要瓶颈集中在 Agent 工作层**：

1. 不是真正的 tool-use loop，而是"伪多轮单发"——LLM 从未以 `role=tool` 消息看到自己工具调用的执行结果。
2. 走 `response_format=json_object` 而非 provider 原生 tool calling，工具 schema 全靠 prompt 文字描述，校验薄弱。
3. System prompt 单次约 2 000 token 且每轮重建，**prompt cache 命中率 0**。
4. Session 状态藏在 `record` 字段里（`fit_control_last_refit_feedback` 等），不可序列化、不可恢复、难以并发隔离。
5. 候选选择是顺序贪心，没有 best-of-N，浪费 LLM 决策的随机性，也错失了天然的 RL preference data 来源。
6. 错误处理脆弱（无重试、无 JSON repair、无 checkpoint）；图片成本无治理。

此外有项目级问题：常量散落、无统一日志、手写 schema 校验、`voigt_fit.py` 单文件 1 596 行、训练管线尚未铺设。

本文给出 **6 个阶段、20 余项具体改造**，并对最关键的 **A1/A2/A3 三个 PR** 给出可直接动手的实施清单。

---

## 1. 项目现状与诊断

### 1.1 仓库地图（核心模块）

```
src/astroagent/        ~8 032 LOC
├── spectra/   1 313   Voigt 物理模型 + UltraNest 贝叶斯拟合
│   └── voigt_fit.py 1596 LOC（**单文件过大**）
├── review/    1 894   连续谱 + 速度坐标 + 诊断图 + 复核包
├── agent/     3 411   LLM 客户端 + 工具调用补丁 + 多轮闭环
│   ├── llm.py        1 041 LOC
│   ├── fit_control.py 1 727 LOC
│   └── loop.py        576 LOC
├── cli/         310   5 个入口命令
└── data/        351   DESI 公开星表访问
tests/         ~2 808
```

### 1.2 工程质量问题（按优先级）

| 优先级 | 问题 | 位置 | 影响 |
|---|---|---|---|
| P0 | 物理常量重复定义 `C_KMS = 299792.458` | `spectra/voigt_fit.py:29`, `review/continuum.py:11` | 调参易漂移 |
| P0 | km/s 阈值散布魔数 | `voigt_fit.py:74,80,133`, `fit_control.py:494,560` | 不可解释 |
| P0 | 无统一日志 | 全项目 | 长循环失败难追踪 |
| P0 | Schema 用手写字典校验 | `agent/llm.py:15`, `fit_control.py:961` | LLM 越界值校验弱 |
| P1 | `voigt_fit.py` 单文件 1 596 行 | — | 维护困难 |
| P1 | 无随机种子约定 | UltraNest / scipy | 复现性差 |
| P1 | 多模态图片无尺寸压缩 | `agent/llm.py:1012` | token 浪费 |
| P2 | `data/raw|interim|processed` 仅 `.gitkeep` | — | 数据流未跑通 |
| P2 | 无训练管线代码 | — | 项目目标关键路径未铺设 |

### 1.3 Agent 工作层问题（深度诊断）

#### 问题 1：根本不是真正的 tool loop

`loop.py:104-235` 的多轮结构：
```
for round in 1..max_rounds:
    LLM 一次性吐出全部 tool_calls
    → patch 层批量规整
    → 确定性 refit
    → gate 评判
    → 把 evaluation 字段塞回 record
```
LLM **从未以 `role=tool` 消息收到工具执行结果**。跨轮反馈通过 `record["fit_control_last_refit_feedback"]` 在下一轮重新 build 整个 prompt 实现。

**后果**：
- LLM 每轮都"冷启动"，无法做 chain-of-thought 跨轮延续
- 失去 provider 端原生 `tool_call_id` ↔ `tool_result` 配对
- 无法自然地导出为 SFT/RL 训练样本（messages 累积才是天然格式）

#### 问题 2：`response_format=json_object` 而非原生 tool calling

`llm.py:130-134` 虽然支持 `tools` 参数，但调用链上**没人传**——`run_fit_control` 永远走 `json_object` 分支，工具 schema 全嵌在 prompt 文本里。

**后果**：
- 失去 provider 端 JSON Schema 校验（OpenAI/Anthropic 都做）
- 工具描述吃 ~300–500 prompt token
- 无法用 `parallel_tool_calls=False` 强制串行

#### 问题 3：System prompt 单次 ≈ 2 000 token，且每轮重建

`llm.py:385-540` 是**条件指令墙**，包含 30+ 条 if-then 规则。当前实现把动态 record 直接拼进单一 user message → 前缀不稳 → **prompt cache 命中率为 0**。

#### 问题 4：Session 状态藏在 record 字段里

跨轮上下文通过这些字段隐式传递：
- `record["fit_control_loop"]`（历史）
- `record["fit_control_last_refit_feedback"]`
- `record["fit_control_last_rejected_refit"]`
- `record["fit_control_selected_candidate"]`

**后果**：record 越来越胖，无显式 `Session` 抽象，难以做 (a) 持久化恢复 (b) 并发隔离 (c) mock 测试。

#### 问题 5：候选选择是顺序贪心

`loop.py:199-205` 每轮采样**一个** LLM 决策，按 score 选 best。物理拟合是确定性的，但 LLM 决策有方差——这部分多样性被完全浪费。

#### 问题 6：错误处理脆弱

- `_loads_first_json_object`（`llm.py:995`）解析失败直接抛
- `OpenAICompatibleClient.complete` 无重试、无限流退避
- 没有 prompt 过长降级策略
- 多模态图片 base64 无尺寸检查
- 无 checkpoint，长 loop 崩溃要重跑

---

## 2. 分阶段优化路线图

| 阶段 | 内容 | 预计工时 | 关键产出 |
|---|---|---|---|
| **0** | 工程基建：常量集中、日志、拆分 voigt_fit、Pydantic schema | 1–2 天 | 后续所有迭代的地基 |
| **1** | 复现性与成本治理：种子、图片压缩、LLM 重试、prompt 模板 | 2–3 天 | 实验可复现，账单可控 |
| **2** | 测试加固：golden-file、recording replay、DESI mark | 2 天 | 防止数值回归 |
| **3** | 数据闭环：统一 Spectrum 格式、DESI 批量、合成谱、专家轨迹 | 1–2 周 | 后训练数据来源 |
| **4** | 评估与训练：evals 模块、SFT、RL（GRPO/DPO） | 4–8 周 | 项目目标达成 |
| **5** | 部署与可观测：FastAPI、trace 持久化 | 按需 | 投产能力 |

**Agent 工作层的 A1/A2/A3 跨阶段 0–1，是本次重点，单独在第 3 节展开。**

---

## 3. Agent 工作层重构：A1 / A2 / A3 详细实施方案

> **总体目标**：把伪多轮单发 → 改造为 OpenAI/Anthropic 原生 tool-use 循环；把 prompt 模板化以触发 prompt caching；引入显式 `Session` 抽象使每条交互直接可作为 SFT 样本。
>
> 完成后预期：单条 record 处理成本降 ≥ 50%，prompt cache 命中率 ≥ 80%，`loop.py` 从 576 行降到 ~250 行。

### 3.1 A1：工具走 provider 原生 tool calling

#### 目标
- 删除 `_loads_first_json_object` 链路
- 工具参数越界由 provider schema 验证拦截
- 工具描述从 prompt 文本中剥离 → 节省 300–500 token

#### 实施清单

**新增文件**

```
src/astroagent/agent/tools/
├── __init__.py
└── registry.py
```

`registry.py` 内容（示意）：

```python
from typing import Final

ADD_ABSORPTION_SOURCE: Final = {
    "type": "function",
    "function": {
        "name": "add_absorption_source",
        "description": (
            "Add a new Voigt component seed at a velocity within the source "
            "work window. Use only when the wavelength overview shows an "
            "independent target trough/shoulder."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transition_line_id": {
                    "type": "string",
                    "enum": ["CIV_1548", "CIV_1550", "MGII_2796", "MGII_2803", "HI_LYA"],
                },
                "delta_v_kms": {"type": "number", "minimum": -2000, "maximum": 2000},
                "initial_logN": {"type": "number", "minimum": 10.0, "maximum": 18.0},
                "initial_b_kms": {"type": "number", "minimum": 2.0, "maximum": 120.0},
                "reason": {"type": "string", "minLength": 10, "maxLength": 400},
            },
            "required": ["transition_line_id", "delta_v_kms", "reason"],
            "additionalProperties": False,
        },
    },
}

# ... REMOVE_ABSORPTION_SOURCE / UPDATE_ABSORPTION_SOURCE / SET_FIT_WINDOW
# ... ADD_FIT_MASK_INTERVAL / SET_CONTINUUM_ANCHOR / REQUEST_REFIT

FIT_CONTROL_TOOLS: Final = [
    ADD_ABSORPTION_SOURCE,
    REMOVE_ABSORPTION_SOURCE,
    UPDATE_ABSORPTION_SOURCE,
    SET_FIT_WINDOW,
    ADD_FIT_MASK_INTERVAL,
    SET_CONTINUUM_ANCHOR,
    REQUEST_REFIT,
]
```

**修改文件**

1. `agent/llm.py`
   - `OpenAICompatibleClient.complete`：当 `tools` 非空时走 tool calling 分支（已有骨架，确认可用）
   - `run_fit_control`：传 `tools=FIT_CONTROL_TOOLS`，从 `result.tool_calls` 直接读结构化字典
   - 删除 `_loads_first_json_object` 在 fit_control 路径上的调用（保留在 fit_review 路径，后续再清理）
   - `validate_fit_control` 改为只校验顶层 `task/status/rationale`，参数校验下沉到 provider

2. `agent/llm.py:385-540` 的 prompt
   - 删除工具用法的字段约束描述（已经在 schema 里）
   - 保留**何时使用何工具**的物理决策指南

**新增测试**

`tests/test_tool_calling.py`：
- `OfflineReviewClient` 在 tool-calling 模式下返回固定 tool_calls
- 越界 `delta_v_kms=99999` 在传入 schema 校验时被拒
- `OpenAICompatibleClient` 的 mock 返回 `tool_calls` 字段时正确解析

#### 验收标准

- [ ] `pytest` 全绿
- [ ] `grep -n "_loads_first_json_object" src/astroagent/agent/llm.py` 只剩 fit_review 路径
- [ ] 单条 fit_control 调用 prompt token ≤ 1 600（baseline ~2 000）
- [ ] 新增至少 3 个越界参数被拒的单元测试

#### 预计代码量

`+200 / -100`，4–6 小时

---

### 3.2 A2：Prompt 三段式重构 + Prompt Caching

#### 目标
- System prompt 前缀稳定 → 触发 OpenAI 自动缓存（≥1 024 token）/ Anthropic `cache_control`
- 让规则可版本化、可 A/B 实验
- user message 只承载本轮真正动态的内容

#### 实施清单

**新增文件**

```
src/astroagent/agent/prompts/
├── system_core.txt          # 角色定位 + 物理基础（~400 token，永不变）
├── system_rules.yaml        # 30 条规则结构化
├── system_tool_guide.txt    # 工具使用注意事项（稳定）
└── user_round.j2            # 当前轮动态内容（Jinja2）
```

`system_rules.yaml` 结构示例：

```yaml
- id: continuum_first
  when: ["fit_summary.quality in [inspect, saturated]", "continuum_visibly_biased"]
  guidance: |
    Use continuum-first triage: if the wavelength-space overview shows the
    local continuum is visibly biased, source/mask edits made on that
    normalization are usually not meaningful until continuum is corrected.
  priority: high

- id: do_not_repeat_continuum_only
  when: ["feedback.warnings includes continuum_changed_fit_not_directly_comparable"]
  guidance: |
    Do not repeat the same continuum-only anchor removals; the next useful
    action must be a source/fit-mask/fit-window follow-up.
  priority: high

# ... 其余 28 条规则
```

**修改文件**

1. `pyproject.toml`：加 `jinja2>=3.1`，`[tool.setuptools.package-data]` 配 `astroagent.agent.prompts = ["*.txt", "*.yaml", "*.j2"]`

2. `agent/llm.py`
   ```python
   from importlib.resources import files
   from jinja2 import Environment, FileSystemLoader

   _PROMPT_ROOT = files("astroagent.agent.prompts")
   _ENV = Environment(loader=FileSystemLoader(_PROMPT_ROOT))

   def _build_system_prompt() -> str:
       return (
           _PROMPT_ROOT.joinpath("system_core.txt").read_text()
           + "\n\n"
           + _PROMPT_ROOT.joinpath("system_tool_guide.txt").read_text()
           + "\n\n"
           + _render_active_rules()
       )

   def build_fit_control_messages(record, ...):
       return [
           LLMMessage("system", _build_system_prompt()),  # 稳定前缀
           LLMMessage("user", _ENV.get_template("user_round.j2").render(record=record, ...)),
       ]
   ```

3. `OpenAICompatibleClient.complete`：传 `extra_headers={"prompt-cache": "auto"}`（OpenAI 自动）；为 Anthropic 客户端预留 `cache_control` 注入点

#### 触发缓存的关键约束

- system 段拼接**顺序固定**、**内容稳定**
- 动态规则按当前 record 状态过滤后渲染——但**永远附加在 user message 里**，不污染 system 前缀
- 多模态图片**永远在最新一条 user message**，不进 system

#### 验收标准

- [ ] system 段长度稳定（多次调用 byte-level 相同 → MD5 一致）
- [ ] OpenAI 账单显示 `cached_input_tokens > 0`，命中率 ≥ 80%
- [ ] 新增 `tests/test_prompt_rendering.py`：覆盖规则触发逻辑、模板渲染稳定性

#### 预计代码量

`+150 / -300`，1 天

---

### 3.3 A3：引入 `AgentSession` + 真正的 tool-use loop

#### 目标
- LLM 第一次以 `role=tool` 消息看到工具执行结果
- 跨轮上下文显式存于 `Session.history` 而非 record 字段
- Session 直接可序列化为 SFT 样本（messages + tool_calls + tool_results）
- `loop.py` 瘦身到 ~250 行

#### 实施清单

**新增文件**

```
src/astroagent/agent/session.py
src/astroagent/agent/tools/executor.py    # tool_call → 工具执行 → tool_result
```

`session.py` 核心 API：

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

@dataclass
class SessionMetrics:
    n_rounds: int = 0
    n_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    image_count: int = 0

@dataclass
class StepResult:
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]   # 含 tool_call_id
    refit_record: dict[str, Any] | None
    evaluation: dict[str, Any]
    decision: str                         # accepted / needs_human_review / rejected

@dataclass
class AgentSession:
    record: dict[str, Any]
    window: pd.DataFrame
    client: "LLMClient"
    output_dir: Path
    seed: int = 42
    history: list["LLMMessage"] = field(default_factory=list)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    _round_index: int = 0

    def build_messages(self) -> list["LLMMessage"]:
        """System prompt + history (含 tool_results) + 当前轮 user message。"""

    def step(self, *, temperature: float = 0.0) -> StepResult:
        """单轮：LLM call → 解析 tool_calls → 执行 → 把 tool/assistant 消息追加到 history。"""

    def commit_round(self, result: StepResult) -> None:
        """Gate accept 后把 candidate 的 record 提升为 mainline。"""

    def to_jsonl(self) -> str:
        """导出为 SFT 样本（每条交互一行 JSON，含 messages + ground_truth_decision）。"""

    def save_checkpoint(self, path: Path) -> None:
        """完整保存 session 状态（record + history + metrics + seed）。"""

    @classmethod
    def load_checkpoint(cls, path: Path, *, client: "LLMClient", window: pd.DataFrame) -> "AgentSession":
        """从 checkpoint 恢复，client/window 由调用方注入。"""
```

`tools/executor.py`：

```python
def execute_tool_call(
    call: dict, *, record: dict, window: pd.DataFrame
) -> dict:
    """根据 tool name 派发到 fit_control 现有的补丁应用逻辑，
    返回结构化结果，供作为 role=tool 消息回灌。"""
    name = call["function"]["name"]
    args = call["function"]["arguments"]  # 已被 provider 校验过
    if name == "request_refit":
        return _execute_refit(record, window, ...)
    # ... 其余工具：调用 fit_control 现有 _build_*_override 函数
    return {
        "tool_call_id": call["id"],
        "name": name,
        "ok": True,
        "result": {...},   # 摘要字段，**不要把整个 record 塞回**
    }
```

**修改文件**

`agent/loop.py`：从 576 行重构为薄编排层（~250 行）

```python
def run_fit_control_loop(
    *,
    record, window, client, output_dir,
    initial_plot_image_path=None,
    max_rounds: int = 3,
    temperature: float = 0.0,
    best_of_n: int = 1,            # NEW，A3 内不一定上，但接口先留
    seed: int = 42,
    sample_id_prefix=None,
    force: bool = False,
) -> FitControlLoopResult:
    session = AgentSession(record=deepcopy(record), window=window,
                           client=client, output_dir=Path(output_dir),
                           seed=seed)

    if not force and not _fit_control_needed(session.record):
        return _early_return(session, "not_needed_good_fit")

    for round_index in range(1, max_rounds + 1):
        result = session.step(temperature=temperature)
        if not result.tool_calls:
            return _finalize(session, "no_refit_requested")
        if result.decision == "rejected":
            session.rollback_round()
            if round_index == max_rounds:
                return _finalize(session, "refit_rejected")
            continue
        session.commit_round(result)
        if result.decision == "needs_human_review":
            return _finalize(session, "needs_human_review")
    return _finalize(session, "max_rounds_reached")
```

**关键设计要点**

1. **Tool result 摘要而非全量**：
   - 一次 refit 的 evaluation 可能上百 KB；回灌时只放 ≤ 2 KB 的关键字段（`fit_rms_delta`, `n_components`, `gate_decision`, top-3 warnings）
   - 完整 evaluation 仍写入 `outputs/<sample>/round{i}/evaluation.json` 供事后分析

2. **图片在 history 中的衰减**：
   - 最近 2 轮的 user message 含图
   - 更早的轮在重新拼接 messages 时把 image content 替换为 `[round N image: <path>]` 文字占位

3. **SFT 样本格式**（`to_jsonl()` 输出，每行一条）：
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [{"type":"text","text":"..."},{"type":"image_url",...}]},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "abc", "content": "{\"fit_rms_delta\": -0.02, ...}"},
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": null, "tool_calls": [...]}
  ],
  "metadata": {
    "session_id": "...", "sample_id": "...", "stop_reason": "...",
    "final_decision": "accepted", "seed": 42
  }
}
```
**这就是 A3 最大的产物——天然 SFT 训练数据格式**。

4. **Checkpoint 兼容性**：写入时附带 `schema_version=1`，加载时校验。

**新增测试**

`tests/test_session.py`：
- 单步 step 后 history 长度 +2（assistant + tool）
- `to_jsonl()` 输出可被 `json.loads` 全部行
- `save_checkpoint` → `load_checkpoint` 恢复后再跑一轮，结果与连续跑等价
- gate rejected 后 `rollback_round()` 不污染 mainline

**迁移策略**

- 阶段 1：保留 `record["fit_control_loop"]` 字段（向后兼容下游消费方），同时 session 有自己的 history
- 阶段 2（后续 PR）：迁移所有消费方到读 session checkpoint，删除 record 内的内嵌历史字段

#### 验收标准

- [ ] `loop.py` 行数 ≤ 280
- [ ] 一次完整 3-round loop 的 `to_jsonl()` 输出含 ≥ 6 条 messages（system+user+(assistant+tool+user) × 3）
- [ ] checkpoint 往返测试通过（行为一致）
- [ ] 既有 `test_fit_control_loop.py` 全绿（必要时做兼容性 shim）

#### 预计代码量

`+400 / -200`，2–3 天

---

## 4. 后续阶段（B 系列）摘要

| PR | 目标 | 依赖 |
|---|---|---|
| **B1** | Tool result 真正回灌（执行结果摘要 → role=tool 消息） | A3 |
| **B2** | 错误处理：HTTP 重试、JSON repair、checkpoint 自动保存 | A3 |
| **B3** | Best-of-N 采样 + 候选并发评分；偏好数据落盘 | A3 |
| **B4** | Async batch + token budget 管理 | A3 |

详见第 5 节"完整路线图"中的对应描述。

---

## 5. 完整路线图（含 Agent 层之外）

### 5.1 阶段 0：工程基建（1–2 天）

| 子项 | 改造 | 预计 LOC |
|---|---|---|
| 0.1 | 抽 `core/constants.py`，集中 `C_KMS`、km/s 阈值、gate thresholds | +80 / -30 |
| 0.2 | `core/logging.py`：`get_logger()` + CLI `--log-level` | +60 |
| 0.3 | 拆分 `voigt_fit.py` → `_voigt/{seeding,priors,lsq_init,nested,fit_runner}.py` | 重构，无逻辑改动 |
| 0.4 | Pydantic v2 替换手写 schema（先只动 `validate_fit_control`） | +120 / -40 |

### 5.2 阶段 1：复现性与成本治理（2–3 天）

| 子项 | 改造 |
|---|---|
| 1.1 | `core/random.py` `seed_all()`，CLI `--seed`，写入 `provenance` |
| 1.2 | `plot.py` 加 `compact=True`；`llm.py` 上传前 `PIL.thumbnail((1024,1024))` |
| 1.3 | LLM 客户端重试（指数退避）+ usage 累加 + Anthropic 客户端 |
| 1.4 | Prompt 模板外置（**与 A2 合并**） |

### 5.3 阶段 2：测试加固（2 天）

| 子项 | 改造 |
|---|---|
| 2.1 | `tests/golden/` 固化 review_record，pytest --update-golden flag |
| 2.2 | LLM `RecordingReplayClient`，离线回放真实 LLM 响应 |
| 2.3 | `pytest.mark.network`，DESI 测试默认跳过；本地 fixture 谱样本 |

### 5.4 阶段 3：数据闭环（1–2 周）

| 子项 | 改造 |
|---|---|
| 3.1 | `data/spectrum_io.py`：`SpectrumRecord` parquet 统一格式 |
| 3.2 | `astroagent-fetch-desi` CLI：批量下载 + 断点续传 + 并发 |
| 3.3 | `data/synthetic.py`：合成谱生成器（带真值） |
| 3.4 | 专家轨迹采集器（streamlit/ipywidgets）→ `data/expert_traces/*.jsonl` |

> **3.4 是后训练管线的命门**，越早越好。配合 A3 的 `Session.to_jsonl()`，可以双源并行采集训练数据。

### 5.5 阶段 4：评估与训练（4–8 周）

| 子项 | 改造 |
|---|---|
| 4.1 | `evals/` 模块：golden_recovery / decision_consistency / physics_score / leaderboard |
| 4.2 | SFT 管线：`training/sft/` + LoRA + Qwen2.5-VL-7B 起步 |
| 4.3 | RL（GRPO/DPO），奖励：`α·ΔBIC + β·physics - γ·n_tools - δ·rejected` |

### 5.6 阶段 5：部署与可观测（按需）

- `astroagent serve`：FastAPI + 速率限制 + API Key
- Trace 持久化：langfuse 或自建 SQLite

---

## 6. 验收与度量

每个 PR 必须给出**量化指标**，以下是关键 KPI：

| 指标 | Baseline | 目标 |
|---|---|---|
| 单条 fit_control 调用 prompt token | ~2 000 (uncached) | ≤ 1 600 |
| Prompt cache 命中率 | 0 % | ≥ 80 % |
| 单条 record 处理总成本（含图） | ~$0.05–0.10 | ≤ $0.025 |
| `loop.py` 行数 | 576 | ≤ 280 |
| `voigt_fit.py` 单文件行数 | 1 596 | ≤ 500（拆分后每个文件） |
| 测试覆盖率（核心路径） | ~70%（估） | ≥ 85% |
| Checkpoint 恢复后行为一致 | N/A | 通过自动测试 |
| SFT 样本可用性 | 无 | `Session.to_jsonl()` 直接产出 |

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| A3 重构破坏现有下游消费 | 阶段 1 保留 `record["fit_control_loop"]`，shim 维持向后兼容 |
| Prompt cache 不生效 | 加自动测试断言 system MD5 稳定；监控 OpenAI usage 字段 |
| 工具 schema 与 patch 层耦合不一致 | `tools/registry.py` 与 `fit_control._normalize_patch_call` 共用同一组常量 |
| LLM 在 tool-use 模式下行为漂移 | 准备 recording 测试集，A1 落地后立即跑 N=20 离线回归 |
| 后训练数据格式提早定型不利于迭代 | `to_jsonl()` 输出附 `schema_version` 字段，未来不破坏 |

---

## 8. 推荐执行顺序

```
阶段 0.1–0.2 (constants + logging)         ── PR-001  (必先)
        ↓
阶段 0.4 (Pydantic) + A1 (tool calling)    ── PR-002  (合并提交，schema 一致)
        ↓
A2 (prompt 三段式 + caching)               ── PR-003
        ↓
阶段 0.3 (拆 voigt_fit)                     ── PR-004  (独立，与 agent 层无耦合)
        ↓
A3 (AgentSession + tool loop)              ── PR-005  (Agent 层重头戏)
        ↓
阶段 1.1–1.3 (seed, image, retry)          ── PR-006
        ↓
阶段 2.1 (golden tests)                     ── PR-007  (此后所有数值变更需更新 golden)
        ↓
B1 / B2 / B3 / B4                          ── 并行
        ↓
阶段 3 (数据闭环) + 阶段 4 (训练)          ── 长期
```

---

## 9. 附录：当前问题快速索引

| 想了解 | 看哪一节 |
|---|---|
| 为什么说"不是真正的 tool loop" | §1.3 问题 1 |
| 为什么 prompt cache 命中率为 0 | §1.3 问题 3 |
| Session 抽象长什么样 | §3.3 |
| SFT 样本怎么产出 | §3.3 关键设计要点 3 |
| 训练管线什么时候开始铺 | §5.4–5.5 |
| 一个 PR 完成后怎么验收 | §6 |

---

*本文档随项目演进持续更新，建议在每个 PR 合入后回到对应章节标记完成状态。
