# 最小实现说明：人工审查包

这个版本的目标不是直接训练模型，而是先搭出一个可以审查、可以运行、可以逐步扩展的项目切片。

它回答三个问题：

- 已知 `line_id` 和 `z_sys` 时，代码能不能切出局域 quasar 吸收谱窗？
- 预期吸收体模式是否真的被局域谱窗支持，而不是看到一个噪声峰就接受？
- 工具层能不能给出一个非常朴素的 rule baseline？
- 结果能不能保存成人类能检查和修改的文件？

## 文件职责

- `configs/line_catalog.json`：谱线常量表。现在包含 C IV、H I Ly alpha、Mg II。
- `src/astroagent/line_catalog.py`：读取谱线表，只负责 `line_id -> rest wavelength`。
- `src/astroagent/review_packet.py`：核心纯函数，包括切窗、速度坐标、窗口摘要、rule baseline、写出审查包。
- `src/astroagent/cli/make_review_packet.py`：命令行入口，只负责参数解析和调用核心函数。
- `scripts/make_review_packet.py`：本地开发用包装脚本，不要求先安装包。
- `tests/test_review_packet.py`：最小单元测试，保证核心流程能跑通。

## 运行方式

从仓库根目录运行：

```bash
python3 scripts/make_review_packet.py
```

默认会生成一个合成的 C IV doublet quasar 吸收谱窗口，输出到：

```text
outputs/review_packet/
```

里面会有：

- `*.review.json`：结构化审查记录。
- `*.window.csv`：局域谱窗数组，包含 `wavelength / flux / ivar / pipeline_mask / velocity_kms`。
- `README.md`：输出文件说明。

也可以传入自己的 CSV：

```bash
python3 scripts/make_review_packet.py \
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
- `absorber_hypothesis_check`：预期线心附近是否有吸收谷，doublet 成员是否相互支持。
- `human_adjudication`：明确哪些科学裁决必须由人完成，例如 virial velocity 冲突或多元素不一致。
- `candidate_results`：保留所有候选测量结果；最小版本先为空，后续接拟合器后填充。
- `task_a_rule_suggestion`：规则方法建议的 mask 和 continuum anchors。
- `human_review`：留给人工填写接受、拒绝或修正。

这个格式故意很朴素。以后接入 DESI 真数据、GPT-5.4 teacher、SFT 数据导出时，都应该先保持这个审查包结构稳定。

## 为什么这样写

项目开发不要一开始写成巨大 pipeline。比较稳的顺序是：

1. 先有一个小输入。
2. 用纯函数做一个确定转换。
3. 写出人能检查的中间产物。
4. 加一个测试保证行为不乱变。
5. 再接真实数据和模型。

这就是这个最小实现的边界。
