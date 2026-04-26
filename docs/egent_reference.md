# EGENT 参考架构与最小复现

这份文档用来说明参考代码 `Egent/` 到底在做什么、怎么最小复现、以及它和我们 BDMI 项目的区别。

`Egent/` 是参考文献 EGENT 的代码基线，不是本项目主体。

## 1. EGENT 做什么

EGENT 的任务是测量高分辨率恒星光谱中的 equivalent width，简称 EW。

输入：

- 已经在恒星静止系中的光谱 CSV：`wavelength`, `flux`, `flux_error`
- 目标谱线列表 CSV：`wavelength`

输出：

- 每条线的 EW 测量值
- EW 不确定度
- 诊断拟合图
- 包含拟合元信息的 JSON 结果

它的工作流比较保守：

1. 先跑确定性的 Voigt profile 拟合。
2. 如果拟合质量差、区域拥挤或残差异常，再把诊断图交给 LLM 检查。
3. LLM 决定接受、调整窗口、换连续谱方法、重新拟合或标记为不可靠。

这里 LLM 不是物理拟合器本身，而是包在传统拟合工具外面的“复核与决策层”。

## 2. 代码结构

```text
Egent/
├── README.md               # 上游说明文档
├── app.py                  # Streamlit 网页界面
├── config.py               # 后端、模型、API key、输出目录配置
├── ew_tools.py             # 光谱读取、连续谱、Voigt 拟合等核心工具
├── llm_client.py           # OpenAI-compatible API 客户端
├── llm_client_local.py     # 本地 MLX-VLM 客户端
├── run_ew.py               # 命令行主流程
├── example/
│   ├── spectrum.csv        # 示例光谱
│   └── linelist.csv        # 示例线表
└── tutorial.ipynb          # 教程 notebook
```

## 3. 主流程

命令行入口是 `run_ew.py`。

整体流程可以理解为：

```text
run_ew_analysis()
  load_spectrum()
  读取 line list
  对每一条目标线：
    process_line()
      direct_fit()
        load_spectrum()
        extract_region(window=3.0)
        set_continuum_method(iterative_linear)
        fit_ew()
        计算拟合质量指标
      如果 direct fit 质量足够好：
        保存结果和图
      否则：
        llm_measure_with_vision()
          get_fit_plot()
          把诊断图和提示词发给 LLM
          LLM 可以调用这些工具：
            extract_region()
            set_continuum_method()
            set_continuum_regions()
            fit_ew(additional_peaks=[...])
            flag_line()
            record_measurement()
        保存 LLM 复核后的结果和图
  写出 results_<timestamp>.json
```

## 4. 核心工具

核心工具定义在 `ew_tools.py`：

- `load_spectrum(spectrum_file)`：读取并检查光谱。
- `extract_region(line_wavelength, window)`：截取目标线附近的局域窗口。
- `set_continuum_method(method, order, sigma_clip, top_percentile)`：设置连续谱拟合方法。
- `set_continuum_regions(regions)`：手动指定连续谱区域。
- `fit_ew(additional_peaks=None)`：拟合“多项式连续谱 + 多个 Voigt 线型”。
- `get_fit_plot()`：生成给 LLM 看用的诊断图。
- `flag_line(line_wavelength, reason)`：标记某条线不可靠。
- `record_measurement(...)`：记录最终 EW 测量。

一个很重要的限制：

EGENT 目前没有显式的 `set_analysis_mask()` 工具。也就是说，LLM 不能直接说“把某个污染区间 mask 掉，不参与拟合”。它只能通过缩小窗口、换连续谱方法、指定连续谱区域、添加额外峰或直接 flag 来间接改善结果。

这正好是我们项目的动机之一：在 quasar 吸收谱任务里，`analysis_mask_intervals` 应该成为模型的一级输出。

## 5. Direct Fitting 细节

对每条目标线，EGENT 会：

1. 截取局域窗口，默认大约是 `±3 Å`。
2. 用迭代 sigma clipping 等方法估计连续谱。
3. 用连续谱归一化 flux。
4. 在 `1 - normalized_flux` 中找吸收峰。
5. 拟合一个模型：
   - 多项式连续谱项
   - 每个检测峰对应一个 Voigt 分量
6. 选择距离目标波长最近的拟合分量作为目标线。
7. 对目标 Voigt profile 积分得到 EW。
8. 计算拟合诊断指标：
   - 归一化残差 RMS
   - 目标附近 central RMS
   - reduced chi-square
   - residual slope
   - 拟合线数

如果 direct fit 质量差、chi-square 太高、区域太拥挤、central RMS 太高，或者残差斜率暗示连续谱问题，就会进入 LLM 复核。

## 6. 最小复现：只跑 direct fit

进入参考代码目录：

```bash
cd Egent
```

如果直接跑完整示例线表：

```bash
python3 run_ew.py \
  --spectrum example/spectrum.csv \
  --lines example/linelist.csv \
  --workers 1 \
  --output-dir /tmp/egent_direct_check
```

注意：如果环境里配置了 API key，质量差的线会触发 LLM。为了只测试 direct fit，可以选一条 direct fit 会直接接受的线：

```bash
cat > /tmp/one_line.csv <<'EOF'
wavelength
6127.91
EOF

python3 run_ew.py \
  --spectrum example/spectrum.csv \
  --lines /tmp/one_line.csv \
  --workers 1 \
  --output-dir /tmp/egent_smoke
```

我们本地实测结果：

```text
6127.91 Å | EW = 45.3 ± 2.1 mÅ
used_llm = false
```

输出文件：

```text
/tmp/egent_smoke/output/results_<timestamp>.json
/tmp/egent_smoke/output/fits/direct/6127.91.png
```

## 7. 最小复现：用 CherryIN GPT-5.4 跑 LLM 复核

不要把 key 写进仓库，用临时环境变量：

```bash
export OPENAI_API_BASE="https://open.cherryin.net/v1"
export EGENT_MODEL="openai/gpt-5.4"
export OPENAI_API_KEY="<your CherryIN key>"
```

选择一条 direct fit 明显很差的线：

```bash
cat > /tmp/one_llm_line.csv <<'EOF'
wavelength
6157.72
EOF

python3 run_ew.py \
  --spectrum example/spectrum.csv \
  --lines /tmp/one_llm_line.csv \
  --workers 1 \
  --output-dir /tmp/egent_cherry_gpt54_refit_6157
```

我们本地实测结果：

```text
Direct fit:
  EW = 61.08 ± 3.09 mÅ
  RMS = 5.79
  reduced chi-square = 42.77
  quality = poor

LLM-guided refit:
  EW = 54.70 ± 2.21 mÅ
  RMS = 1.36
  reduced chi-square = 2.45
  quality = good
```

这次 LLM 的有效改进来自：

- 把局域窗口从 `±3 Å` 缩到 `±2 Å`
- 把连续谱方法改成 `iterative_poly`
- 拟合分量从 7 个减少到 5 个

它不是通过显式 analysis mask 改进的，因为 EGENT 当前没有这个工具。

## 8. 最小架构伪代码

如果只想理解 EGENT 最小实现，可以压缩成下面这个流程：

```python
for target_wave in line_list:
    region = extract_region(spectrum, target_wave, window=3.0)
    continuum = fit_continuum(region)
    normalized = region.flux / continuum
    peaks = detect_absorption_peaks(normalized)
    fit = fit_multi_voigt(normalized, peaks)
    metrics = compute_fit_metrics(fit)

    if metrics.good_enough:
        result = accept(fit.target_line)
    else:
        plot = make_diagnostic_plot(region, fit, metrics)
        llm_action = ask_llm(plot, metrics, allowed_tools)
        result = execute_llm_tool_loop(llm_action)

    save_result(result)
    save_plot(result)
```

最小确定性部分：

- 读取光谱
- 截取局域窗口
- 拟合连续谱
- 检测吸收峰
- 多 Voigt 分量拟合
- EW 积分
- 质量指标计算
- 诊断图生成

最小 LLM 部分：

- 提示词：要求只关注目标线附近，不要被窗口边缘干扰
- 工具 schema
- 工具调用执行循环
- 支持图像输入
- 记录最终结果

## 9. 我们实测发现的限制

- 当前本地环境没装 `streamlit`，所以 `app.py` 暂时不能直接启动。
- DeepSeek 文本 API 能通，但我们测试的接口不接受 EGENT 使用的 `image_url` 图像消息格式。
- CherryIN `openai/gpt-5.4` 能跑通文本 ping 和 EGENT LLM 视觉复核。
- CherryIN / 模型服务偶尔会慢或返回服务端错误，需要重试。
- 最终 JSON 没保存完整 LLM conversation，不方便复盘完整 tool call。
- 最终 JSON 保存了 iteration RMS，但没保存 iteration reduced chi-square。
- 没有显式 analysis mask 工具。

## 10. 对我们项目的启发

EGENT 对我们有价值，是因为它证明了：LLM 包在传统拟合工具外面，确实可以改善一些边界拟合案例。

但我们的 quasar absorption 项目要做几件不同的事：

- 用已知 `line_id` 和 `z_sys` 自动切局域窗口
- 显式预测 analysis mask
- 显式预测 continuum anchors
- 在速度空间里复核拟合动作
- 用 SFT 和 RL 训练较小的 student model
- 把 GPT-5.4 当 teacher / upper reference，而不是最终部署模型
- 用 schema 指标、规则 validator 和小规模 gold/QC 集评估
