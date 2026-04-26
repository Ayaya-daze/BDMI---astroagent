# SRT code 工具评估

来源目录：`/Users/mac/Desktop/srt/code`

本文档只记录可复用性判断，不把外部代码直接搬进本项目。`astro` 大作业仓库仍保持独立、干净。

## 1. 总体结论

`srt/code` 里最有用的是两套本地 Python 包：

- `astro`：天文基础工具，包含原子线表、速度/红移换算、丰度、统计和绘图。
- `ABSpec`：吸收线光谱分析工具，包含光谱片段、连续谱、LSF、Voigt profile 和 `vpfit` 拟合。

还有一个脚本：

- `auto_absorption_pipeline.py`：自动吸收线拟合流程示例，已经包含离子选择、局部吸收深度打分、速度成分初猜、ABSpec 拟合和 JSON 输出。

当前建议：**先复用思想和小函数，不直接引入整套 ABSpec 作为项目依赖。**  
原因是 ABSpec 依赖较重，包含 `astropy / lmfit / emcee / corner / astroquery / colossus` 等；而我们的最小版本现在只需要稳定生成人工审查包。

## 2. 最适合马上用上的部分

### 2.1 原子线表：`astro.atomic`

关键函数：

- `extract_line(label)`
- `extract_ion(ion)`
- `extract_zlines(z, wlmin, wlmax, elems=None, fthre=0)`
- `decomp_ion(ion)`
- `decomp_line(line)`

用途：

- 替代我们手写的 `configs/line_catalog.json`，或至少用它校验 C IV / Mg II / H I 的 rest wavelength、oscillator strength、gamma。
- 自动判断某个红移下哪些谱线落入当前观测波段。
- 给多元素交叉验证提供统一线表。

已验证：

```text
atomic.extract_line("CIV1548")
-> wave=1548.187, f=0.19, gamma=0.00324, elem=C, state=IV
```

注意：

- 它的 label 风格是 `CIV1548`，我们当前 schema 是 `CIV_1548` / `CIV_doublet`。需要写一层映射，不要直接改 schema。

### 2.2 速度/红移换算：`astro.basic`

关键函数：

- `z2v(z)`
- `v2z(v)`
- `vzz(z1, z2)`
- `sub_v(V1, V2)`

用途：

- 用相对论速度公式替换或校验我们当前的近似 `velocity_kms`。
- 计算 absorber redshift 和 host/system redshift 之间的速度差。
- 后续做 `velocity_exceeds_virial_expectation` flag 时会用到。

### 2.3 局部吸收深度打分：`auto_absorption_pipeline.py`

关键函数：

- `local_absorption_score(wave, flux, w0, dv_window=350)`
- `auto_pick_ions(...)`
- `find_velocity_components(...)`
- `build_fit_plan(...)`

用途：

- 很适合接到我们现在的 `absorber_hypothesis_check` 后面。
- 可以把“线心附近是否有吸收谷”升级成“哪些离子值得拟合、速度成分初猜在哪里”。
- 输出可以进入我们保留的 `candidate_results`。

建议最先移植的是：

- `local_absorption_score`
- `find_velocity_components`

这两个函数依赖少，容易审查，和当前最小实现的风格一致。

## 3. 第二阶段可用的部分

### 3.1 光谱片段与等效宽度：`ABSpec.spectrum.segment`

可用功能：

- `extract_zline(z, line, vmax=...)`
- `measure(...)`
- `extract_seg(...)`

用途：

- 后续接真实光谱后，可以对局部窗口测 EW、W90、中心波长。
- 可以帮助人工审查包多输出一层 candidate measurement。

暂不直接接入原因：

- ABSpec 的对象是 dict 子类，字段约定是 `wave / flux / error / mask / continuum / nflux`，和我们当前 `wavelength / flux / ivar / pipeline_mask` 需要适配。
- 它更偏 COS/STIS 使用习惯；DESI 光谱接入前要先做字段映射。

### 3.2 Voigt profile：`ABSpec.voigt`

关键函数：

- `line_shape(label, logN, b, v=0, c=1, dv=None)`
- `line_shape_wl(label, logN, b, v, c, z, wl)`
- `N2EW(N, line_label, b, vmax=1000, nbins=101)`

用途：

- 给候选拟合生成理论 profile。
- 做 toy model / synthetic sample。
- 在 review packet 里保存候选模型参数。

适合我们后续做：

- C IV doublet synthetic hard cases
- DLA / H I 简化 profile 演示
- RL reward 里的模型残差特征

### 3.3 `ABSpec.vpfit`

可用功能：

- `QSO_specs.extract_zlines(...)`
- `vpfit.add_ion(...)`
- `vpfit.add_continuum(...)`
- `vpfit.lmfit()`
- `vpfit.cal_BIC()`

用途：

- 真正做多离子、多成分 Voigt 拟合。
- 输出 column density、BIC、component velocity。

暂不作为 MVP 依赖原因：

- 依赖重。
- 需要规范 LSF、误差、连续谱和 mask。
- 自动拟合很容易给出“看似数值完整但物理不可信”的结果，必须先有人工审查包和 conflict flags。

## 4. 和我们项目的对应关系

| 我们的项目需求 | `srt/code` 可复用工具 | 建议 |
| --- | --- | --- |
| 谱线常量表 | `astro.atomic.extract_line` | 写 adapter 校验/生成 line catalog |
| velocity / redshift | `astro.basic.z2v/v2z/vzz` | 可直接参考或移植小函数 |
| 判断线心附近是否有吸收 | `local_absorption_score` | 可优先移植 |
| 初猜速度成分 | `find_velocity_components` | 可优先移植 |
| 多元素候选 | `ION_LINES`, `auto_pick_ions` | 改成 quasar/DESI 版本 |
| Voigt toy model | `ABSpec.voigt.line_shape` | 第二阶段用 |
| 多成分拟合 | `ABSpec.vpfit` | 暂缓，等 review packet 稳定后接 |
| 最终科学判断 | 无 | 仍然人工裁决 |

## 5. 推荐下一步

先做一个很小的 adapter，不引入重依赖：

1. 新增 `src/astroagent/srt_reference.py` 或 `src/astroagent/absorption_features.py`。
2. 从 `auto_absorption_pipeline.py` 里移植并重写两个纯函数：
   - `local_absorption_score`
   - `find_velocity_components`
3. 把输出接入 `absorber_hypothesis_check`：
   - 每条预期线的 `local_absorption_score`
   - 候选速度成分列表
   - `single_line_only` / `doublet_supported` flag
4. 暂不接 `ABSpec.vpfit`，等我们有真实 DESI 小样本和人工审查流程后再上。

这样可以让项目继续保持“人能审查的最小实现”，同时吸收已有代码里最有价值的经验。
