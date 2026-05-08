# 数据源说明：DESI quasar 吸收谱

本文档记录本项目优先考虑的数据来源。所有数据路线都限定在 **quasar 光谱 / quasar sightline** 上：我们研究的是背景 quasar 光经过中间吸收系统后留下的吸收特征，不把普通恒星光谱、星系发射线光谱或银河系恒星丰度目录混入训练主线。

## 1. 数据边界

本项目的输入对象是 DESI quasar spectra，基础字段为：

- `TARGETID`
- `Z_QSO`
- `wavelength`
- `flux`
- `ivar`
- `pipeline_mask`

样本构造时，`line_id` 与 `z_sys` 已知。工具层根据静止波长和系统红移切出局域谱窗，LLM 只在局域窗口和工具摘要上做结构化决策。

数据产品需要分成三类：

- 吸收体 catalog：提供正样本、红移、EW、NHI、SNR、拟合摘要等弱标签起点。
- quasar catalog：提供 quasar redshift、BAL flag、HEALPixel、parent sample 和按 object 划分数据集所需元数据。
- 光谱数组：提供真正送入模型的 `wavelength / flux / ivar / mask`。

## 2. MVP：C IV doublet

优先数据源：

- 官方文档：[CIV Absorber Catalog from DESI DR1 Quasars](https://data.desi.lbl.gov/doc/releases/dr1/vac/civ-absorber/)
- 数据目录：[DESI DR1 C IV absorber VAC](https://data.desi.lbl.gov/public/dr1/vac/dr1/civ-absorber/v1.0/)

适合作为 MVP 的原因：

- 明确来自 DESI DR1 quasar spectra。
- C IV 是双线，静止波长约为 1548 A 和 1550 A。
- catalog 提供 `Z_ABS`、EW、EW error、double Gaussian fit、SNR、速度弥散、column density 等字段。
- 正样本可以由 `TARGETID + Z_ABS` 唯一定位吸收系统。
- parent QSO catalog 可以用来构造负样本和按 quasar object 做 train / valid / test 划分。

初始使用文件：

- `CIV-Absorbers-dr1-v1.0.fits`
- `CIV-Absorbers-parent-QSO-dr1-v1.0.fits`

初始用途：

- 正样本：按 `TARGETID + Z_ABS` 切 C IV doublet 局域窗口。
- 负样本：从 parent QSO 中避开已知 absorber redshift 后采样。
- hard cases：优先抽取低 SNR、饱和、VDISP 异常、多 absorber 同 sightline 样本。

## 3. H I 路线：Ly alpha / DLA

H I 不是普通意义上“找一条氢线”这么简单。本项目里的 H I 默认指 **quasar 背景光中的 Ly alpha / DLA 吸收系统**。

优先数据源：

- 官方文档：[Damped Lya Catalog from the DLA Toolkit](https://data.desi.lbl.gov/doc/releases/dr1/vac/dla-toolkit/)
- 数据目录：[DESI DR1 DLA Toolkit VAC](https://data.desi.lbl.gov/public/dr1/vac/dr1/dla-toolkit/v2.0/)

适合作为 H I 扩展的原因：

- 明确来自 DESI Y1 quasar spectra。
- 目标物理量与 H I 直接相关，核心字段包括 `Z_DLA`、`NHI`、`NHI_ERR`、`SNR_FOREST`、`SNR_REDSIDE`、`DELTACHI2`。
- 官方推荐质量筛选：`SNR_REDSIDE > 2`、`NHI > 20.3`、`DELTACHI2 > 0.3`，并排除 BAL quasars。
- DLA 有 damping wing，适合扩展到更难的 continuum 与拟合复核任务。

初始使用文件：

- `dlacat-dlatoolkit-dr1-main-dark-v2.0.fits`

初始用途：

- H I 正样本：按 `TARGETID + Z_DLA` 切 Ly alpha / DLA 局域窗口。
- 质量筛选：先保守使用官方推荐 cut，降低错误标签污染。
- 评测样本：保留一批低 SNR、BAL 附近、NHI 边界样本做人审或 teacher/QC。

可选备份数据源：

- 官方文档：[Damped Lyman-alpha object catalog: CNN + GP](https://data.desi.lbl.gov/doc/releases/dr1/vac/dla-cnn-gp/)
- 数据目录：[DESI DR1 DLA CNN+GP VAC](https://data.desi.lbl.gov/public/dr1/vac/dr1/dla-cnn-gp/v1.0/)
- 文件：`dla_catalog_cnngp_combine_dr1.fits`

该目录适合作为交叉验证或 hard case 来源，不建议一开始取代 DLA Toolkit 主线。

## 4. Mg II 路线：低风险扩展 / 备选 MVP

Mg II 不是 H I，但对项目很有用：它也是 quasar sightline 上的 doublet absorption，样本量大，适合作为 C IV 之外的稳健扩展。

数据源：

- 官方文档：[Mg II Absorber Catalog](https://data.desi.lbl.gov/doc/releases/dr1/vac/mgii-absorber/)
- 数据目录：[DESI DR1 Mg II absorber VAC](https://data.desi.lbl.gov/public/dr1/vac/dr1/mgii-absorber/v1.0/)
- 文件：`MgII-Absorbers-DR1.fits`

适合使用的原因：

- 明确来自 DESI QSO spectra。
- 文件约 65 MB，catalog 规模大，包含约 270k 个 Mg II absorber entries。
- doublet 静止波长约为 2796 A 和 2803 A，物理约束清楚。
- 字段包括 `TARGETID`、`Z_QSO`、`Z_MGII`、`EW_2796`、`EW_2803`、EW errors、Gaussian/MCMC 参数、continuum method、`LINE_SNR_MIN/MAX`。

当前定位：

- 作为 C IV 之后的稳健扩展。
- 验证 schema 是否能跨离子泛化。
- 给后续 reward 和评测提供波长区域不同但结构相似的样本。

## 5. zlya：quasar redshift / BAL 辅助表

数据源：

- 官方文档：[Lya Working Group Quasar Redshift Catalog (zlya)](https://data.desi.lbl.gov/doc/releases/dr1/vac/zlya/)
- 数据目录：[DESI DR1 zlya VAC](https://data.desi.lbl.gov/public/dr1/vac/dr1/zlya/v1.0/)
- 文件：`qso_cat_dr1_main_dark_healpix_zlya-v0.fits`

它不是吸收体 catalog，但对我们很关键：

- 提供 DESI DR1 `z > 1.6` main dark quasar redshift。
- 提供 BAL 相关字段，例如 `BI_CIV`、`AI_CIV`、`NCIV_2000`、`VMIN_CIV_2000`、`VMAX_CIV_2000`、`ZMASK`。
- DLA Toolkit 官方建议用 zlya 排除 BAL quasars。
- 可用于构造 parent quasar pool、BAL hard cases、以及按 `TARGETID` join DLA/C IV/Mg II catalog。

当前定位：

- 辅助表，不直接产生局域吸收线正样本。
- 下载优先级高，因为 BAL 排除和 quasar-level split 都需要它。

## 6. Ly alpha deltas：背景场 / continuum 参考，不是吸收体标签

数据源：

- 官方文档：[Lyman-alpha Year 1 Deltas Catalog](https://data.desi.lbl.gov/doc/releases/dr1/vac/lya-deltas/)
- 数据目录：[DESI DR1 lya-deltas VAC](https://data.desi.lbl.gov/public/dr1/vac/dr1/lya-deltas/v1.0/)

它包含 Ly alpha / Ly beta / CIII 区域的 flux transmission fluctuation、weights 和 continuum 相关数据。它适合：

- 理解 Ly alpha forest 的数据分布。
- 做 continuum / forest 相关辅助研究。
- 给 H I / DLA 样本构造提供背景统计。

它不适合：

- 直接当作单个 H I absorber 的标签表。
- 直接训练 `present / fit_ok / NHI` 这类局域吸收体任务。

## 7. 光谱获取

catalog 只给吸收系统和 quasar 元数据，不直接等价于模型输入。模型训练仍需要取回对应 quasar 的原始光谱数组：

- `wavelength`
- `flux`
- `ivar`
- `mask`

推荐通过 SPARCL 或 DESI 光谱文件路径按 `TARGETID` 拉取。后续数据构造脚本应只下载/缓存小批量样本，先验证 100 到 1000 个局域窗口的完整流程，不一开始搬运全量数据。

可行路径：

- 小样本验证：用 SPARCL 按 `TARGETID` 拉取 HEALPixel coadd spectra。
- 可复现批处理：用 catalog 里的 `HPXPIXEL` 定位 DESI DR1 healpix spectra 文件。
- 大规模下载：只在流程稳定后再考虑 Globus 或 NERSC/AWS。

## 8. 当前取舍

第一阶段以 C IV doublet 为 MVP，因为它的双线约束更容易形成可程序化 validator。H I / DLA 作为第二阶段扩展，文档和 schema 设计时提前兼容：

- `line_id = HI_LYA`
- `rest_wavelength_A = 1215.6701`
- `z_sys = Z_DLA`
- `present`
- `fit_ok`
- `NHI`
- `quality_flag`

这能保证项目主线始终是 quasar absorption spectra，同时给后续 H I 科学目标留下空间。

推荐优先级：

1. `CIV-Absorbers-dr1-v1.0.fits` + parent QSO：第一阶段 MVP。
2. `dlacat-dlatoolkit-dr1-main-dark-v2.0.fits`：H I / DLA 第二阶段。
3. `qso_cat_dr1_main_dark_healpix_zlya-v0.fits`：BAL 排除、redshift 和 quasar-level split。
4. `MgII-Absorbers-DR1.fits`：doublet 泛化与扩展。
5. `lya-deltas`：背景/continuum 研究，不作为初始标签源。
