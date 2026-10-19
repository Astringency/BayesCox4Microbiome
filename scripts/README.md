# 实验入口

从项目根目录运行。`JULIA_BIN` 指定 Julia 可执行文件；默认使用 `julia +1.11.7`。
Shell 入口支持 `DRY_RUN=true` 预览命令。`WORKERS` 控制并行任务数，`CHAINS` 控制链数；
运行日志写入输出目录的 `logs/`。

实现统一放在 `scripts/lib/`：

| 目录 | 内容 |
| --- | --- |
| `plotting/` | Julia 和 R 的系数、模拟结果、方法比较和诊断图 |
| `simulation/` | 主实验与 S1–S4 数据生成、场景设置、删失校准、统一拟合、信息量校准、汇总和监控 |
| `realdata/` | T1D 完整分析、通用数据入口、LOO、CV 重放及结果汇总 |
| `diagnostics/` | 数值诊断、Schoenfeld 检查、时间效应模型和贝叶斯因子 |
| `analysis_common.jl`、`*.sh` | 共用参数、文件和执行函数 |

调参候选值在 [`../mainfuncs/tuning_grids.jl`](../mainfuncs/tuning_grids.jl) 中维护，
通过 `BayesCox4Micro` 导出的 `PRIOR_PARAM_GRID_VALUES` / `PRIOR_PARAM_GRIDS` 引用。
内层调参、外层评估和全数据调参使用同一网格。

## 主模拟实验

四个 `run_simulation*.sh` 文件保持原样，由共用 shell 执行函数调用
`lib/simulation/run_simulation.jl`。

```bash
DRY_RUN=true bash scripts/run_simulation.sh
DRY_RUN=true bash scripts/run_simulation_hspriors.sh
DRY_RUN=true bash scripts/run_simulation_proxpriors.sh
DRY_RUN=true bash scripts/run_simulation_blassopriors.sh
bash scripts/run_summary.sh --root results/simulation/YOUR_RUN
bash scripts/run_monitor.sh --root results/simulation/YOUR_RUN
bash scripts/plot_simulation.sh --root results/simulation/YOUR_RUN
```

24 个基础设置是 `n={200,400} × p={50,150} × nonzero={6,12} ×`
`baseline={weibull,exponential,gompertz}`。先验族 shell 默认只选择 Weibull/Gompertz，
`run_simulation.sh` 默认运行单组。保持各入口原有参数和默认值。
支持 `N_VALUES`、`P_VALUES`、`NONZERO_VALUES`、`BASELINES`、`PRIORS`、
`CONSTRAINTS`、`REPS`、`SAMPLES`、`WARMUP`、`OUT_ROOT`。

```bash
# 独立生成主实验或特殊场景数据，参数格式见相应 Julia 入口。
julia +1.11.7 --project=. scripts/lib/simulation/generate_data.jl --scenario S4 --outdir results/cohorts/S4
```

## 实际数据

```bash
# 默认 T1D：8 组完整静态分析，然后执行 PH 诊断并汇总全部模型的贝叶斯因子。
bash scripts/analyze_real_data.sh
# 仅预览；不采样。
DRY_RUN=true bash scripts/analyze_real_data.sh
# 选择一组；默认仍执行完整流程。
PRIORS=cox-rhs CONSTRAINTS=soft_center ADJUSTMENTS=adjusted bash scripts/analyze_real_data.sh
```

T1D 只有一个工作流 `t1d`（默认），包含 `cox-rhs,prox-sslasso` ×
`soft_center,prox` × `unadjusted,adjusted`。调整模型联合拟合 Sex 和 Antibiotics，
临床系数不参加成分变量的零和约束。

执行顺序：每个外层折的训练数据内层调参 → 外层测试折评估 → 全数据内层调参 →
最终全数据拟合 → Schoenfeld 图和时间效应模型 → 跨组贝叶斯因子。
默认静态模型每链预热 4000、保留 5000，4 链，HMC leapfrog 300、目标接受率 0.95。
`--samples`、`--warmup`、`--chains` 控制 CV 预算；`--final-samples`、
`--final-warmup` 可独立设置最终预算。默认 5 外层折、5 内层折。

输出根目录默认 `results/realdata/T1D/<时间戳>`，其下每个
`<constraint>/<adjustment>/<method>` 包含：

| 路径 | 内容 |
| --- | --- |
| `analysis_specification.json` | 数据、采样、协变量和实验设置 |
| `nested_cv/` | 内层候选结果、外层折拟合和外层测试评估 |
| `full_data_tuning/` | 全数据内层调参及选定参数 |
| `runs/`、`posterior_features.csv` | 最终静态后验和系数汇总 |
| `ph_diagnostics/` | Schoenfeld 图、时间效应后验、证据重复及 Bridge sampling 诊断 |
| `ph_comparison/` | 组内模型证据及所有有向贝叶斯因子比较 |

输出根目录还有 `ph_comparison/`，汇总所有组，包括跨方法、约束和临床调整设定的比较。

分阶段和候选分区仍可运行：

```bash
bash scripts/analyze_real_data.sh --stage prepare
bash scripts/analyze_real_data.sh --source-arm ARM --stage 1 --config-ids 1,2
bash scripts/analyze_real_data.sh --source-arm ARM --stage full_tuning
bash scripts/analyze_real_data.sh --source-arm ARM --stage all
bash scripts/analyze_real_data.sh --retry-from OLD_ARM --source-arm NEW_ARM --stage final \
  --final-samples 20000 --final-warmup 5000
```

`--stage all` 复用经指纹验证的已完成阶段；变更数据、代码或设置须使用新输出目录。
`--smoke true` 只保留一个候选，用于检查流程，不代表完整调参。
`RUN_PH=false` 或 `--run-ph false` 可显式跳过 PH。

其他实际数据和保存结果的工作流：

```bash
RUN_DATASETS=HCT_bcorsis bash scripts/analyze_real_data.sh standard
bash scripts/analyze_real_data.sh loo --help
bash scripts/analyze_real_data.sh cv-replay --help
```

通用 CSV 的 Julia 入口必须显式给出 `--workflow standard`，并指定 `--input`、
`--time-col`、`--status-col`、`--comp-cols` / `--exclude-cols`。

## PH 诊断

R 需要 `survival` 包；`RSCRIPT_BIN` 可指定 Rscript。
Schoenfeld 残差在静态联合后验均值处计算，保留 Efron 结点和完整风险集，
逐系数画残差曲线，同时绘制整体累计标准化得分过程及风险得分方向的残差图。
`cox.zph` 的整体和逐系数 p 值仅作探索性 plug-in 诊断，不能当成贝叶斯检验概率。

时间效应模型为 `β(t)=β+δ g(t)`，`g(t)` 为中心化、标准化的 log 事件时间。
提供仅成分变量、仅非成分变量及两者同时引入时间效应三种模型；无临床协变量的
未调整模型只有成分时间效应，其余两项在输出中记录为不适用或重复。
成分时间系数采用正交零和对比，临床时间系数独立，固定 Normal(0,0.5) 先验尺度。
所有时间效应模型仅在全数据拟合，沿用对应静态模型所选参数，不再交叉验证。

证据使用 Bridge sampling：分别估计后验核和含约束先验的归一化常数，再取对数差。
后验使用 HMC；RHS 先验使用独立拒绝抽样及条件高斯抽样，SSLASSO 先验使用
半径坐标下的椭圆切片、log 半径切片及指示变量 Gibbs 更新，保留完整 Jacobian。
SSLASSO 的坐标尺度同时包含半径宽度和 Moreau 平滑宽度，窄 spike/slab 下也保留精确 Jacobian。
这些先验抽样方法均保持原先验；Gaussian 只用于辅助抽样及 Bridge proposal。
同一静态模型的独立时间斜率先验积分为 1，因此各时间效应组合共用该静态先验的归一化常数。
每条链前半段拟合条件高斯坐标中的 proposal，后半段估计 bridge；MCSE 用 batch means 保留链内相关性。
重复 proposal 共用抽样结果，不能把重复间 SD 除以重复次数平方根当成总 MCSE。
检查 `bridge_diagnostics.csv` 的 overlap、MCSE、`numerical_screen`，以及 HMC 的 R-hat/ESS。
可用 `--ph-samples`、`--ph-prior-samples`、`--ph-warmup`、`--ph-chains`、`--ph-leapfrog`、
`--ph-target-accept`、`--ph-repetitions`、`--ph-time-prior-scale` 调整。
须检查混合与重复间稳定性；这些是固定已选超参数下的 Cox 部分似然贝叶斯因子，
不包含完整基线生存分布的证据，也不积分超参数选择的不确定性。

```bash
julia +1.11.7 --project=. scripts/lib/diagnostics/ph.jl --source-arm ARM
julia +1.11.7 --project=. scripts/lib/diagnostics/ph.jl --root ROOT --aggregate-only true
```

SSLASSO 的先验与后验可使用不同抽样长度。例如，在已经完成静态拟合的独立输出目录中：

```bash
julia +1.11.7 --project=. scripts/lib/diagnostics/ph.jl --source-arm ARM \
  --samples 10000 --prior-samples 60000 --warmup 6000 \
  --chains 4 --leapfrog 300 --target-accept 0.95 --repetitions 4
```

`diagnostic_table` 对保存的 `gamma[j]` 二元序列输出 `selection_mixing` 和
`pip_consistency`。后者包含各 taxon 的链间 PIP 极差，以及六个链对中最大的
`abs(PIP_c-PIP_d)/hypot(MCSE_c,MCSE_d)`（四链时）。链内 MCSE 考虑自相关；
常数链的 MCSE 记为缺失，主标准化指标要求所有链对可估计。
同时保留可估计链对数、常数链数及常数链是否互相冲突，不把未切换解释为零 Monte Carlo 误差。

## 补充实验和归档

当前补充实验仅保留 [review/README.md](review/README.md) 中的五类。
论文图表、服务器专用调度器、探查脚本、历史补丁及过时计划保存在本地
`audit/script_refactor_20261018/`，不再作为公开入口。
`audit/` 已由 `.gitignore` 排除。历史结果及其冻结代码不随本次重构改写。

T1D 的 Prox-SSLASSO 使用独立的 21 组网格：spike/slab 配对为
`(2,10), (2,20), (5,20), (5,50), (10,50), (0.04,0.5), (1,10)`，
每对配合 `prox_l1_lambda = 1e-5, 1e-4, 1e-3`；默认通用 57 组网格不变。
新输出的 `analysis_specification.json` 冻结完整网格，旧规格不能静默切换。

T1D 零和约束敏感性分析可在新输出目录指定 `--prox-sum0-lambda 1e-3`，
并用 `--prior-grid-version relaxed_five_plus_two_lambda1e6_v2` 将 `1e-6`
加入上述每一组 spike/slab 配对（共 28 组）。新增 7 组追加在原 21 组之后，
保留既有候选编号及其采样种子；默认 21 组网格和已有分析规格不变。
