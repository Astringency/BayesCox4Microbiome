# 实验脚本

从项目根目录执行命令。`JULIA_BIN` 可指定 Julia 可执行文件路径；未设置时使用
`julia +1.11.7`。`DRY_RUN=true` 仅打印 Shell 入口的执行命令。

| 用途 | 入口 |
| --- | --- |
| 模拟实验、基础网格 | `run_simulation.jl`、`run_simulation.sh` |
| Horseshoe/RHS、prox、Bayesian LASSO 系列默认配置 | `run_simulation_hspriors.sh`、`run_simulation_proxpriors.sh`、`run_simulation_blassopriors.sh` |
| 实际数据、T1D review、LOO、CV 重算 | `analyze_real_data.jl`、`analyze_real_data.sh` |
| 模拟汇总、进度监控 | `run_summary.sh`、`run_monitor.sh` 及相应 Julia 文件 |
| 诊断、实际数据汇总 | `run_diagnose.jl`、`summarize_revision_diagnostics.jl`、`summarize_realdata_*.jl` |
| 绘图和 PH 检验 | `plot_simulation.sh`、`plot_*.R`、`check_t1d_ph.R` |
| 数据生成、删失校准 | `generate_simulation_data.jl`、`calibrate_censoring.jl` |
| 补充模拟实验 | [review/README.md](review/README.md) |
| 共用执行函数及实际数据内部实现 | `lib/` |

`analysis_common.jl`、`tune_hyperparams_cindex.jl`、`nested_cindex_evaluation.jl`、
`loo_evaluation.jl`、`validation_common.jl`、`simulation_scenarios.jl` 和
`simulation_coverage.jl` 是入口依赖的实现文件。

## 基础网格

24 个设置是 `n={200,400} × p={50,150} × nonzero={6,12} ×`
`baseline={weibull,exponential,gompertz}`，不包括先验/约束组合和重复次数。
三个 `run_simulation_*priors.sh` 各自保存参数和循环，默认仅选择 Weibull 和 Gompertz，
即每个方法/约束的 16 个数据设置。`run_simulation.sh` 默认是单组试验：
`n=200, p=150, nonzero=12, Weibull, prox-sslasso, prox`，重复 1 次。
完整 24 设置可显式设置 `BASELINES=weibull,exponential,gompertz`。

```bash
# 预览 Cox-RHS 两种约束；不运行采样。
DRY_RUN=true bash scripts/run_simulation_hspriors.sh

# 预览 Prox-SSLASSO。参数显式给出，不代表该方案已在全部设置中通过收敛诊断。
DRY_RUN=true PRIORS=prox-sslasso USE_CAR=false INIT=rand \
  SAMPLER=hmc HMC_N_LEAPFROG=300 PROX_SUM0_LAMBDA=1e-3 PROX_L1_LAMBDA=1e-6 \
  bash scripts/run_simulation_proxpriors.sh
```

可配置 `N_VALUES`、`P_VALUES`、`NONZERO_VALUES`、`BASELINES`、`CONSTRAINTS`、
`PRIORS`、`REPS`、`SAMPLES`、`WARMUP`、`CHAINS`、`WORKERS`、`OUT_ROOT`。
默认 `WORKERS=1`；每个任务默认 4 链。额外 `--参数 值` 传入 Julia，后面的值优先。
输出默认使用新的时间戳目录；续跑须指定同一个 `OUT_ROOT`，并保持实验设置一致。
高维网格仍需先解决收敛问题，整理脚本不会自动启动或恢复它。

### 运行进度和日志

各模拟 Shell 入口在终端显示每组设置的 `已完成/总数`、百分比、等待/运行/完成/失败
状态，以及全部组的总进度。计数单位为一次拟合，即一个重复中的一种先验；
例如 50 次重复 × 2 种先验为 100 次拟合，4 条链合计为一次拟合。
续跑时，核验通过且成功复用的结果也计入完成数；失败会保留实际完成数并以非零状态退出。

采样进度条、标准输出和错误输出写入 `OUT_ROOT/logs/组名_时间戳_PID.log`，
不直接打印到终端。同名 `.log.progress.tsv` 记录完成数与总数，供脚本读取。
`LOG_ROOT` 可单独指定日志目录，`PROGRESS_INTERVAL` 控制刷新间隔（默认 2 秒）。
设置 `WORKERS=2` 可同时运行两组；每个运行脚本均直接调用 Julia，互不转调。
日志和进度的公共实现位于 `lib/`，模型参数仍保留在各运行脚本中。

## 实际数据

```bash
DRY_RUN=true RUN_DATASETS=T1D bash scripts/analyze_real_data.sh standard
DRY_RUN=true bash scripts/analyze_real_data.sh t1d-review
bash scripts/analyze_real_data.sh loo --help
```

`t1d-review` 默认包含 2 种先验 × 2 种约束 × 2 种协变量设置，共 8 组。
可用 `PRIORS`、`CONSTRAINTS`、`ADJUSTMENTS` 选择子集。
嵌套 CV 使用完整候选网格；默认内层/外层每链 800 预热和 800 保留，最终拟合
每链 2000 预热和 5000 保留，HMC 100 步、目标接受率 0.9。指定 `--stage prepare`
只生成配置和指纹，不采样。`--source-arm DIR` 操作一个已有分组。

`--stage 1` 至 `5` 分配外层折，`--stage full_tuning` 分配全数据内层调参，
`--config-ids 1,2` 支持互不重叠的候选分区；完成后 `--stage all` 会复用有效缓存、
汇总外层结果并拟合最终模型。保持候选集合、数据和设置不变才可复用缓存。
CV 指纹也包含 Julia 代码路径和内容；此次目录重构会改变指纹。
整理前的任务若需原样续跑，应使用其冻结代码快照；新入口使用新目录，
不会修改旧清单来绕过指纹检查。

```bash
# 长预算拟合使用新目录，保留原始结果与已选超参数。
bash scripts/analyze_real_data.sh t1d-review \
  --retry-from OLD_ARM --source-arm NEW_ARM --stage final \
  --final-samples 20000 --final-warmup 5000
```

普通完整 review 入口在最终拟合后运行 R 的 PH 诊断；`RUN_PH=false` 可跳过。
直接分阶段运行时，可对最终产出的 `ph_design.csv`、`ph_coefficients.csv`
调用 `check_t1d_ph.R`。`convergence` 和 `cv-replay` 子命令保留实际数据的
收敛网格与保存设置重放功能。旧 `.ps1` 入口已被统一 LOO 入口替代。
普通数据分析、T1D review 和收敛网格的 Shell 入口会显示分组开始/完成/失败状态，
详细输出写入相应输出根目录的 `logs/`（可通过 `LOG_ROOT` 指定）。

## 汇总与监控

```bash
bash scripts/run_summary.sh --root results/simulation/YOUR_RUN
bash scripts/run_monitor.sh --root results/simulation/YOUR_RUN
bash scripts/plot_simulation.sh results/simulation/YOUR_RUN
```

Python 调度、诊断、历史修复和 review 报告工具保存在本地
`audit/script_cleanup_20261013/`，已被 `.gitignore` 排除。
这些归档保留原始部署记录；部分包含服务器旧路径，不是当前实验入口。
