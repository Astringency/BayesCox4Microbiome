# Review 补充实验

此目录保留 `results/review_20261012` 对应的补充模拟实验。
基础 24 设置用上一级 `run_simulation*`；T1D 用上一级 `analyze_real_data*`。

| 实验 | 入口 |
| --- | --- |
| 零和强度、L1 平滑、p0 敏感性 | `run_sensitivity_revision.sh lambda\|l1\|p0` |
| S1–S3 弱信号、相关零变量、计数测量情景 | `run_scenarios_revision.sh` |
| S4 Brownian 系统发育及 independent/correct/misspecified 图 | `run_car_revision.sh` |
| exact/none/soft_center/prox 零和约束比较 | `run_zero_sum_revision.sh` |
| 弱信号确认、高维平滑与 HMC 轨迹长度试验 | `run_followup.jl` + `plans/*.json` |
| 固定强度、适应 CAR 的诊断性图消融 | `probe_car_strength.jl`、`probe_adaptive_car.jl`，参数为计划 JSON 和 job 序号 |
| Prox 小规模收敛检查 | `probe_prox_convergence.jl` |
| 基线方法所需的配对数据导出 | `export_paired_baseline_cohorts.jl` |

```bash
DRY_RUN=true bash scripts/review/run_sensitivity_revision.sh l1
DRY_RUN=true bash scripts/review/run_scenarios_revision.sh
DRY_RUN=true bash scripts/review/run_car_revision.sh

julia +1.11.7 --project=. scripts/review/run_followup.jl \
  --plan scripts/review/plans/hmc_150_pilot.json \
  --outdir results/review_runs/hmc_150 --workers 1 --dry-run true
```

2026-10-13 起，本目录中除 L1 平滑敏感性分析外，固定的 `prox_l1_lambda` / `l1_lambda`
统一为 `1e-6`，包括 S1–S3、零和约束、零和强度、弱信号确认及后续采样试验。
Cox-RHS 入口的同名占位参数也统一为 `1e-6`，但该方法不使用 L1 平滑项。
`prox_sum0_lambda`（零和参数）与 L1 平滑参数不同，按各实验配置使用。

S1–S3 的后续运行统一使用以下配置：

| 方法 | 零和约束 | λΣ | λL1 |
| --- | --- | --- | --- |
| Cox-RHS | `soft_center` | 不使用 prox 零和参数 | 不使用 L1 平滑参数 |
| Prox-SSLASSO | `prox` | `1e-3` | `1e-6` |

两种方法默认均为 4 链，每链预热 5,000、保留 5,000，HMC leapfrog 300，目标接受率 0.8，
不抽稀。数据场景、50 次重复和其余先验保留。Shell 输出写入
`$REVISION_OUT/scenarios/revised_20261013/<S1|S2|S3>/<method>`，与旧结果分开。
`weak_confirm_rhs.json`、`weak_confirm_ssl.json` 及两种子试验 `weak_rhs_high_accept.json`
已同步这一新 S1 配置；文件名保留，但不再代表旧的长链或高接受率设置。
计划中的旧选择理由与种子分类标为 `historical_*`，不作为新配置已验证的证据。

p₀ 敏感性实验现使用 `soft_center` 零和约束，p₀ 仍为 `3,6,12`，其余先验与采样预算不变。
结果写入 `$REVISION_OUT/p0/soft_center/cox-rhs/<p0>`，包括 p₀=6 在内均独立运行，
不再复用 λΣ 敏感性实验的 prox 基线。已有 p₀ 敏感性结果使用 prox λΣ=`1e-3`，
不能改标为 soft_center；λΣ 与 L1 敏感性实验仍使用 prox。

L1 敏感性分析保留变化的取值：`run_sensitivity_revision.sh l1` 当前为
`1e-2,1e-3,1e-4,1e-5,1e-6`；`plans/highdim_l1_sensitivity.json` 保留 `1e-4` 与 `1e-6`
的对照。`probe_prox_convergence.jl` 的普通对照及非 L1 试验固定 `1e-6`，
显式的 `l1_1e3`、`l1_1e4`、`l1_1e6` 分支保留各自的敏感性设置。

这些是后续运行配置；结果目录中已有的 CSV、metadata、协议及冻结源码仍记录实际运行参数，
未随本次修改重写。特别是既有弱信号 SSL 长链验证使用 `1e-4`，其通过情况不能用于证明
新 `1e-6` 配置已收敛。需要复现历史结果时使用结果目录保存的原协议和冻结代码。
T1D 由内层 CV 选择的 L1 候选值属于真实数据调参，不是本目录固定参数实验。

除上述 p₀ 约束与 S1–S3 配置调整外，其余实验设置保留；这些配置并不保证收敛。
运行前应先预览。所有失败结果保留，不自动挑选表现好的重复或再次重跑。

Shell 入口支持 `REVISION_OUT`、`REPS`、`CHAINS`、`SAMPLES`、`WARMUP`，
S1–S3 还支持 `HMC_N_LEAPFROG`、`TARGET_ACCEPT` 覆盖默认采样配置；
S4 还支持 `REP_IDS`、`GRAPHS`；可用互不重叠的 `REP_IDS` 和独立输出目录
划分任务。旧 Python 的机器专属排队、结果搬运和历史结果复用工具已归档。
当前 Julia 入口按数据和设置核验缓存；更改设置应使用新目录。
四个 Shell 入口与基础模拟共用分组及总进度显示，采样输出写入各实验输出根目录的
`logs/`，可用 `LOG_ROOT` 指定位置、`WORKERS` 控制并行组数。
Cox-RHS 的 prox 基线仍使用文件锁，避免同时写入同一结果目录；p₀ 的 soft_center 结果单独保存。

BCCox 已移到 `BayesCox4micro_baseline` R 项目，并自带 50 组配对 S4 数据、
图和作者原始代码；从其 `simulation_bccox_survival_analysis.R` 运行。
