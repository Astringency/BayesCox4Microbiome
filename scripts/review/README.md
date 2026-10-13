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

计划保留历史样本量、种子、先验、平滑参数和采样预算；它们是可复现的配置，
并不保证收敛。`weak_confirm_ssl.json` 包含 100000 保留样本的长预算设置，
运行前应先预览。所有失败结果保留，不自动挑选表现好的重复或再次重跑。

Shell 入口支持 `REVISION_OUT`、`REPS`、`CHAINS`、`SAMPLES`、`WARMUP`，
S4 还支持 `REP_IDS`、`GRAPHS`；可用互不重叠的 `REP_IDS` 和独立输出目录
划分任务。旧 Python 的机器专属排队、结果搬运和历史结果复用工具已归档。
当前 Julia 入口按数据和设置核验缓存；更改设置应使用新目录。
四个 Shell 入口与基础模拟共用分组及总进度显示，采样输出写入各实验输出根目录的
`logs/`，可用 `LOG_ROOT` 指定位置、`WORKERS` 控制并行组数。
敏感性实验中共享的 Cox-RHS 基线仍使用文件锁，避免同时写入同一结果目录。

BCCox 已移到 `BayesCox4micro_baseline` R 项目，并自带 50 组配对 S4 数据、
图和作者原始代码；从其 `simulation_bccox_survival_analysis.R` 运行。
