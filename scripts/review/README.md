# 补充实验

所有实验共用 `../lib/simulation/` 的数据生成、拟合、诊断和结果格式。
本目录仅保留五类实验的 shell 入口，不再有单独的 S4 或 T1D 补丁实现。

| 实验 | 命令 |
| --- | --- |
| 零和参数敏感性 | `bash scripts/review/run_sensitivity_revision.sh lambda` |
| Prox L1 平滑敏感性 | `bash scripts/review/run_sensitivity_revision.sh l1` |
| Cox-RHS p0 敏感性 | `bash scripts/review/run_sensitivity_revision.sh p0` |
| S1 弱信号、S2 相关零变量、S3 测序计数 | `bash scripts/review/run_scenarios_revision.sh` |
| S4 系统发育图比较 | `bash scripts/review/run_car_revision.sh` |
| 零和处理效果 | `bash scripts/review/run_zero_sum_revision.sh` |
| 训练数据零点信息量校准与配对拟合 | `bash scripts/review/run_scaling_revision.sh` |

均支持 `DRY_RUN=true`。`REVISION_OUT` 指定输出根目录，`REPS` 默认 50；
`REP_IDS=1,2` 可选重复子集，`CHAINS` 默认 4。
`WORKERS` 控制并行组数，采样日志写入输出目录的 `logs/`。
S1–S3 默认预热/保留均为 5000、HMC leapfrog 300；其他默认预热 2000、保留 5000、
leapfrog 100。非 L1 敏感性实验的 Prox-SSLASSO L1 平滑固定 `1e-6`。

零和参数默认 `1e-2,1e-3,1e-4,1e-5`；L1 默认 `1e-2,1e-3,1e-4,1e-5,1e-6`；
p0 默认 `3,6,12`（soft_center）。敏感性取值可通过 `VALUES` 覆盖。
S1–S3 默认比较 Cox-RHS soft_center 与 Prox-SSLASSO prox，两种方法均默认随机初始化（`init=rand`）。

S4 使用 Brownian 系统发育丰度模型；真实先验图只保留非零系数之间的树邻域边，
所有零系数节点均孤立，图版本为 `signal_only_v3`。四种设定分别为：

- `independent`：Cox-RHS 独立拟合，不启用 CAR。
- `correct`：Oracle，使用真实非零支持上的正确图。
- `misspecified`：固定叶标签置换后的错误图，保留边数、权重和谱。
- `abundance`：仅使用训练丰度，经原生 `car_weight_matrix` 学习边和权重；不使用生存结局、测试数据或真实支持。

四种设定均覆盖 `soft_center,prox`，默认 8 组且配对使用相同数据种子。
`GRAPHS`、`CONSTRAINTS` 可选择子集。生成树、协方差、真实图、错误图及每次实际拟合图均保存。
使用新目录 `S4/signal_only_v3`，旧图和新图结果不能混用。不保留删边消融或重复基线代码；
BcCox PH / BCCox PH 继续由各自的 R 项目维护。

零和处理默认比较 `exact,none,soft_center,prox(1e-3),prox(1e-6)`，两种方法共 10 组。
`CONSTRAINTS` 选择约束，`SUM0_LAMBDAS` 选择 prox 参数。
Prox-SSLASSO 默认随机初始化（`init=rand`），Cox-RHS 默认零初始化（`init=zero`）。
S1–S3 和零和处理实验均可用 `INIT` 覆盖初始化设置。

全局尺度校准先计算训练集 Cox 信息矩阵在 `β=0` 的对角元 `I_jj(0)`（支持 Efron ties），
求解 `sum_j τ sqrt(I_jj)/(1+τ sqrt(I_jj)) = p0`。相对于原 `cox_tau0` 的乘数写入
`calibration/*.json`，并记录训练数据 SHA256。参考与校准分支共用种子、生成器、CAR、
p0=6 和其他超参数；校准不读取真实系数、测试集或后验估计。
这是对角信息近似的配对敏感性分析，不替代完整的列标准化研究。

旧探查、论文绘图、专属服务器计划及补丁脚本已移入本地忽略目录
`audit/script_refactor_20261018/review/`。T1D 统一使用 `scripts/analyze_real_data.sh`。
