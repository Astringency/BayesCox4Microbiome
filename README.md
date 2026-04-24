# Bayesian Cox Proportional Hazards Model for Analyzing Microbiome Data

## Version and Package Management

```{julia}
versioninfo()
```

```{julia}
using Pkg

Pkg.activate(pwd())
Pkg.instantiate()
Pkg.status()
```

## Paper Priors

The package provides `cox_paper_prior(name; ...)` for the eight prior specifications
used in `docs/cox4microbiome_detail.pdf`:

- `hs`: horseshoe with `tau ~ Cauchy+(0, 1)`.
- `rhs`: regularized horseshoe using the Eq. (6) `tau0` scale with `pseudo_sigma^2 ~ half-t_3(0, 10)`, where `10` is the Student-t scale parameter on the variance-scale auxiliary quantity.
- `new-rhs`: regularized horseshoe using Eq. (6) with `pseudo_sigma = 1`.
- `est-rhs`: regularized horseshoe using Eq. (6) with `pseudo_sigma^2 = 1 / muhat`;
  `muhat` is computed from a preliminary `Survival.coxph` fit as `mean(exp(X * beta_hat))`.
- `cox-hs`: horseshoe using the Cox-specific Eq. (7) `tau0`.
- `cox-rhs`: regularized horseshoe using the Cox-specific Eq. (7) `tau0`.
- `lasso`: Bayesian LASSO with `lambda2 ~ Gamma(1, 0.1)`.
- `sslasso`: spike-and-slab LASSO with simulation defaults `s0 = 0.04`, `s1 = 0.5`;
  the real-data helper uses `s0 = 1`, `s1 = 10`.
  The Turing/NUTS branch uses a normal-exponential scale-mixture form for the
  Laplace spike/slab components. A dedicated `sampler=gibbs` is available for
  SSLASSO and updates `theta`, latent `gamma`, and local variances in Gibbs
  steps with a block MH update for `beta`.

Example:

```julia
prior = cox_paper_prior("cox-rhs"; p0 = 6, X = X, time = time, event = event)
fit = sample_bayes_cox(X, time, event; prior = prior, sampler = :nuts, use_car = true)
```

## Constraint And MALA Options

`sample_bayes_cox` accepts `constraint = :soft_center` (default), `:prox`, or
`:none`. The older `soft_center = true/false` keyword is still accepted and is
mapped to `:soft_center` or `:none`.

Examples:

```julia
# HS + MALA + soft-center
fit_hs_mala = sample_bayes_cox(
    X, time, event;
    prior = HorseshoePrior(p0 = 6),
    sampler = :mala,
    constraint = :soft_center,
)

# HS + MALA + Moreau-Yosida zero-sum penalty
fit_hs_prox = sample_bayes_cox(
    X, time, event;
    prior = HorseshoePrior(p0 = 6),
    sampler = :mala,
    constraint = :prox,
    prox_sum0_lambda = 1e-3,
)

# Bayesian LASSO Proximal MCMC using the l1 epigraph formulation
fit_lasso_prox = sample_bayes_cox(
    X, time, event;
    prior = BayesianLassoPrior(),
    sampler = :mala,
    constraint = :prox,
    prox_l1_lambda = 1e-3,
)

# SSLASSO Proximal MCMC. This is conditional on latent gamma indicators and
# uses the weighted l1 epigraph, not the marginal mixture-Laplace Gibbs model.
fit_ssl_prox = sample_bayes_cox(
    X, time, event;
    prior = SpikeSlabLassoPrior(),
    sampler = :mala,
    constraint = :prox,
)
```

If `p0` is omitted or set to `nothing`, HS/RHS-style priors estimate it by
fitting one univariate Cox PH model per covariate with `Survival.jl`; `p0` is
the number of covariates with Wald p-value below `p0_alpha`, clamped to
`1:(p - 1)`.

## Simulation And Real-Data Scripts

Simulation smoke test:

```powershell
julia --project=. scripts\run_simulation.jl --sampler mh --samples 50 --warmup 20 --reps 1 --n 200 --p 50 --nonzero 6 --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso --baseline weibull
```

Use `--p0 6` or `--p0 12` to force a prior nonzero count. If `--p0` is omitted
or set to `auto`, the scripts estimate it by univariate Cox screening. In
simulation runs, `--p0 truth` uses the known `--nonzero` value. Use
`--p0-alpha` to change the screening threshold.

Paper-scale simulation grid:

```powershell
julia --project=. scripts\run_simulation.jl --sampler nuts --samples 5000 --warmup 2000 --chains 4 --target-accept 0.99 --reps 50 --n 200,400 --p 50,100,150 --nonzero 6,12 --baseline weibull,exponential,gompertz --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso --use-car true
```

T1D analysis:

```powershell
julia --project=. scripts\analyze_real_data.jl --input data\T1D.csv --sampler nuts --samples 5000 --warmup 4000 --chains 4 --target-accept 0.99 --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso --use-car true
```

Outputs are written under `results/` as CSV summaries plus dependency-light
PDF plots, with EPS copies saved by the built-in vector writer. For quick
debugging use `--sampler mh`; for SSLASSO Gibbs use `--sampler gibbs --priors sslasso`;
for the paper-style workflow use `--sampler nuts`.
The Eq. (6) RHS scale can be numerically demanding under NUTS, so use a conservative
`--target-accept` and adequate warmup for full runs.

NUTS uses threaded parallel chains when `--chains` is greater than one. Start
Julia with threads enabled, for example:

```powershell
$env:JULIA_NUM_THREADS = "4"
julia --project=. scripts\run_simulation.jl --sampler nuts --samples 50 --warmup 50 --chains 4 --target-accept 0.99 --reps 1 --n 80 --p 20 --nonzero 6 --priors hs --baseline weibull --use-car true --p0 truth
```

For Revise-based script development, set `USE_REVISE=true`; the scripts will
use `Revise.includet` for `scripts/analysis_common.jl` when Revise is available.
The script metrics CSV files include sampling diagnostics:
`warmup_seconds`, `sample_seconds`, `total_seconds`, `allocated_mb`,
`gc_seconds`, and `gc_percent`. For NUTS, warmup/sample are split using the
first kept sample callback; for MH only total timing and allocation are recorded.

scripts/run_simulation.jl 当前支持这些参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --outdir | results/simulation | 输出目录 |
| --reps | 1 | 重复模拟次数 |
| --n | 200 | 样本量；支持逗号列表，如 200,400 |
| --p | 50 | 协变量/菌群数；支持逗号列表，如 50,100,150 |
| --nonzero | 6 | 真实非零系数数；支持 6 或 12，也支持逗号列表 |
| --baseline | weibull | 基线风险；支持 weibull,exponential,gompertz |
| --priors | 全 8 种 | 逗号列表：hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso |
| --sampler | nuts | nuts、mh，或 SSLASSO 专用 gibbs |
| --samples | 500 | 后验采样数 |
| --warmup | 200 | warmup 数 |
| --chains | 4 | NUTS 链数；MH 不使用多链 |
| --target-accept | 0.8 | NUTS 目标接受率 |
| --seed | 1 | 随机种子起点 |
| --use-car | true | 是否对 HS/RHS 系列启用 CAR |
| --car-distance | bray | CAR 距离：bray, jaccard, jsd |
| --p0 | auto | auto/不写表示单变量 Cox 筛选；也可给数值；simulation 额外支持 truth |
| --p0-alpha | 0.05 | 单变量 Cox 筛选显著性阈值 |

scripts/analyze_real_data.jl 当前支持这些参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| --input | data/T1D.csv | 输入 CSV |
| --outdir | results/realdata/T1D | 输出目录 |
| --time-col | time | 生存时间列名 |
| --status-col | status | 事件状态列名，需为 0/1 |
| --exclude-cols | Sex,Antibiotics | 从 taxa 矩阵中排除的非菌群列 |
| --priors | 全 8 种 | hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso |
| --sampler | nuts | nuts、mh，或 SSLASSO 专用 gibbs |
| --samples | 500 | 后验采样数 |
| --warmup | 200 | warmup 数 |
| --chains | 4 | NUTS 链数；MH 不使用多链 |
| --target-accept | 0.8 | NUTS 目标接受率 |
| --seed | 1 | 随机种子 |
| --use-car | true | 是否对 HS/RHS 系列启用 CAR |
| --car-distance | bray | CAR 距离：bray, jaccard, jsd |
| --p0 | auto | auto/不写表示单变量 Cox 筛选；也可给数值；不支持 truth |
| --p0-alpha | 0.05 | 单变量 Cox 筛选显著性阈值 |

布尔参数如 --use-car 支持 true, 1, yes, y，其他值会被解析为 false。

## Program Structure

### `priors`

1. **`abstract.jl`**：
    * 定义了一个抽象基础类型 `AbstractCoxPrior`。
    * 所有的先验类型都继承自这个抽象类型，这样可以在主程序中统一接口进行调用。

2. **`horseshoe.jl` (马蹄形先验)**：
    * 定义了 `HorseshoePrior` 结构体。
    * **作用**：标准马蹄形先验（Horseshoe Prior）是一种著名的全局-局部收缩先验，对大信号不进行过度压缩，同时能强力压缩噪声（零或接近零的系数）。
    * **参数**：包含全局收缩参数的尺度 `tau_scale`，是否使用特定于 Cox 模型的 tau0 计算的开关 `use_cox_tau0`，期望的非零特征数 `p0`，以及局部尺度参数 `local_scale`。

3. **`regularized_horseshoe.jl` (正则化马蹄形先验)**：
    * 定义了 `RegularizedHorseshoePrior` 结构体。
    * **作用**：为了解决标准马蹄形先验在处理极大系数时可能导致的采样不稳定问题而引入的正则化版本。这也是该项目推荐的默认先验。
    * **参数**：除了基础马蹄先验的参数外，还增加了控制“厚尾（Slab）”部分的自由度参数 `slab_df` 和尺度参数 `slab_s2`，用于限制极大系数的范围。

4. **`bayesian_lasso.jl` (贝叶斯 Lasso 先验)**：
    * 定义了 `BayesianLassoPrior` 结构体。
    * **作用**：Lasso 回归的贝叶斯等价形式。它基于拉普拉斯分布（Laplace distribution），倾向于产生稀疏的系数估计。
    * **参数**：可以通过 `lambda` 直接指定惩罚参数，或者将其设为 `nothing` 以便让模型自行推断，同时提供了用于推断的超先验参数 `lambda2_shape` 和 `lambda2_rate`。

5. **`spike_slab_lasso.jl` (尖峰厚尾 Lasso 先验)**：
    * 定义了 `SpikeSlabLassoPrior` 结构体。
    * **作用**：结合了尖峰厚尾（Spike-and-Slab）和 Lasso 的特点，常用于更严格的变量选择。它假设变量有两种状态：一种是被排除在模型之外（受极强的“尖峰”先验控制，系数趋于0），另一种是包含在模型内（受较宽的“厚尾”先验控制）。
    * **参数**：包含用于压缩零系数的极小尺度 `spike_scale`，用于非零系数的较大尺度 `slab_scale`，以及控制变量被纳入模型概率的 Beta 分布超参数 `theta_a` 和 `theta_b`。

### `mainfuncs`

1. 基础数据结构与准备 (Data & Types)
    * **`types.jl`**：定义了项目运行所需的底层数据结构。例如，用于存储预处理后数据的 `CoxPreparedData`、用于风险集计算的 `CoxRiskSet`，以及用于保存采样结果的 `MHSamples` 和 `CoxSamplerResult`。
    * **`data.jl`**：实现了 `prepare_cox_data` 函数。它负责接收原始的协变量（微生物数据）、生存时间和事件，进行诸如添加伪计数、闭包化（转换为相对丰度）、对数转换和标准化等预处理步骤。

2. Cox 模型似然计算 (Likelihood)
    * **`cox_likelihood.jl`**：实现了 Cox 比例风险模型的核心数学计算。包含了计算 Cox 偏对数似然（`cox_logpartiallikelihood`）以及处理事件发生似然（`cox_loglik_events`）的底层逻辑。

3. 具体先验的底层数学实现 (Prior Methods)
    这部分文件与 `priors` 文件夹里的结构体一一对应，专门用于处理不同先验在 MCMC 采样过程中的特定数学推导和参数更新逻辑：
    * **`prior_common.jl`**：多个先验可能共用的一些基础数学函数。
    * **`horseshoe_methods.jl`**：马蹄先验的具体计算实现。
    * **`regularized_horseshoe_methods.jl`**：正则化马蹄先验的具体计算实现。
    * **`bayesian_lasso_methods.jl`**：贝叶斯 Lasso 先验的具体计算实现。
    * **`spike_slab_lasso_methods.jl`**：尖峰厚尾 Lasso 先验的具体计算实现。

4. 采样器后端 (Samplers)
    * **`turing_sampler.jl`**：对接强大且高效的外部包 `Turing.jl`。当用户在调用主函数时选择 `sampler = :nuts`（默认值）时，就会由这个文件中的逻辑构建概率模型并执行 NUTS（No-U-Turn Sampler）采样。
    * **`mh_sampler.jl`**：一个轻量级的后备采样方案。当用户选择 `sampler = :mh` 时，将使用项目自己实现的自适应随机游走 Metropolis-Hastings 算法进行采样。
    * **`gibbs_sampler.jl`**：SSLASSO 专用的分块 Gibbs / Metropolis-within-Gibbs 采样器。

5. 后验推断与模型评估 (Posterior & API)
    * **`posterior.jl`**：处理从采样器获得的原始结果，提取后验分布信息。
    * **`api.jl`**：这是向普通用户暴露的**顶层接口大汇总**。它将上面的所有底层功能包装成了用户友好的函数，例如：
        * `sample_bayes_cox`：执行数据预处理并启动 MCMC 采样的统一主入口。
        * `posterior_beta_summary`：计算并返回回归系数（贝塔值）的均值、中位数和 95% 置信区间。
        * `sslasso_inclusion_probabilities`：计算特征的纳入概率（专用于 Spike-and-Slab 先验）。
        * `concordance_index`：计算模型的一致性指数（C-index），用于评价预测性能。

