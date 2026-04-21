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

5. 后验推断与模型评估 (Posterior & API)
    * **`posterior.jl`**：处理从采样器获得的原始结果，提取后验分布信息。
    * **`api.jl`**：这是向普通用户暴露的**顶层接口大汇总**。它将上面的所有底层功能包装成了用户友好的函数，例如：
        * `sample_bayes_cox`：执行数据预处理并启动 MCMC 采样的统一主入口。
        * `posterior_beta_summary`：计算并返回回归系数（贝塔值）的均值、中位数和 95% 置信区间。
        * `sslasso_inclusion_probabilities`：计算特征的纳入概率（专用于 Spike-and-Slab 先验）。
        * `concordance_index`：计算模型的一致性指数（C-index），用于评价预测性能。

