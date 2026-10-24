# Bayesian Variable Selection for Cox Proportional Hazards Models with Microbiome Compositional Data

## 环境

- Julia 1.11。
- `Project.toml` 声明项目依赖和 Julia 版本要求；`Manifest.toml` 锁定具体依赖版本。
- 在仓库根目录用 Pkg 安装环境：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 文件结构

```text
.
├── Project.toml, Manifest.toml   # Julia 项目依赖与锁定版本
├── src/BayesCox4Micro.jl         # Julia 包入口，加载 main.jl
├── main.jl                       # 模块定义及核心实现加载入口
├── priors/                       # 基础先验类型：Lasso、Horseshoe、SSLASSO 等
├── mainfuncs/                    # 模型实现
│   ├── data/                     # 数据预处理
│   ├── likelihood/               # Cox 部分似然
│   ├── priors/                   # 先验计算与约束
│   ├── targets/                  # 后验目标分布
│   ├── samplers/                 # MCMC 采样器
│   ├── diagnostics/              # 收敛、采样和可视化诊断
│   ├── metrics/                  # 模型评估与模拟指标
│   ├── evidence/                 # 贝叶斯证据与 Bridge sampling
│   └── api.jl、posterior.jl、types.jl、progress.jl、tuning_grids.jl
│                                 # 公共接口、后验、数据类型、进度和调参网格
├── data/                         # T1D.csv 生存数据；T1D_metadata.csv 分类注释
├── scripts/                      # 模拟和实际数据分析入口
│   ├── run_simulation*.sh         # 模拟实验启动脚本
│   ├── analyze_real_data.sh       # T1D 实际数据分析入口
│   ├── lib/simulation/            # 模拟数据生成、拟合和汇总
│   ├── lib/realdata/              # T1D 分析流程
│   ├── lib/diagnostics/           # PH 等诊断分析
│   ├── lib/plotting/              # 结果绘图
│   └── review/                    # 补充实验脚本
└── test/                         # Julia 与 R 测试
```

本地工作区还可能包含被 `.gitignore` 排除的目录：`results/` 保存运行输出，`docs/` 保存参考论文和 R 代码，`audit/` 保存历史材料，`check/` 保存临时检查文件。

## 用法示例

### 模拟实验

生成一组 Weibull 基线风险数据，并拟合 Cox-RHS 与 Prox-SSLASSO：

```bash
julia -t 4 --project=. scripts/lib/simulation/run_simulation.jl \
  --outdir results/simulation/example --reps 1 --n 200 --p 50 --nonzero 6 \
  --baseline weibull --priors cox-rhs,prox-sslasso \
  --sampler nuts --nuts-backend explicit --samples 1000 --warmup 500 \
  --chains 4 --multichains true --p0 truth --use-car false
```

### 实际数据分析

运行内置 T1D 数据的完整分析流程（嵌套交叉验证、最终拟合和 PH 诊断）：

```bash
JULIA_BIN=julia bash scripts/analyze_real_data.sh \
  --outdir results/realdata/T1D/example
```

实际数据的 Schoenfeld 诊断还需要安装 R 和 R 包 `survival`。更多实验入口和参数见 [`scripts/README.md`](scripts/README.md)。
