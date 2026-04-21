# Analysis Scripts

These Julia scripts reproduce the simulation and real-data analysis workflow described in
`docs/cox4microbiome.pdf` and `docs/cox4microbiome_detail.pdf`.

The scripts currently use the package's built-in MH sampler because the public
`posterior_beta_draws` helper extracts beta draws for `MHSamples`.

## Simulation

Quick smoke run:

```powershell
julia --project=. scripts\run_simulation.jl --samples 50 --warmup 20 --reps 1 --n 200 --p 50 --nonzero 6 --priors hs,cox-rhs --baseline weibull
```

Paper-scale grid:

```powershell
julia --project=. scripts\run_simulation.jl --samples 5000 --warmup 2000 --reps 50 --n 200,400 --p 50,100,150 --nonzero 6,12 --baseline weibull,exponential,gompertz --priors hs,cox-rhs,lasso,sslasso --use-car true
```

Outputs are written under `results/simulation/`:

- `tables/coef_*.csv`: posterior mean, median, 95% credible intervals, true beta
- `tables/simulation_metrics_long.csv`: MSE, LPML, WAIC, DIC, C-index
- `tables/simulation_false_discovery.csv`: FP, FN, TP, TN, FDR
- `tables/simulation_metrics_summary.csv`: replicate means and standard deviations
- `figures/coef_*.png`: coefficient interval plots
- `figures/simulation_metrics.png`: WAIC, LPML, C-index comparison

## Real Data

Default T1D run:

```powershell
julia --project=. scripts\analyze_real_data.jl --input data\T1D.csv --samples 5000 --warmup 4000 --priors hs,cox-rhs,lasso,sslasso --use-car true
```

For a different dataset, pass `--time-col`, `--status-col`, and optional non-taxa columns:

```powershell
julia --project=. scripts\analyze_real_data.jl --input path\to\data.csv --time-col time --status-col status --exclude-cols Sex,Antibiotics
```

Outputs are written under `results/realdata/T1D/` by default:

- `tables/coef_*.csv`: posterior summaries and selected taxa indicators
- `tables/model_metrics.csv`: LPML, WAIC, DIC, C-index
- `tables/taxa_columns.csv`: taxa columns included in the model
- `figures/coef_*.png`: coefficient interval plots
- `figures/model_metrics.png`: WAIC, LPML, C-index comparison
