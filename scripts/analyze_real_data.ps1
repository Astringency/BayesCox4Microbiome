$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

julia --project=. scripts\analyze_real_data.jl `
  --input data\T1D.csv `
  --time-col time `
  --status-col status `
  --exclude-cols Sex,Antibiotics `
  --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso `
  --sampler nuts `
  --samples 5000 `
  --warmup 2000 `
  --chains 4 `
  --target-accept 0.8 `
  --seed 1 `
  --use-car true `
  --car-distance bray `
  --p0 auto `
  --p0-alpha 0.05 `
  --outdir results\realdata\T1D
