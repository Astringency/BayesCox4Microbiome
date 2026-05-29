@echo off
setlocal
cd /d "%~dp0\.."

if not defined GKSwstype set "GKSwstype=100"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%i"

set test_n=2000
set test_p=150
set test_nonzero=6
set test_baseline=exponential

set /a test_p_plus_1=test_p+1

REM explicit,turing 
REM soft_center,prox,none
REM hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso,prox-lasso,prox-sslasso
REM weibull,exponential,gompertz

julia "+1.11.7" -t 4 --project=. scripts\run_simulation.jl ^
    --outdir "results/simulation/test_n%test_n%_p%test_p%_non%test_nonzero%_%test_baseline%_%timestamp%" ^
    --reps 1 ^
    --n "%test_n%" ^
    --p "%test_p%" ^
    --nonzero "%test_nonzero%" ^
    --baseline "%test_baseline%" ^
    --priors prox-lasso ^
    --sampler hmc ^
    --nuts-backend explicit ^
    --hmc-n-leapfrog 20 ^
    --samples 5000 ^
    --warmup 5000 ^
    --thin 1 ^
    --chains 4 ^
    --multichains true ^
    --target-accept 0.95 ^
    --noncentered true ^
    --seed 1 ^
    --use-car true ^
    --progress true ^
    --car-weight-quantile 0.99 ^
    --car-precision 1 ^
    --sample-car-strength true ^
    --car-strength-prior-a 1 ^
    --car-strength-prior-b 1 ^
    --car-distance "bray" ^
    --p0 truth ^
    --p0-alpha 0.05 ^
    --constraint prox ^
    --prox-sum0-lambda 1e-3 ^
    --prox-l1-lambda 1e-3 ^
    --prox-root-tol 1e-10 ^
    --prox-root-maxiter 100 ^
    --mala-step-size 1e-6 ^
    --mala-target-accept 0.574 ^
    --mala-adapt-every 50 ^
    --mala-min-step 1e-10 ^
    --mala-max-step 1e-2 ^
    --diagnose true ^
    --debug false ^
    --diagnose-format "pdf" ^
    --diagnose-top-k-beta 10 ^
    --diagnose-max-lag 50 ^
    --resume true ^
    --pseudocount 0.0 ^
    --standardize false ^
    --plot-first-rep-only true ^
    --sample-thin 10 ^
    --first-rep-sample-thin 1 ^
    --save-wide-csv true ^
    --save-jld2 false ^
    --alpha-prior-shape %test_p_plus_1% ^
    --alpha-prior-rate 1 ^
    --rhs-slab-df 4 ^
    --rhs-slab-s2 2 ^
    --hs-tau-scale 1 ^
    --hs-local-scale 1 ^
    --rhs-tau-scale 1 ^
    --rhs-local-scale 1 ^
    --rhs-pseudo-sigma-df 3 ^
    --rhs-pseudo-sigma-scale 10 ^
    --lasso-lambda2-shape 1 ^
    --lasso-lambda2-rate 0.1 ^
    --sslasso-spike-scale 0.04 ^
    --sslasso-slab-scale 0.5 ^
    --sslasso-theta-a 1 ^
    --sslasso-theta-b 1 ^
