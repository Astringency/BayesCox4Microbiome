param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $env:GKSwstype) {
    $env:GKSwstype = "100"
}

$p = 150
$alphaPriorShape = $p + 1
$samplers = @("hmc", "mala")
$ns = @(200, 400, 1000, 2000, 3000)
$nonzeros = @(6, 12)
$baselines = @("weibull", "exponential", "gompertz")

foreach ($sampler in $samplers) {
    foreach ($n in $ns) {
        foreach ($nonzero in $nonzeros) {
            foreach ($baseline in $baselines) {
                $label = "rep1_prox-lasso_${baseline}_n${n}_p${p}_nz${nonzero}"
                $samplerOutdir = Join-Path $Root $sampler
                $completePath = Join-Path $samplerOutdir "runs\$label\complete.txt"
                if (Test-Path $completePath) {
                    Write-Host "[SKIP] $sampler $label"
                    continue
                }

                Write-Host "[RUN] $sampler $label"
                $juliaArgs = @(
                    "+1.11.7",
                    "-t", "4",
                    "--project=.",
                    "scripts\run_simulation.jl",
                    "--outdir", $samplerOutdir,
                    "--reps", "1",
                    "--n", "$n",
                    "--p", "$p",
                    "--nonzero", "$nonzero",
                    "--baseline", $baseline,
                    "--priors", "prox-lasso",
                    "--sampler", $sampler,
                    "--nuts-backend", "explicit",
                    "--hmc-n-leapfrog", "20",
                    "--samples", "5000",
                    "--warmup", "5000",
                    "--thin", "1",
                    "--chains", "4",
                    "--multichains", "true",
                    "--target-accept", "0.95",
                    "--noncentered", "true",
                    "--seed", "1",
                    "--use-car", "true",
                    "--progress", "false",
                    "--car-weight-quantile", "0.99",
                    "--car-precision", "1",
                    "--sample-car-strength", "true",
                    "--car-strength-prior-a", "1",
                    "--car-strength-prior-b", "1",
                    "--car-distance", "bray",
                    "--p0", "truth",
                    "--p0-alpha", "0.05",
                    "--constraint", "prox",
                    "--prox-sum0-lambda", "1e-3",
                    "--prox-l1-lambda", "1e-3",
                    "--prox-root-tol", "1e-10",
                    "--prox-root-maxiter", "100",
                    "--mala-step-size", "1e-6",
                    "--mala-target-accept", "0.574",
                    "--mala-adapt-every", "50",
                    "--mala-min-step", "1e-10",
                    "--mala-max-step", "1e-2",
                    "--diagnose", "true",
                    "--debug", "false",
                    "--diagnose-format", "pdf",
                    "--diagnose-top-k-beta", "10",
                    "--diagnose-max-lag", "50",
                    "--resume", "true",
                    "--pseudocount", "0.0",
                    "--standardize", "false",
                    "--plot-first-rep-only", "true",
                    "--sample-thin", "10",
                    "--first-rep-sample-thin", "1",
                    "--save-wide-csv", "true",
                    "--save-jld2", "false",
                    "--alpha-prior-shape", "$alphaPriorShape",
                    "--alpha-prior-rate", "1",
                    "--rhs-slab-df", "4",
                    "--rhs-slab-s2", "2",
                    "--hs-tau-scale", "1",
                    "--hs-local-scale", "1",
                    "--rhs-tau-scale", "1",
                    "--rhs-local-scale", "1",
                    "--rhs-pseudo-sigma-df", "3",
                    "--rhs-pseudo-sigma-scale", "10",
                    "--lasso-lambda2-shape", "1",
                    "--lasso-lambda2-rate", "0.1",
                    "--sslasso-spike-scale", "0.04",
                    "--sslasso-slab-scale", "0.5",
                    "--sslasso-theta-a", "1",
                    "--sslasso-theta-b", "1"
                )
                & julia @juliaArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Julia simulation failed for $sampler $label with exit code $LASTEXITCODE"
                }
            }
        }
    }
}

& julia "+1.11.7" "--project=." "scripts\summarize_simulation_results.jl" "--root" $Root
if ($LASTEXITCODE -ne 0) {
    throw "summarize_simulation_results.jl failed with exit code $LASTEXITCODE"
}

& julia "+1.11.7" "--project=." "scripts\summarize_requested_grid.jl" "--root" $Root
if ($LASTEXITCODE -ne 0) {
    throw "summarize_requested_grid.jl failed with exit code $LASTEXITCODE"
}
