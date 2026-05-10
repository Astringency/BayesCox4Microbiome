$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $env:GKSwstype) {
    $env:GKSwstype = "100"
}

$juliaArgs = @(
    "--project=.",
    "scripts\run_simulation.jl",
    "--sampler", "mh",
    "--samples", "50",
    "--warmup", "20",
    "--reps", "1",
    "--n", "200",
    "--p", "50",
    "--nonzero", "6",
    "--baseline", "weibull",
    "--priors", "hs,cox-rhs,lasso",
    "--p0", "truth"
) + $args

& julia @juliaArgs
exit $LASTEXITCODE
