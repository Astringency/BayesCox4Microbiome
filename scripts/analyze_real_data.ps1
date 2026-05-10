$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $env:GKSwstype) {
    $env:GKSwstype = "100"
}

$juliaArgs = @(
    "--project=.",
    "scripts\analyze_real_data.jl",
    "--input", "data\T1D.csv",
    "--time-col", "time",
    "--status-col", "status",
    "--exclude-cols", "Sex,Antibiotics",
    "--sampler", "nuts",
    "--samples", "500",
    "--warmup", "200",
    "--chains", "4",
    "--priors", "hs,cox-rhs,lasso,sslasso",
    "--p0", "auto"
) + $args

& julia @juliaArgs
exit $LASTEXITCODE
