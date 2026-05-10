$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$juliaArgs = @(
    "--project=.",
    "scripts\run_diagnose.jl"
) + $args

& julia @juliaArgs
exit $LASTEXITCODE
