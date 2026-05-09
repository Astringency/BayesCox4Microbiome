$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:GKSwstype = "100"

$currentTime = Get-Date -Format "yyyyMMdd_HHmmss"

# TEST
$testN = 400
$testP = 150

julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl `
    --sampler ula `
    --samples 5000 `
    --warmup 2000 `
    --chains 4 `
    --multichains `
    --reps 1 `
    --n $testN `
    --p $testP `
    --nonzero 12 `
    --baseline weibull `
    --priors prox-lasso `
    --constraint prox `
    --target-accept 0.8 `
    --mala-step-size 1e-2 `
    --mala-target-accept 0.564 `
    --seed 1 `
    --use-car true `
    --car-distance bray `
    --p0 truth `
    --p0-alpha 0.05 `
    --outdir "results\simulation\test_n${testN}_p${testP}_non12_weibull_${currentTime}"

if ($LASTEXITCODE -ne 0) {
    Write-Error "SIMULATION FAILED!"
    exit $LASTEXITCODE
}

# MAIN LOOP
$nVals = @(200, 400)
$pVals = @(50, 150)
$nonVals = @(6, 12)
$baselineVals = @("weibull", "exponential", "gompertz")

foreach ($n in $nVals) {
    foreach ($p in $pVals) {
        foreach ($baseline in $baselineVals) {
            foreach ($nonzero in $nonVals) {
                Write-Host "[SIMULATION] n=$n, p=$p, $nonzero nonzero coefs with $baseline baseline."

                julia +1.11.7 -t 4 --project=. scripts\run_simulation.jl `
                    --sampler nuts `
                    --nuts-backend explicit `
                    --samples 5000 `
                    --warmup 4000 `
                    --chains 4 `
                    --reps 50 `
                    --n $n `
                    --p $p `
                    --nonzero $nonzero `
                    --baseline $baseline `
                    --constraint soft_center `
                    --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso `
                    --target-accept 0.9 `
                    --seed 1 `
                    --use-car true `
                    --car-distance bray `
                    --p0 truth `
                    --p0-alpha 0.05 `
                    --outdir "results\simulation\softcenter\n${n}_p${p}_nonzero${nonzero}_${baseline}"

                if ($LASTEXITCODE -ne 0) {
                    Write-Error "SIMULATION FAILED!"
                    exit $LASTEXITCODE
                }
            }
        }
    }
}

Write-Host "ALL SIMULATION DONE!"
