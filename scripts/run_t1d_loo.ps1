param(
    [ValidateSet('subject_refit', 'fixed_risk', 'both')]
    [string]$LooMode = 'both',
    [ValidateSet('soft_center', 'prox')]
    [string]$CoxRhsConstraint = 'soft_center',
    [ValidateSet('soft_center', 'prox')]
    [string]$ProxSslassoConstraint = 'prox',
    [ValidateRange(1, 128)]
    [int]$Threads = 4,
    [string]$OutDir = '',
    [string]$Folds = 'all',
    [int]$Samples = 0,
    [int]$Warmup = -1,
    [int]$Chains = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot 'results/loo/T1D'
}
$juliaArgs = @(
    '+1.11.7', '--startup-file=no', "--threads=$Threads", "--project=$repoRoot",
    (Join-Path $PSScriptRoot 'rerun_t1d_loo.jl'),
    '--loo-mode', $LooMode,
    '--cox-rhs-constraint', $CoxRhsConstraint,
    '--prox-sslasso-constraint', $ProxSslassoConstraint,
    '--parallel', ($Threads -gt 1).ToString().ToLowerInvariant(),
    '--folds', $Folds, '--outdir', $OutDir, '--resume', 'true'
)
if ($Samples -gt 0) { $juliaArgs += @('--samples', "$Samples") }
if ($Warmup -ge 0) { $juliaArgs += @('--warmup', "$Warmup") }
if ($Chains -gt 0) { $juliaArgs += @('--chains', "$Chains") }
& julia @juliaArgs
if ($LASTEXITCODE -ne 0) { throw "T1D LOO failed (exit code $LASTEXITCODE). Check the fold files before resuming." }
