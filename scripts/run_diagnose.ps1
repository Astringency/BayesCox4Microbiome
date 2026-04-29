param(
    [string]$Input = "results/runs/example",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir

julia --project="$RepoDir" (Join-Path $ScriptDir "run_diagnose.jl") --input "$Input" @RemainingArgs
