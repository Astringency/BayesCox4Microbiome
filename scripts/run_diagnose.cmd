@echo off
setlocal
set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..

if "%~1"=="" (
  julia --project="%REPO_DIR%" "%SCRIPT_DIR%run_diagnose.jl"
) else (
  julia --project="%REPO_DIR%" "%SCRIPT_DIR%run_diagnose.jl" %*
)
