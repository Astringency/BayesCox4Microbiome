@echo off
setlocal
cd /d "%~dp0\.."

julia --project=. scripts\run_diagnose.jl %*

exit /b %ERRORLEVEL%
