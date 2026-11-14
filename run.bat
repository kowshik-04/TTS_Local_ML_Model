@echo off
setlocal
cd /d "%~dp0"

where pwsh.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
