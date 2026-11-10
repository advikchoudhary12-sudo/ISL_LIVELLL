@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%windows\isl-launcher.ps1" %*
if errorlevel 1 (
    echo.
    echo ISL Sign Recognition exited with an error.
    pause
)
