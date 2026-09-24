@echo off
rem Run bexa from any directory on Windows: puts the repo root on PYTHONPATH
rem and starts the command-line interface.
setlocal
set "BEXA_ROOT=%~dp0"
if "%PYTHONPATH%"=="" (
    set "PYTHONPATH=%BEXA_ROOT%"
) else (
    set "PYTHONPATH=%BEXA_ROOT%;%PYTHONPATH%"
)
python -m bexa %*
