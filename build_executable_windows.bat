@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
  goto run_build
)

where python >nul 2>nul
if errorlevel 1 (
  echo 未找到可用的 python。
  echo 请先安装 Windows Python，或在项目目录创建 .venv。
  exit /b 1
)

set "PYTHON_EXE=python"

:run_build
set "MOTION_AI_HOME=%SCRIPT_DIR%"
"%PYTHON_EXE%" src\build_executable.py

endlocal
