@echo off
setlocal
cd /d "%~dp0\.."

cmake -S . -B build
if errorlevel 1 exit /b 1

cmake --build build --config Release
if errorlevel 1 exit /b 1

if exist build\Release\liplab_runner.exe (
    build\Release\liplab_runner.exe .
) else if exist build\liplab_runner.exe (
    build\liplab_runner.exe .
) else (
    echo liplab_runner.exe not found under build\Release or build\
    exit /b 1
)
if errorlevel 1 exit /b 1

python scripts\summarize.py
if errorlevel 1 exit /b 1

python scripts\check_grades.py
if errorlevel 1 exit /b 1

endlocal
