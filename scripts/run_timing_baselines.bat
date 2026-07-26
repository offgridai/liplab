@echo off
setlocal
cd /d "%~dp0\.."

if not exist "build-ninja\liplab_runner.exe" (
    echo Missing build-ninja\liplab_runner.exe. Run scripts\verify.bat first.
    exit /b 1
)
if not exist "build-ninja\liplab_timing_baseline.exe" (
    echo Missing build-ninja\liplab_timing_baseline.exe. Run scripts\verify.bat first.
    exit /b 1
)

build-ninja\liplab_runner.exe . --fast-batch --tick-ms 20 --export-timing-dataset
if errorlevel 1 exit /b 1

python scripts\summarize_timing_dataset.py
if errorlevel 1 exit /b 1

build-ninja\liplab_timing_baseline.exe .
if errorlevel 1 exit /b 1

echo Timing baseline experiment completed.
