@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "LIBTORCH_ROOT=%~1"
if not defined LIBTORCH_ROOT set "LIBTORCH_ROOT=C:\aitoolkit\libtorch-2.13.0-cu130"
if not exist "%LIBTORCH_ROOT%\share\cmake\Torch\TorchConfig.cmake" (
    echo TorchConfig.cmake not found under %LIBTORCH_ROOT%
    exit /b 1
)

set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%CMAKE_EXE%" exit /b 1
if not exist "%VS_DEV_CMD%" exit /b 1

call "%VS_DEV_CMD%" -arch=x64 >nul
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" -S . -B build-torch -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIPLAB_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH="%LIBTORCH_ROOT%"
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build build-torch --config Release --target liplab_timing_torch
if errorlevel 1 exit /b 1

call scripts\run_timing_baselines.bat
if errorlevel 1 exit /b 1

set "PATH=%LIBTORCH_ROOT%\lib;%PATH%"
build-torch\liplab_timing_torch.exe .
if errorlevel 1 exit /b 1

if not exist outputs\runs\timing_model_artifacts mkdir outputs\runs\timing_model_artifacts
copy /y outputs\runs\latest\timing_torch_model.pt outputs\runs\timing_model_artifacts\timing_torch_model.pt >nul
copy /y outputs\runs\latest\timing_torch_report.json outputs\runs\timing_model_artifacts\timing_torch_report.json >nul
copy /y outputs\runs\latest\timing_torch_predictions.csv outputs\runs\timing_model_artifacts\timing_torch_predictions.csv >nul

build-ninja\liplab_runner.exe . --fast-batch --tick-ms 20 --timing-advice-csv outputs\runs\timing_model_artifacts\timing_torch_predictions.csv
if errorlevel 1 exit /b 1
python scripts\summarize.py
if errorlevel 1 exit /b 1
python scripts\summarize_timing_model.py
if errorlevel 1 exit /b 1
python scripts\check_grades.py
if errorlevel 1 exit /b 1

echo CUDA timing experiment completed.
