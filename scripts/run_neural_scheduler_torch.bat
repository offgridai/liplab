@echo off
setlocal
cd /d "%~dp0\.."

set "LIBTORCH_ROOT=%~1"
if not defined LIBTORCH_ROOT set "LIBTORCH_ROOT=C:\aitoolkit\libtorch-2.13.0-cu130"
if not exist "%LIBTORCH_ROOT%\share\cmake\Torch\TorchConfig.cmake" exit /b 1

set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call "%VS_DEV_CMD%" -arch=x64 >nul
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" -S . -B build-torch -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIPLAB_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH="%LIBTORCH_ROOT%"
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build build-torch --target liplab_runner liplab_timing_baseline liplab_neural_scheduler_torch
if errorlevel 1 exit /b 1

copy /y build-torch\liplab_runner.exe build-ninja\liplab_runner.exe >nul
copy /y build-torch\liplab_timing_baseline.exe build-ninja\liplab_timing_baseline.exe >nul

call scripts\run_timing_baselines.bat
if errorlevel 1 exit /b 1

set "PATH=%LIBTORCH_ROOT%\lib;%PATH%"
build-torch\liplab_neural_scheduler_torch.exe . transcript_audio
if errorlevel 1 exit /b 1
build-torch\liplab_neural_scheduler_torch.exe . with_deterministic
if errorlevel 1 exit /b 1

if not exist outputs\runs\neural_scheduler_artifacts mkdir outputs\runs\neural_scheduler_artifacts
copy /y outputs\runs\latest\neural_scheduler_* outputs\runs\neural_scheduler_artifacts\ >nul

build-ninja\liplab_runner.exe . --fast-batch --tick-ms 20 --timing-advice-csv outputs\runs\latest\neural_scheduler_with_deterministic_predictions.csv
if errorlevel 1 exit /b 1
python scripts\summarize.py
if errorlevel 1 exit /b 1
python scripts\summarize_timing_model.py
if errorlevel 1 exit /b 1
copy /y outputs\runs\latest\timing_model_grade_summary.json outputs\runs\neural_scheduler_artifacts\neural_scheduler_grade_summary.json >nul
python scripts\check_grades.py
if errorlevel 1 exit /b 1

echo Neural scheduler experiment completed.
