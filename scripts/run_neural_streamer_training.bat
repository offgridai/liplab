@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "TORCH_ROOT=C:\aitoolkit\libtorch-2.13.0-cu130"
if not "%~1"=="" set "TORCH_ROOT=%~1"
set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "ARTIFACT_DIR=outputs\runs\neural_streamer_artifacts\current"

if not exist "%TORCH_ROOT%\share\cmake\Torch\TorchConfig.cmake" (
    echo TorchConfig.cmake not found under %TORCH_ROOT%
    exit /b 1
)
if not exist "%VS_DEV_CMD%" (
    echo Visual Studio developer command prompt not found
    exit /b 1
)

call "%VS_DEV_CMD%" -arch=x64 >nul
if errorlevel 1 exit /b 1

echo [phase] configure_cuda
"%CMAKE_EXE%" -S . -B build-torch -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DLIPLAB_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH="%TORCH_ROOT%"
if errorlevel 1 exit /b 1

echo [phase] build_cuda
"%CMAKE_EXE%" --build build-torch --target liplab_runner liplab_monotonic_aligner_torch -j 2
if errorlevel 1 exit /b 1

echo [phase] audit_cuda_compatibility
python scripts\check_cuda_architectures.py build-torch\liplab_neural_streamer_cuda.lib --required sm_89
if errorlevel 1 exit /b 1

echo [phase] export_sequence_dataset
build-torch\liplab_runner.exe . --fast-batch --tick-ms 40 --export-monotonic-dataset
if errorlevel 1 exit /b 1

echo [phase] train_curriculum
set "PATH=%TORCH_ROOT%\lib;%PATH%"
build-torch\liplab_monotonic_aligner_torch.exe .
if errorlevel 1 exit /b 1

if not exist "%ARTIFACT_DIR%" mkdir "%ARTIFACT_DIR%"
copy /y outputs\runs\latest\neural_streamer.pt "%ARTIFACT_DIR%\neural_streamer.pt" >nul
if errorlevel 1 exit /b 1
copy /y outputs\runs\latest\neural_streamer_cuda.bin "%ARTIFACT_DIR%\neural_streamer_cuda.bin" >nul
if errorlevel 1 exit /b 1
copy /y outputs\runs\latest\neural_streamer_predictions.csv "%ARTIFACT_DIR%\neural_streamer_predictions.csv" >nul
if errorlevel 1 exit /b 1
copy /y outputs\runs\latest\neural_streamer_report.json "%ARTIFACT_DIR%\neural_streamer_report.json" >nul
if errorlevel 1 exit /b 1

echo [phase] replay_neural_owned_stream
build-torch\liplab_runner.exe . --fast-batch --tick-ms 40 ^
    --neural-track-csv "%ARTIFACT_DIR%\neural_streamer_predictions.csv"
if errorlevel 1 exit /b 1

echo [phase] summarize_neural_owned_stream
python scripts\summarize_neural_track.py --output "%ARTIFACT_DIR%\grade_summary.json"
if errorlevel 1 exit /b 1

echo Neural streamer training and scoring completed.
endlocal
