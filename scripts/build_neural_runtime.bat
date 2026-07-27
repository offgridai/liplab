@echo off
setlocal
cd /d "%~dp0\.."

set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%VS_DEV_CMD%" (
    echo Visual Studio developer command prompt not found.
    exit /b 1
)

call "%VS_DEV_CMD%" -arch=x64 >nul
if errorlevel 1 exit /b 1

echo [phase] configure_standalone_cuda_runtime
"%CMAKE_EXE%" -S . -B build-neural-runtime -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DLIPLAB_ENABLE_NEURAL_RUNTIME=ON -DLIPLAB_ENABLE_TORCH=OFF
if errorlevel 1 exit /b 1

echo [phase] build_self_contained_binary
"%CMAKE_EXE%" --build build-neural-runtime --target liplab_neural_runtime_smoke -j 2
if errorlevel 1 exit /b 1

echo [phase] audit_rtx_4090_image
python scripts\check_cuda_architectures.py build-neural-runtime\liplab_neural_streamer_cuda.lib --required sm_89
if errorlevel 1 exit /b 1

echo [phase] run_embedded_model_inference
build-neural-runtime\liplab_neural_runtime_smoke.exe
if errorlevel 1 exit /b 1

echo [phase] reject_libtorch_linkage
dumpbin /dependents build-neural-runtime\liplab_neural_runtime_smoke.exe | findstr /i "torch c10" >nul
if not errorlevel 1 (
    echo Runtime binary unexpectedly depends on LibTorch.
    exit /b 1
)

echo Standalone neural runtime is ready: build-neural-runtime\liplab_neural_runtime_smoke.exe
endlocal
exit /b 0
