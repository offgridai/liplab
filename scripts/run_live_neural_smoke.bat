@echo off
setlocal
cd /d "%~dp0\.."

set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "MODEL=offgrid_dropin\Private\Lipsync\Models\OffgridAINeuralStreamerV3.bin"

call "%VS_DEV_CMD%" -arch=x64 >nul
if errorlevel 1 exit /b 1

call scripts\build_neural_runtime.bat
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build build-neural-runtime --target liplab_runner -j 2
if errorlevel 1 exit /b 1

build-neural-runtime\liplab_runner.exe --root . --case %1 --neural-checkpoint "%MODEL%" --fast-batch
if errorlevel 1 exit /b 1

echo Live neural session smoke passed for %1.
endlocal
