@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "CMAKE_EXE=cmake"
set "NINJA_EXE=ninja"
set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
where /q "%CMAKE_EXE%"
if errorlevel 1 (
    set "VS_CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if exist "!VS_CMAKE!" (
        set "CMAKE_EXE=!VS_CMAKE!"
    ) else (
        echo cmake not found on PATH and Visual Studio CMake was not found at:
        echo   !VS_CMAKE!
        exit /b 1
    )
)

where /q "%NINJA_EXE%"
if errorlevel 1 (
    set "VS_NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    if exist "!VS_NINJA!" (
        set "NINJA_EXE=!VS_NINJA!"
        set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;!PATH!"
    ) else (
        echo ninja not found on PATH and Visual Studio Ninja was not found at:
        echo   !VS_NINJA!
        exit /b 1
    )
)

if exist "!VS_DEV_CMD!" (
    call "!VS_DEV_CMD!" -arch=x64 >nul
    if errorlevel 1 (
        echo Failed to initialize Visual Studio developer command prompt:
        echo   !VS_DEV_CMD!
        exit /b 1
    )
) else (
    echo Visual Studio developer command prompt not found at:
    echo   !VS_DEV_CMD!
    exit /b 1
)

set "CC="
set "CXX="

set "LIPLAB_WINDOWS_SDK_VERSION="
if defined WindowsSDKVersion (
    set "LIPLAB_WINDOWS_SDK_VERSION=%WindowsSDKVersion%"
    if "!LIPLAB_WINDOWS_SDK_VERSION:~-1!"=="\" (
        set "LIPLAB_WINDOWS_SDK_VERSION=!LIPLAB_WINDOWS_SDK_VERSION:~0,-1!"
    )
)

if not defined LIPLAB_WINDOWS_SDK_VERSION (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$root = 'C:\Program Files (x86)\Windows Kits\10\Lib'; if (Test-Path $root) { $v = Get-ChildItem -LiteralPath $root -Directory | Select-Object -ExpandProperty Name | Sort-Object {[version]$_} -Descending | Select-Object -First 1; if ($v) { Write-Output $v } }"`) do (
        set "LIPLAB_WINDOWS_SDK_VERSION=%%I"
    )
)

if defined LIPLAB_WINDOWS_SDK_VERSION (
    set "CMAKE_SDK_ARG=-DCMAKE_SYSTEM_VERSION=!LIPLAB_WINDOWS_SDK_VERSION!"
    set "WindowsTargetPlatformVersion=!LIPLAB_WINDOWS_SDK_VERSION!"
    echo Using Windows SDK !LIPLAB_WINDOWS_SDK_VERSION!
)

set "BUILD_DIR=build-ninja"
set "CMAKE_GENERATOR_ARG=-G Ninja"

if exist "!BUILD_DIR!\CMakeCache.txt" del /q "!BUILD_DIR!\CMakeCache.txt"

echo [phase] configure
"%CMAKE_EXE%" -S . -B "!BUILD_DIR!" !CMAKE_GENERATOR_ARG! !CMAKE_SDK_ARG! -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

echo [phase] build
"%CMAKE_EXE%" --build "!BUILD_DIR!"
if errorlevel 1 exit /b 1

echo [phase] validate_gold
python scripts\check_gold.py --include-drafts
if errorlevel 1 exit /b 1

echo [phase] run_corpus
set "LIPLAB_PREROLL_MS=350"
if not "%~1"=="" set "LIPLAB_PREROLL_MS=%~1"
echo Using preroll !LIPLAB_PREROLL_MS! ms

if exist "!BUILD_DIR!\liplab_runner.exe" (
    "!BUILD_DIR!\liplab_runner.exe" . --preroll-ms !LIPLAB_PREROLL_MS!
) else if exist build\Release\liplab_runner.exe (
    build\Release\liplab_runner.exe . --preroll-ms !LIPLAB_PREROLL_MS!
) else if exist build\liplab_runner.exe (
    build\liplab_runner.exe . --preroll-ms !LIPLAB_PREROLL_MS!
) else (
    echo liplab_runner.exe not found under !BUILD_DIR!, build\Release, or build\
    exit /b 1
)
if errorlevel 1 exit /b 1

echo [phase] summarize
python scripts\summarize.py
if errorlevel 1 exit /b 1

echo [phase] signal_audits
python scripts\signal_audits.py
if errorlevel 1 exit /b 1

echo [phase] check_grades
python scripts\check_grades.py
if errorlevel 1 exit /b 1

endlocal
