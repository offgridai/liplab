@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set "CMAKE_EXE=cmake"
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

"%CMAKE_EXE%" -S . -B build
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build build --config Release
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
