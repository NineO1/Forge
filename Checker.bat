@echo off
setlocal EnableDelayedExpansion
title Forge Dependency Checker
cls

set "PY=venv\Scripts\python.exe"
set /a PASS=0
set /a FAIL=0

echo ============================================================
echo   FORGE - DEPENDENCY CHECKER
echo ============================================================
echo.

echo [1/8] Python environment...
if not exist "%PY%" (
    echo   [RED] venv python not found at %PY%
    set /a FAIL+=1
) else (
    set "PYVER="
    for /f "delims=" %%v in ('%PY% -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "PYVER=%%v"
    if "!PYVER!"=="" (
        echo   [RED] venv exists but Python fails to run
        set /a FAIL+=1
    ) else (
        echo   [GREEN] venv OK - Python !PYVER!
        set /a PASS+=1
    )
)
echo.

echo [2/8] Core packages...
for %%p in (torch faster_whisper scenedetect cv2 mediapipe PyQt6 diffusers transformers) do (
    "%PY%" -c "import %%p" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [GREEN] %%p
        set /a PASS+=1
    ) else (
        echo   [RED] %%p - FIX: pip install %%p
        set /a FAIL+=1
    )
)
echo.

echo [3/8] GPU check...
"%PY%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if !errorlevel! equ 0 (
    set "GPUNAME="
    for /f "delims=" %%g in ('%PY% -c "import torch; print(torch.cuda.get_device_name(0))" 2^>nul') do set "GPUNAME=%%g"
    if "!GPUNAME!"=="" (
        echo   [GREEN] CUDA available
    ) else (
        echo   [GREEN] CUDA available - !GPUNAME!
    )
    set /a PASS+=1
) else (
    echo   [RED] CUDA not available - video generation will be disabled
    echo   NOTE: extraction features still work on CPU
    set /a FAIL+=1
)
echo.

echo [4/8] Tokenizer extras (sentencepiece, tiktoken)...
for %%p in (sentencepiece tiktoken) do (
    "%PY%" -c "import %%p" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [GREEN] %%p
        set /a PASS+=1
    ) else (
        echo   [RED] %%p - FIX: pip install %%p
        set /a FAIL+=1
    )
)
echo.

echo [5/8] FFmpeg...
if exist "bin\ffmpeg.exe" (
    bin\ffmpeg.exe -version >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [GREEN] ffmpeg OK
        set /a PASS+=1
    ) else (
        echo   [RED] ffmpeg.exe present but fails to run
        set /a FAIL+=1
    )
) else (
    echo   [RED] bin\ffmpeg.exe missing - FIX: copy full gyan.dev build into bin\
    set /a FAIL+=1
)
if exist "bin\ffprobe.exe" (
    echo   [GREEN] ffprobe OK
    set /a PASS+=1
) else (
    echo   [RED] bin\ffprobe.exe missing - FIX: copy full gyan.dev build into bin\
    set /a FAIL+=1
)
echo.

echo [6/8] Model cache...
call :CheckModelSize "SDXL" "models\hf\hub\models--stabilityai--stable-diffusion-xl-base-1.0" 5368709120
call :CheckModelSize "LTX-Video" "models\hf\hub\models--Lightricks--LTX-Video" 5368709120
call :CheckModelSize "Whisper large-v3" "models\whisper" 2000000000
call :CheckModelSize "MediaPipe detector" "models\mediapipe" 1048576
echo.

echo [7/8] Folder structure...
for %%d in (forge bin models output) do (
    if exist "%%d\" (
        echo   [GREEN] %%d\
        set /a PASS+=1
    ) else (
        echo   [RED] %%d\ missing - FIX: create it or reinstall package
        set /a FAIL+=1
    )
)
echo.

echo [8/8] Summary...
echo ============================================================
if !FAIL! equ 0 (
    echo   ALL GREEN - !PASS! checks passed. Forge is ready.
) else (
    echo   !PASS! passed, !FAIL! FAILED - see RED lines above for fixes.
)
echo ============================================================
echo.
pause
exit /b 0

:CheckModelSize
set "MS=" & set "MF="
for /f "tokens=1,2" %%s in ('%PY% -c "import os;s=sum(os.path.getsize(os.path.join(r,f)) for r,d,fs in os.walk(r'%~2') for f in fs);print((str(round(s/1048576,1))+'MB' if s<1073741824 else str(round(s/1073741824,1))+'GB'),'OK' if s>%~3 else 'MISSING')" 2^>nul') do (
    set "MS=%%s"
    set "MF=%%t"
)
if /i "!MF!"=="OK" (
    echo   [GREEN] %~1 cached ^(!MS!^)
    set /a PASS+=1
) else (
    echo   [RED] %~1 not cached - FIX: run that feature once to download it
    set /a FAIL+=1
)
goto :eof