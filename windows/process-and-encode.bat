@echo off
setlocal DisableDelayedExpansion

set "EXIT_CODE=0"

if "%~1"=="" (
    echo [Error] Missing input video path.
    echo Usage: %~nx0 "N:\path\to\video.ts"
    set "EXIT_CODE=2"
    goto :finish
)

if not "%~2"=="" (
    echo [Error] Only one input video path is expected.
    echo Usage: %~nx0 "N:\path\to\video.ts"
    set "EXIT_CODE=2"
    goto :finish
)

where ssh.exe >nul 2>&1
if errorlevel 1 (
    echo [Error] OpenSSH client ssh.exe was not found in PATH.
    set "EXIT_CODE=2"
    goto :finish
)

rem Resolve the supplied path to an absolute Windows path.
for %%I in ("%~1") do set "WINDOWS_VIDEO=%%~fI"

if /I not "%WINDOWS_VIDEO:~0,3%"=="N:\" (
    echo [Error] The input file must be located on the mapped NAS drive N:.
    echo Received: "%WINDOWS_VIDEO%"
    set "EXIT_CODE=2"
    goto :finish
)

if not exist "%WINDOWS_VIDEO%" (
    echo [Error] Input file does not exist or is not accessible:
    echo "%WINDOWS_VIDEO%"
    set "EXIT_CODE=2"
    goto :finish
)

rem Convert N:\WDRed2\... to /NAS/WDRed2/...
set "RELATIVE_PATH=%WINDOWS_VIDEO:~3%"
set "RELATIVE_PATH=%RELATIVE_PATH:\=/%"
set "LINUX_VIDEO=/NAS/%RELATIVE_PATH%"

rem Escape apostrophes for a POSIX single-quoted shell argument.
rem Example: John's.ts becomes 'John'\''s.ts'.
set "LINUX_VIDEO_ESCAPED=%LINUX_VIDEO:'='\''%"

echo Windows path: "%WINDOWS_VIDEO%"
echo Linux path:   "%LINUX_VIDEO%"
echo.
echo Starting remote processing on NanoPi...
echo.

ssh.exe -tt rkvenc-nanopi "cd '/home/marus/NanoPi-M5-Video-Encoder/nanopi' && bash 'process-and-encode.sh' '%LINUX_VIDEO_ESCAPED%'"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [Error] Remote processing failed with exit code %EXIT_CODE%.
)

goto :finish

:finish
echo.
pause
endlocal & exit /b %EXIT_CODE%
