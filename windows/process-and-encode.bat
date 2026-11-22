@echo off
setlocal DisableDelayedExpansion

rem Usage: process-and-encode.bat "N:\path\to\video.ts" "/NAS"
rem   1. absolute Windows path of the video, on a mapped network drive (any drive letter)
rem   2. path where that network share is mounted on the NanoPi
rem The context-menu entry created by add-menu-shortcut.bat supplies both arguments.

rem Pause at the end only when the script runs in its own console window (double-click,
rem Explorer context menu), because that window closes as soon as the script ends.
rem In that case cmd.exe was started with the /c switch (CMDCMDLINE variable, see the help of
rem the SET command); an interactive terminal runs the script in the existing shell, whose
rem original command line has no /c switch.
rem The value is copied into a normal variable and its quotes are removed before the
rem comparison, so that quotes and special characters in paths cannot break the test.
set "PAUSE_AT_END="
set "_CMDLINE=%cmdcmdline%"
set "_CMDLINE=%_CMDLINE:"=%"
if not "%_CMDLINE%"=="%_CMDLINE:/c =%" set "PAUSE_AT_END=1"
set "_CMDLINE="

set "EXIT_CODE=0"

if "%~1"=="" (
    echo [Error] Missing input video path.
    echo Usage: %~nx0 "N:\path\to\video.ts" "/NAS"
    set "EXIT_CODE=2"
    goto :finish
)

if "%~2"=="" (
    echo [Error] Missing the NAS mount point on the NanoPi.
    echo Usage: %~nx0 "N:\path\to\video.ts" "/NAS"
    echo Reinstall the context-menu entry with add-menu-shortcut.bat.
    set "EXIT_CODE=2"
    goto :finish
)

if not "%~3"=="" (
    echo [Error] Only two arguments are expected: the video path and the NAS mount point.
    echo Usage: %~nx0 "N:\path\to\video.ts" "/NAS"
    set "EXIT_CODE=2"
    goto :finish
)

where ssh.exe >nul 2>&1
if errorlevel 1 (
    echo [Error] OpenSSH client ssh.exe was not found in PATH.
    set "EXIT_CODE=2"
    goto :finish
)

where scp.exe >nul 2>&1
if errorlevel 1 (
    echo [Error] OpenSSH client scp.exe was not found in PATH.
    set "EXIT_CODE=2"
    goto :finish
)

rem The INI file is stored beside this BAT file, independent of the current directory.
set "CONFIG_FILE=%~dp0process-and-encode.ini"
if not exist "%CONFIG_FILE%" (
    echo [Error] Configuration file not found:
    echo "%CONFIG_FILE%"
    set "EXIT_CODE=2"
    goto :finish
)

rem Resolve the supplied path to an absolute Windows path.
for %%I in ("%~1") do set "WINDOWS_VIDEO=%%~fI"

rem The path must start with a drive letter, like "N:\".
if not "%WINDOWS_VIDEO:~1,2%"==":\" (
    echo [Error] The input file must be given as an absolute path starting with a drive letter.
    echo Received: "%WINDOWS_VIDEO%"
    set "EXIT_CODE=2"
    goto :finish
)

rem That drive must be a mapped network drive. Without this check, a file on a local
rem drive could be translated to a NAS path and silently match a different file.
set "DRIVE=%WINDOWS_VIDEO:~0,2%"
net use %DRIVE% >nul 2>&1
if errorlevel 1 (
    echo [Error] Drive %DRIVE% is not a mapped network drive.
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

rem Normalize the NAS mount point: it must be an absolute Linux path, without trailing slashes.
set "NAS_MOUNT=%~2"
:trim_slash
if not "%NAS_MOUNT%"=="/" if "%NAS_MOUNT:~-1%"=="/" (
    set "NAS_MOUNT=%NAS_MOUNT:~0,-1%"
    goto :trim_slash
)

if not "%NAS_MOUNT:~0,1%"=="/" (
    echo [Error] The NAS mount point must be an absolute Linux path, starting with "/".
    echo Received: "%NAS_MOUNT%"
    set "EXIT_CODE=2"
    goto :finish
)

rem Replace the drive prefix ("N:\") with the NAS mount point and turn the
rem remaining backslashes into slashes: N:\WDRed2\video.ts becomes /NAS/WDRed2/video.ts
rem A mount point of "/" (NAS mounted at the root) must not produce a double slash.
set "MOUNT_PREFIX=%NAS_MOUNT%"
if "%MOUNT_PREFIX%"=="/" set "MOUNT_PREFIX="

set "RELATIVE_PATH=%WINDOWS_VIDEO:~3%"
set "RELATIVE_PATH=%RELATIVE_PATH:\=/%"
set "LINUX_VIDEO=%MOUNT_PREFIX%/%RELATIVE_PATH%"

rem Escape apostrophes for a POSIX single-quoted shell argument.
rem Example: John's.ts becomes 'John'\''s.ts'.
set "LINUX_VIDEO_ESCAPED=%LINUX_VIDEO:'='\''%"

echo Windows path : %WINDOWS_VIDEO%
echo Linux path   : %LINUX_VIDEO%
echo INI file     : %CONFIG_FILE%
echo.
echo Uploading configuration to NanoPi RAM...
scp.exe -q "%CONFIG_FILE%" rkvenc-nanopi:/dev/shm/process-and-encode.ini
set "SCP_EXIT_CODE=%ERRORLEVEL%"
if not "%SCP_EXIT_CODE%"=="0" (
    echo.
    echo [Error] Could not upload the configuration file to NanoPi. SCP exit code %SCP_EXIT_CODE%.
    set "EXIT_CODE=%SCP_EXIT_CODE%"
    goto :finish
)

echo Starting remote processing on NanoPi...
echo.

ssh.exe -tt rkvenc-nanopi "cd '/home/marus/NanoPi-M5-Video-Encoder/nanopi' && bash 'process-and-encode.sh' '%LINUX_VIDEO_ESCAPED%'"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [Error] Remote processing failed with exit code %EXIT_CODE%.
)

:finish
if defined PAUSE_AT_END echo. && pause
endlocal & exit /b %EXIT_CODE%
