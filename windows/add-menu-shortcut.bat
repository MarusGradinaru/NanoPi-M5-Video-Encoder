@echo off
setlocal

rem Installs a right-click context-menu entry for one file extension.
rem The entry runs process-and-encode.bat (located next to this script) with two arguments:
rem   1. the selected video file
rem   2. the path where the NAS is mounted on the NanoPi
rem Usage:   add-menu-shortcut.bat [extension] [nas-mount-point]
rem Example: add-menu-shortcut.bat ts /NAS
rem Any argument that is missing is asked for interactively.

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

rem %~dp0 = folder of this script, with trailing backslash
set "TARGET=%~dp0process-and-encode.bat"

if not exist "%TARGET%" (
    echo ERROR: cannot find "%TARGET%"
    goto :fail
)

if not "%~3"=="" (
    echo ERROR: too many arguments.
    echo Usage: %~nx0 [extension] [nas-mount-point]
    goto :fail
)

rem These also clear any same-named variable inherited from the environment.
set "EXT=%~1"
set "NAS_MOUNT=%~2"

echo Install the "Send to NanoPi Encoder..." context-menu entry.
echo.

if not defined EXT set /p "EXT=File extension to install the command for (e.g. ts): "

if not defined EXT (
    echo ERROR: no extension entered.
    goto :fail
)

rem Strip a leading dot if the user typed ".ts" instead of "ts"
if "%EXT:~0,1%"=="." set "EXT=%EXT:~1%"

if not defined EXT (
    echo ERROR: no extension entered.
    goto :fail
)

if not defined NAS_MOUNT set /p "NAS_MOUNT=Where is the NAS mounted on the NanoPi ? [/NAS]: "
if not defined NAS_MOUNT set "NAS_MOUNT=/NAS"

if not "%NAS_MOUNT:~0,1%"=="/" (
    echo ERROR: the mount point must be an absolute Linux path, starting with "/".
    goto :fail
)

set "KEY=HKCU\Software\Classes\SystemFileAssociations\.%EXT%\shell\NanoPiEncoder"

reg add "%KEY%" /ve /d "Send to NanoPi Encoder..." /f >nul
if errorlevel 1 goto :err

reg add "%KEY%\command" /ve /d "\"%TARGET%\" \"%%1\" \"%NAS_MOUNT%\"" /f >nul
if errorlevel 1 goto :err

echo Context menu installed for .%EXT% files.
echo Command: "%TARGET%" "%%1" "%NAS_MOUNT%"
if defined PAUSE_AT_END echo. && pause
exit /b 0

:err
echo ERROR: failed to write to the registry.

:fail
if defined PAUSE_AT_END echo. && pause
exit /b 1
