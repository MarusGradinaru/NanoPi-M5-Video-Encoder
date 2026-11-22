@echo off
setlocal

rem Removes the right-click context-menu entry installed by add-menu-shortcut.bat.
rem Usage: remove-menu-shortcut.bat [extension]
rem If the extension is missing it is asked for interactively.

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

if not "%~2"=="" (
    echo ERROR: too many arguments.
    echo Usage: %~nx0 [extension]
    goto :fail
)

rem This also clears any same-named variable inherited from the environment.
set "EXT=%~1"

if not defined EXT set /p "EXT=File extension to remove the command from (e.g. ts): "

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

set "KEY=HKCU\Software\Classes\SystemFileAssociations\.%EXT%\shell\NanoPiEncoder"

reg query "%KEY%" >nul 2>&1
if errorlevel 1 (
    echo Nothing to remove: no NanoPi Encoder context menu found for .%EXT% files.
    goto :done
)

reg delete "%KEY%" /f >nul
if errorlevel 1 (
    echo ERROR: failed to delete the registry key.
    goto :fail
)
echo Context menu removed for .%EXT% files.

:done
if defined PAUSE_AT_END echo. && pause
exit /b 0

:fail
if defined PAUSE_AT_END echo. && pause
exit /b 1
