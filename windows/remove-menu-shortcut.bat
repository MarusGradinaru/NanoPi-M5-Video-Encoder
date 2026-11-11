@echo off
setlocal

rem Usage: uninstall-context-menu.bat <extension>
rem Example: uninstall-context-menu.bat ts

set "EXT=%~1"

if "%EXT%"=="" (
    echo Usage: %~nx0 ^<extension^>
    echo Example: %~nx0 ts
    pause
    exit /b 1
)

rem Strip a leading dot if the user typed ".ts" instead of "ts"
if "%EXT:~0,1%"=="." set "EXT=%EXT:~1%"

set "KEY=HKCU\Software\Classes\SystemFileAssociations\.%EXT%\shell\NanoPiEncoder"

reg query "%KEY%" >nul 2>&1
if errorlevel 1 (
    echo Nothing to remove: no NanoPi Encoder context menu found for .%EXT% files.
    pause
    exit /b 0
)

reg delete "%KEY%" /f >nul
if errorlevel 1 (
    echo ERROR: failed to delete the registry key.
    pause
    exit /b 1
)

echo Context menu removed for .%EXT% files.
pause
exit /b 0