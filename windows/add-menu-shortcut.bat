@echo off
setlocal

rem Usage: install-context-menu.bat <extension>
rem Example: install-context-menu.bat ts

set "EXT=%~1"

if "%EXT%"=="" (
    echo Usage: %~nx0 ^<extension^>
    echo Example: %~nx0 ts
    pause
    exit /b 1
)

rem Strip a leading dot if the user typed ".ts" instead of "ts"
if "%EXT:~0,1%"=="." set "EXT=%EXT:~1%"

rem %~dp0 = folder of this script, with trailing backslash
set "TARGET=%~dp0process-and-encode.bat"

if not exist "%TARGET%" (
    echo ERROR: cannot find "%TARGET%"
    pause
    exit /b 1
)

set "KEY=HKCU\Software\Classes\SystemFileAssociations\.%EXT%\shell\NanoPiEncoder"

reg add "%KEY%" /ve /d "Send to NanoPi Encoder..." /f >nul
if errorlevel 1 goto :err

reg add "%KEY%\command" /ve /d "\"%TARGET%\" \"%%1\"" /f >nul
if errorlevel 1 goto :err

echo Context menu installed for .%EXT% files.
echo Command: "%TARGET%" "%%1"
pause
exit /b 0

:err
echo ERROR: failed to write to the registry.
pause
exit /b 1