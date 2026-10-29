@echo off
setlocal EnableExtensions

rem Install an OpenSSH client alias for the current Windows user.
rem Usage: install_ssh_alias.bat <user> <host_ip> [private_key_filename]
rem If supplied, the private-key file is expected in %USERPROFILE%\.ssh\.

set "ALIAS_NAME=nanopi-encoder"
set "SSH_DIR=%USERPROFILE%\.ssh"
set "CONFIG=%SSH_DIR%\config"
set "SSH_USER=%~1"
set "HOST_IP=%~2"
set "PRIVATE_KEY_NAME=%~3"

if "%SSH_USER%"=="" goto :usage
if "%HOST_IP%"=="" goto :usage
if not "%~4"=="" goto :usage

rem Keep arguments simple: these values must not contain spaces.
if not "%SSH_USER%"=="%SSH_USER: =%" goto :invalid_user
if not "%HOST_IP%"=="%HOST_IP: =%" goto :invalid_host
if not "%PRIVATE_KEY_NAME%"=="%PRIVATE_KEY_NAME: =%" goto :invalid_key

if not exist "%SSH_DIR%\" mkdir "%SSH_DIR%" 2>nul
if not exist "%SSH_DIR%\" goto :ssh_dir_error

if not exist "%CONFIG%" type nul > "%CONFIG%"
if not exist "%CONFIG%" goto :config_error

rem Avoid creating duplicate Host blocks. Existing entries are left untouched.
findstr /I /R /C:"^[ ]*Host[ ][ ]*.*nanopi-encoder.*$" "%CONFIG%" >nul 2>nul
if not errorlevel 1 goto :alias_exists

set "IDENTITY_FILE="
if not "%PRIVATE_KEY_NAME%"=="" goto :check_private_key

goto :write_alias

:check_private_key
rem Interpret the optional key argument as a filename inside the user's .ssh folder.
if not exist "%SSH_DIR%\%PRIVATE_KEY_NAME%" goto :key_not_found
for %%I in ("%SSH_DIR%\%PRIVATE_KEY_NAME%") do set "IDENTITY_FILE=%%~fI"
rem Forward slashes are accepted by Windows OpenSSH and avoid config escaping issues.
set "IDENTITY_FILE=%IDENTITY_FILE:\=/%"

goto :write_alias

:write_alias
echo.
echo Installing SSH alias "%ALIAS_NAME%" for the current Windows user...
echo Config file: "%CONFIG%"
echo SSH user:   %SSH_USER%
echo Host:       %HOST_IP%
if not "%IDENTITY_FILE%"=="" echo Private key: %IDENTITY_FILE%
if "%IDENTITY_FILE%"=="" echo Private key: not specified (OpenSSH default key discovery will be used)

echo.>>"%CONFIG%"
>>"%CONFIG%" echo # Added by install_ssh_alias.bat
>>"%CONFIG%" echo Host %ALIAS_NAME%
>>"%CONFIG%" echo     HostName %HOST_IP%
>>"%CONFIG%" echo     User %SSH_USER%
if not "%IDENTITY_FILE%"=="" >>"%CONFIG%" echo     IdentityFile %IDENTITY_FILE%
if not "%IDENTITY_FILE%"=="" >>"%CONFIG%" echo     IdentitiesOnly yes

echo.
echo [OK] SSH alias "%ALIAS_NAME%" was added.
echo.
echo Test it with:
echo     ssh %ALIAS_NAME%
echo.
goto :success

:alias_exists
echo.
echo [Warning] An SSH Host entry mentioning "%ALIAS_NAME%" already exists in:
echo     "%CONFIG%"
echo No changes were made, so existing SSH aliases and settings remain untouched.
echo Edit or remove the existing "%ALIAS_NAME%" entry before running this installer again.
echo.
goto :failure

:key_not_found
echo.
echo [Error] Private key file was not found:
echo     "%SSH_DIR%\%PRIVATE_KEY_NAME%"
echo The optional third parameter must be the key filename located in the user's .ssh folder.
echo.
goto :failure

:invalid_user
echo.
echo [Error] The SSH username must not contain spaces.
echo.
goto :usage_failure

:invalid_host
echo.
echo [Error] The host IP/name must not contain spaces.
echo.
goto :usage_failure

:invalid_key
echo.
echo [Error] The private-key filename must not contain spaces.
echo.
goto :usage_failure

:ssh_dir_error
echo.
echo [Error] Could not create or access "%SSH_DIR%".
echo.
goto :failure

:config_error
echo.
echo [Error] Could not create or access "%CONFIG%".
echo.
goto :failure

:usage
echo.
echo Install an SSH alias named "%ALIAS_NAME%" in the current user's OpenSSH config.
echo.
echo Usage:
echo     %~nx0 ^<user^> ^<host_ip^> [private_key_filename]
echo.
echo Examples:
echo     %~nx0 user 192.168.0.5 user_key
echo     %~nx0 user 192.168.0.5
echo.
echo If provided, private_key_filename must name a file inside:
echo     %%USERPROFILE%%\.ssh\
echo The installer's existing config and all other aliases are preserved.
echo.

:usage_failure
endlocal
exit /b 2

:success
endlocal
exit /b 0

:failure
endlocal
exit /b 1
