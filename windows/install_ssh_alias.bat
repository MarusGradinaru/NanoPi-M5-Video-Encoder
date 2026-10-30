@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Install SSH aliases for the NanoPi encoder and the video NAS.
rem Usage: install_ssh_alias.bat <nanopi_user> <nanopi_host_ip> <nanopi_key|-> <nas_user> <nas_host_ip> <nas_key|->
rem Use a dash (-) when no private key filename is specified.
rem Private-key files, when specified, must be located in %USERPROFILE%\.ssh\.

set "SSH_DIR=%USERPROFILE%\.ssh"
set "CONFIG=%SSH_DIR%\config"

set "NANOPI_USER=%~1"
set "NANOPI_HOST=%~2"
set "NANOPI_KEY=%~3"
set "NAS_USER=%~4"
set "NAS_HOST=%~5"
set "NAS_KEY=%~6"

if "%NANOPI_USER%"=="" goto :usage
if "%NANOPI_HOST%"=="" goto :usage
if "%NANOPI_KEY%"=="" goto :usage
if "%NAS_USER%"=="" goto :usage
if "%NAS_HOST%"=="" goto :usage
if "%NAS_KEY%"=="" goto :usage
if not "%~7"=="" goto :usage

rem Keep arguments simple: usernames, hostnames/IPs and key filenames must not contain spaces.
if not "%NANOPI_USER%"=="%NANOPI_USER: =%" goto :invalid_nanopi_user
if not "%NANOPI_HOST%"=="%NANOPI_HOST: =%" goto :invalid_nanopi_host
if not "%NANOPI_KEY%"=="-" if not "%NANOPI_KEY%"=="%NANOPI_KEY: =%" goto :invalid_nanopi_key
if not "%NAS_USER%"=="%NAS_USER: =%" goto :invalid_nas_user
if not "%NAS_HOST%"=="%NAS_HOST: =%" goto :invalid_nas_host
if not "%NAS_KEY%"=="-" if not "%NAS_KEY%"=="%NAS_KEY: =%" goto :invalid_nas_key

if not exist "%SSH_DIR%\" mkdir "%SSH_DIR%" 2>nul
if not exist "%SSH_DIR%\" goto :ssh_dir_error

if not exist "%CONFIG%" type nul > "%CONFIG%"
if not exist "%CONFIG%" goto :config_error

rem Resolve the optional NanoPi private key.
set "NANOPI_IDENTITY_FILE="
if "%NANOPI_KEY%"=="-" goto :prepare_nas_key
if not exist "%SSH_DIR%\%NANOPI_KEY%" goto :nanopi_key_not_found
for %%I in ("%SSH_DIR%\%NANOPI_KEY%") do set "NANOPI_IDENTITY_FILE=%%~fI"
set "NANOPI_IDENTITY_FILE=%NANOPI_IDENTITY_FILE:\=/%"

:prepare_nas_key
rem Resolve the optional NAS private key.
set "NAS_IDENTITY_FILE="
if "%NAS_KEY%"=="-" goto :install_aliases
if not exist "%SSH_DIR%\%NAS_KEY%" goto :nas_key_not_found
for %%I in ("%SSH_DIR%\%NAS_KEY%") do set "NAS_IDENTITY_FILE=%%~fI"
set "NAS_IDENTITY_FILE=%NAS_IDENTITY_FILE:\=/%"

:install_aliases
echo.
echo OpenSSH config: "%CONFIG%"
echo.

rem Install the NanoPi alias unless a matching Host entry already exists.
findstr /I /R /C:"^[ ]*Host[ ][ ]*.*rkvenc-nanopi.*$" "%CONFIG%" >nul 2>nul
if not errorlevel 1 goto :nanopi_already_exists
call :write_alias "rkvenc-nanopi" "%NANOPI_HOST%" "%NANOPI_USER%" "%NANOPI_IDENTITY_FILE%"
if errorlevel 1 goto :config_write_error
echo [OK] Installed SSH alias "rkvenc-nanopi" for %NANOPI_USER%@%NANOPI_HOST%.
goto :install_nas_alias

:nanopi_already_exists
echo [Info] SSH alias "rkvenc-nanopi" already exists. It was left unchanged.

:install_nas_alias
rem Install the NAS alias unless a matching Host entry already exists.
findstr /I /R /C:"^[ ]*Host[ ][ ]*.*rkvenc-nas.*$" "%CONFIG%" >nul 2>nul
if not errorlevel 1 goto :nas_already_exists
call :write_alias "rkvenc-nas" "%NAS_HOST%" "%NAS_USER%" "%NAS_IDENTITY_FILE%"
if errorlevel 1 goto :config_write_error
echo [OK] Installed SSH alias "rkvenc-nas" for %NAS_USER%@%NAS_HOST%.
goto :show_summary

:nas_already_exists
echo [Info] SSH alias "rkvenc-nas" already exists. It was left unchanged.

goto :show_summary

:write_alias
rem Arguments: alias name, host, user, optional identity-file path.
>>"%CONFIG%" echo.
>>"%CONFIG%" echo # Added by install_ssh_alias.bat
>>"%CONFIG%" echo Host %~1
>>"%CONFIG%" echo     HostName %~2
>>"%CONFIG%" echo     User %~3
if not "%~4"=="" >>"%CONFIG%" echo     IdentityFile %~4
if not "%~4"=="" >>"%CONFIG%" echo     IdentitiesOnly yes
exit /b 0

:show_summary
echo.
echo SSH aliases are ready.
echo.
echo Test the connections with:
echo     ssh rkvenc-nanopi
echo     ssh rkvenc-nas
echo.
goto :success

:nanopi_key_not_found
echo.
echo [Error] NanoPi private-key file was not found:
echo     "%SSH_DIR%\%NANOPI_KEY%"
echo Use - as the third argument if no dedicated key is required.
echo.
goto :failure

:nas_key_not_found
echo.
echo [Error] NAS private-key file was not found:
echo     "%SSH_DIR%\%NAS_KEY%"
echo Use - as the sixth argument if no dedicated key is required.
echo.
goto :failure

:invalid_nanopi_user
echo.
echo [Error] The NanoPi SSH username must not contain spaces.
echo.
goto :usage_failure

:invalid_nanopi_host
echo.
echo [Error] The NanoPi host IP/name must not contain spaces.
echo.
goto :usage_failure

:invalid_nanopi_key
echo.
echo [Error] The NanoPi private-key filename must not contain spaces.
echo.
goto :usage_failure

:invalid_nas_user
echo.
echo [Error] The NAS SSH username must not contain spaces.
echo.
goto :usage_failure

:invalid_nas_host
echo.
echo [Error] The NAS host IP/name must not contain spaces.
echo.
goto :usage_failure

:invalid_nas_key
echo.
echo [Error] The NAS private-key filename must not contain spaces.
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

:config_write_error
echo.
echo [Error] Could not write the SSH aliases to "%CONFIG%".
echo Check the file permissions and available disk space.
echo.
goto :failure

:usage
echo.
echo Install the NanoPi and NAS SSH aliases in the current user's OpenSSH config.
echo.
echo Usage:
echo     %~nx0 ^<nanopi_user^> ^<nanopi_host_ip^> ^<nanopi_key^> ^<nas_user^> ^<nas_host_ip^> ^<nas_key^>
echo.
echo Examples:
echo     %~nx0 marus 192.168.0.30 nanopi_key marus 192.168.0.20 nas_key
echo     %~nx0 marus 192.168.0.30 - marus 192.168.0.20 -
echo     %~nx0 marus 192.168.0.30 nanopi_key marus 192.168.0.20 -
echo.
echo Use a dash (-) when no private key is specified.
echo Key filenames must refer to files inside:
echo     %%USERPROFILE%%\.ssh\
echo Existing alias entries are left untouched; the other alias is still processed.
echo.
goto :usage_failure

:usage_failure
endlocal
exit /b 2

:success
endlocal
exit /b 0

:failure
endlocal
exit /b 1
