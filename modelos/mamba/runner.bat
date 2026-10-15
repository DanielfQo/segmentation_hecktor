@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MambaLite3D - ejecutor de trabajos (cierra esta ventana para detenerlo)
cd /d "%~dp0"
for %%D in (jobs jobs\pending jobs\running jobs\done jobs\logs) do if not exist "%%D" mkdir "%%D"
echo ==================================================================
echo  Ejecutor de trabajos de MambaLite3D
echo  Ejecuta, uno a la vez, los scripts .bat que aparecen en jobs\pending
echo  y guarda su salida en jobs\logs. Cierra esta ventana para detenerlo.
echo ==================================================================
:loop
>"jobs\runner_heartbeat.txt" echo %date% %time%
for %%F in ("jobs\pending\*.bat") do (
    set "JOB=%%~nF"
    echo [!date! !time!] Iniciando !JOB!
    move /y "%%~fF" "jobs\running\!JOB!.bat" >nul
    >"jobs\logs\!JOB!.started" echo !date! !time!
    cmd /d /c jobs\running\!JOB!.bat > "jobs\logs\!JOB!.log" 2>&1
    set "RC=!errorlevel!"
    >"jobs\logs\!JOB!.exit" echo !RC!
    move /y "jobs\running\!JOB!.bat" "jobs\done\!JOB!.bat" >nul
    echo [!date! !time!] Terminado !JOB! con codigo !RC!
)
ping -n 6 127.0.0.1 >nul
goto loop
