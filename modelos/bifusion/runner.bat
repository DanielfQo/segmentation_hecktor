@echo off
setlocal EnableExtensions EnableDelayedExpansion
title BiFusion - ejecutor de trabajos (cierra esta ventana para detenerlo)
cd /d "%~dp0"
for %%D in (jobs jobs\pending jobs\running jobs\done jobs\logs) do if not exist "%%D" mkdir "%%D"
rem No arranca si ya hay un entrenamiento o una evaluacion en marcha, por ejemplo de otra ventana.
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'bifusion3d|mambalite3d' }) { exit 7 }"
if %errorlevel% equ 7 (
  echo.
  echo  Ya hay un entrenamiento o una evaluacion en marcha: python con bifusion3d o mambalite3d.
  echo  Si es de otra ventana del ejecutor, deja esa abierta y cierra esta.
  echo  Si no hay otra ventana, quedo colgado: terminalo en el Administrador de tareas,
  echo  proceso python.exe, y abre runner.bat de nuevo.
  echo.
  pause
  exit /b 1
)
rem Una sola instancia: el candado jobs\runner.lock queda abierto mientras corre :principal.
2>nul (
  9>"jobs\runner.lock" (
    call :principal
  )
) || (
  echo.
  echo  Ya hay otro ejecutor abierto en esta carpeta: no se abre un segundo.
  echo  Cierra esta ventana.
  echo.
  pause
  exit /b 1
)
exit /b 0

:principal
echo ==================================================================
echo  Ejecutor de trabajos de modelos\bifusion
echo  Ejecuta, uno a la vez y en orden, los .bat de jobs\pending y guarda
echo  su salida en jobs\logs. Cierra esta ventana para detenerlo.
echo  Si se corto (apagon, reinicio, ventana cerrada), al abrirlo de nuevo
echo  el trabajo a medias vuelve a la cola y continua con --resume.
echo  No lo abras a la vez que el ejecutor de modelos\mamba (comparten la GPU).
echo ==================================================================
for %%F in ("jobs\running\*.bat") do (
    echo [!date! !time!] %%~nF quedo a medias: vuelve a la cola
    >>"jobs\logs\%%~nF.log" echo [!date! !time!] ejecutor reiniciado: %%~nF vuelve a la cola y continua
    move /y "%%~fF" "jobs\pending\%%~nxF" >nul
)
:loop
>"jobs\runner_heartbeat.txt" echo %date% %time%
for %%F in ("jobs\pending\*.bat") do (
    set "JOB=%%~nF"
    echo [!date! !time!] Iniciando !JOB!
    move /y "%%~fF" "jobs\running\!JOB!.bat" >nul
    >"jobs\logs\!JOB!.started" echo !date! !time!
    >>"jobs\logs\!JOB!.log" echo ===== [!date! !time!] inicio de !JOB! =====
    cmd /d /c jobs\running\!JOB!.bat >> "jobs\logs\!JOB!.log" 2>&1
    set "RC=!errorlevel!"
    >"jobs\logs\!JOB!.exit" echo !RC!
    move /y "jobs\running\!JOB!.bat" "jobs\done\!JOB!.bat" >nul
    echo [!date! !time!] Terminado !JOB! con codigo !RC!
)
ping -n 6 127.0.0.1 >nul
goto loop
