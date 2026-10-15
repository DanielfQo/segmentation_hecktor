@echo off
echo === procesos python y cmd ===
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','cmd.exe' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | Format-List"
echo === GPU ===
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
echo === arranque y bateria ===
powershell -NoProfile -Command "'Ultimo arranque: ' + (Get-CimInstance Win32_OperatingSystem).LastBootUpTime; Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | Format-List"
echo === eventos de energia desde ayer a las 14:00 ===
powershell -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).Date.AddHours(-10); Id=1,12,13,41,42,107,506,507,1074,6005,6006,6008} -ErrorAction SilentlyContinue | Where-Object { $_.ProviderName -notin 'Microsoft-Windows-FilterManager','Microsoft-Windows-IsolatedUserMode','VBoxNetLwf' } | Select-Object TimeCreated,Id,ProviderName,@{n='Mensaje';e={($_.Message -split [char]10)[0]}} | Format-Table -AutoSize -Wrap | Out-String -Width 250"
echo === motivo de cada suspension desde ayer a las 14:00 (evento 42) ===
powershell -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).Date.AddHours(-10); Id=42} -ErrorAction SilentlyContinue | Select-Object TimeCreated,Message | Format-List | Out-String -Width 250"
echo === configuracion de suspension (solo lectura) ===
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
powercfg /query SCHEME_CURRENT SUB_BUTTONS LIDACTION
exit /b 0
