@echo off
echo === procesos python y cmd: si el entrenamiento sigue vivo y cuantos ejecutores hay ===
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','cmd.exe' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | Format-List"
echo === GPU ===
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
echo === arranque, energia y bateria ===
powershell -NoProfile -Command "'Ultimo arranque: ' + (Get-CimInstance Win32_OperatingSystem).LastBootUpTime; Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | Format-List; Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).Date; Id=1,12,13,41,42,107,506,507,1074,6005,6006,6008} -ErrorAction SilentlyContinue | Select-Object TimeCreated,Id,ProviderName,@{n='Mensaje';e={($_.Message -split [char]10)[0]}} | Format-Table -AutoSize -Wrap | Out-String -Width 250"
exit /b 0
