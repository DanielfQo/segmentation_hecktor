@echo off
echo === arranque ===
powershell -NoProfile -Command "'Ultimo arranque: ' + (Get-CimInstance Win32_OperatingSystem).LastBootUpTime"
echo === eventos de energia y arranque desde el 11/10 18:00 ===
powershell -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'2026-10-11T18:00:00'; Id=1,12,13,41,42,107,109,506,507,1074,6005,6006,6008} -ErrorAction SilentlyContinue | Where-Object { $_.ProviderName -notin 'Microsoft-Windows-FilterManager','Microsoft-Windows-IsolatedUserMode','VBoxNetLwf' } | Select-Object TimeCreated,Id,ProviderName,@{n='Mensaje';e={($_.Message -split [char]10)[0]}} | Format-Table -AutoSize -Wrap | Out-String -Width 250"
echo === motivo de cada suspension (evento 42) ===
powershell -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'2026-10-11T18:00:00'; Id=42} -ErrorAction SilentlyContinue | Select-Object TimeCreated,Message | Format-List | Out-String -Width 250"
echo === errores de aplicaciones desde el 11/10 18:00 ===
powershell -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=[datetime]'2026-10-11T18:00:00'; Level=1,2} -ErrorAction SilentlyContinue | Select-Object -First 40 TimeCreated,Id,ProviderName,@{n='Mensaje';e={($_.Message -split [char]10)[0]}} | Format-Table -AutoSize -Wrap | Out-String -Width 250"
echo === consolas y procesos relacionados ===
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'cmd.exe','python.exe','conhost.exe','OpenConsole.exe','WindowsTerminal.exe' } | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | Format-List | Out-String -Width 250"
exit /b 0
