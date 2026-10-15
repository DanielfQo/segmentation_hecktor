@echo off
echo === bateria ===
powershell -NoProfile -Command "Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | Format-List"
echo === energia desde ayer a las 14:00: cambios de cargador (105), reposo moderno (506, 507) y suspension (42, 107) ===
powershell -NoProfile -Command "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; StartTime=(Get-Date).Date.AddHours(-10); Id=42,105,107,506,507} -ErrorAction SilentlyContinue | Sort-Object TimeCreated | Select-Object TimeCreated,Id,@{n='Mensaje';e={($_.Message -split [char]10 | Where-Object { $_.Trim() }) -join ' / '}},@{n='Datos';e={($_.Properties | ForEach-Object { $_.Value }) -join ', '}} | Format-Table -AutoSize -Wrap | Out-String -Width 250"
exit /b 0
