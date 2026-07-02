# Nvidia telemetry permanent turn off

reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options\NvTelemetryContainer.exe" /v Debugger /t REG_SZ /d "%windir%\System32\taskkill.exe" /f

# What Windows antimalware service scans
New-MpPerformanceRecording -RecordTo "C:\av_scan_log.etl"
...
Wait a little and then press "Enter"
...
Get-MpPerformanceReport -Path "C:\av_scan_log.etl" -TopFiles 20 | Format-Table -AutoSize
