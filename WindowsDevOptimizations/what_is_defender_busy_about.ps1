# Set log path
$logPath = "C:\av_scan_log.etl"

# Clear old log if exists
if (Test-Path $logPath) { Remove-Item $logPath -Force }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "1. STARTING PERFORMANCE RECORDING..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Recording has started in the background." -ForegroundColor Yellow
Write-Host "--> Action required: Switch to Visual Studio and START your C++ build now! <--`n" -ForegroundColor Green

# Start the recording using correct parameter syntax
$recordTask = Start-Job -ScriptBlock { New-MpPerformanceRecording -RecordTo $using:logPath }

# Count down time for compilation trace
$seconds = 25
while ($seconds -gt 0) {
    Write-Progress -Activity "Recording Windows Defender activity. Keep building your project..." -Status "$seconds seconds remaining" -PercentComplete (($25-$seconds)/25*100)
    Start-Sleep -Seconds 1
    $seconds--
}
Write-Progress -Activity "Recording Windows Defender activity" -Completed

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "2. STOPPING RECORDING AND GENERATING REPORT..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Stop background job and hit Enter to finalize recording
$recordTask | Stop-Job
$recordTask | Remove-Job
[Microsoft.VisualBasic.Interaction]::AppActivate((Get-Process -Id $PID).Id) 2>$null
[System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
Start-Sleep -Seconds 2

# Check if log was created successfully
if (Test-Path $logPath) {
    Write-Host "Log saved to $logPath. Analyzing data...`n" -ForegroundColor Green
    
    # Display Top 20 heaviest files for the antivirus
    Get-MpPerformanceReport -Path $logPath -TopFiles 20 | Format-Table -Property Duration, Count, Path -AutoSize
} else {
    Write-Host "[ERROR] Failed to create log file at $logPath. Please run the script again." -ForegroundColor Red
}