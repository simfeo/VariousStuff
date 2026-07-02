# List of processes to exclude
$processes = @(
    "devenv.exe", "cl.exe", "clang.exe", "clang++.exe", "clang-cl.exe",
    "link.exe", "lld-link.exe", "msbuild.exe", "vcpkg.exe", "ninja.exe"
)

# List of standard development folders to exclude
# Note: Edit the first line if your projects are located elsewhere
$folders = @(
    "$env:USERPROFILE\source\repos",                           # Default Visual Studio projects folder
    "$env:LOCALAPPDATA\Microsoft\VisualStudio",                 # Visual Studio cache & IntelliSense
    "C:\Program Files\Microsoft Visual Studio",                 # IDE and MSVC compiler files
    "C:\Program Files (x86)\Windows Kits"                       # Windows SDK headers and libraries
)

Write-Host "Adding processes to Windows Defender exclusions..." -ForegroundColor Cyan
foreach ($process in $processes) {
    try {
        Add-MpPreference -ExclusionProcess $process -ErrorAction Stop
        Write-Host "[SUCCESS] Added process: $process" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to add process $process" -ForegroundColor Red
    }
}

Write-Host "`nAdding folders to Windows Defender exclusions..." -ForegroundColor Cyan
foreach ($folder in $folders) {
    if (Test-Path $folder) {
        try {
            Add-MpPreference -ExclusionPath $folder -ErrorAction Stop
            Write-Host "[SUCCESS] Added folder: $folder" -ForegroundColor Green
        } catch {
            Write-Host "[ERROR] Failed to add folder $folder" -ForegroundColor Red
        }
    } else {
        Write-Host "[SKIP] Folder not found: $folder" -ForegroundColor Yellow
    }
}

# Verification step: fetch and display all exclusions
Write-Host "`n--- VERIFICATION ---" -ForegroundColor Cyan
Write-Host "Excluded Processes:" -ForegroundColor Cyan
(Get-MpPreference).ExclusionProcess | ForEach-Object { Write-Host " -> $_" -ForegroundColor Yellow }

Write-Host "`nExcluded Folders:" -ForegroundColor Cyan
(Get-MpPreference).ExclusionPath | ForEach-Object { Write-Host " -> $_" -ForegroundColor Yellow }

Write-Host "`nDone! No reboot required." -ForegroundColor Cyan
