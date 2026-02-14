# Theta Terminal v3 Launcher
$ErrorActionPreference = "Continue"
$JAVA = "C:\Program Files\Eclipse Adoptium\jdk-21.0.10.7-hotspot\bin\java.exe"
$JAR = Join-Path $PSScriptRoot "ThetaTerminalv3.jar"

# Kill any existing Theta Terminal process
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*ThetaTerminalv3*" } | ForEach-Object {
    Write-Host "Killing existing Theta Terminal (PID $($_.ProcessId))..."
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

# Verify Java 21
Write-Host "Java version:" -ForegroundColor Cyan
& $JAVA -version 2>&1 | ForEach-Object { "$_" } | Write-Host
Write-Host ""

# Ensure log dir exists
if (-not (Test-Path "C:\tmp")) { New-Item -ItemType Directory -Path "C:\tmp" | Out-Null }

Write-Host "Starting Theta Terminal v3..." -ForegroundColor Green
Write-Host "JAR: $JAR"
Write-Host "Logs: C:\tmp"
Write-Host ""

Set-Location $PSScriptRoot
& $JAVA -jar $JAR

Write-Host ""
Write-Host "Theta Terminal exited." -ForegroundColor Yellow
Read-Host "Press Enter to close"
