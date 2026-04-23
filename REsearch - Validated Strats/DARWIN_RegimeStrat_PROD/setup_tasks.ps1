# Task 1
$action1 = New-ScheduledTaskAction -Execute "python.exe" -Argument "C:\REGIME_DARWIN_PROD\execute_trade.py" -WorkingDirectory "C:\REGIME_DARWIN_PROD"
$trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday,Wednesday,Thursday,Friday,Saturday -At "02:02"
$trigger1.StartBoundary = [DateTime]::Parse($trigger1.StartBoundary).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss") + "Z"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew -StartWhenAvailable
$task1 = Register-ScheduledTask -TaskName "DARWIN_RegimeStrat" -Action $action1 -Trigger $trigger1 -Settings $settings -RunLevel Highest -Force
if ($task1) { Write-Host "[OK] DARWIN_RegimeStrat registered" -ForegroundColor Green } else { Write-Host "[FAILED] DARWIN_RegimeStrat" -ForegroundColor Red }

# Task 2
$action2 = New-ScheduledTaskAction -Execute "python.exe" -Argument "C:\REGIME_DARWIN_PROD\daily_recap.py" -WorkingDirectory "C:\REGIME_DARWIN_PROD"
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday,Wednesday,Thursday,Friday,Saturday -At "04:30"
$trigger2.StartBoundary = [DateTime]::Parse($trigger2.StartBoundary).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss") + "Z"
$task2 = Register-ScheduledTask -TaskName "DARWIN_DailyRecap" -Action $action2 -Trigger $trigger2 -Settings $settings -RunLevel Highest -Force
if ($task2) { Write-Host "[OK] DARWIN_DailyRecap registered" -ForegroundColor Green } else { Write-Host "[FAILED] DARWIN_DailyRecap" -ForegroundColor Red }

# Task 3
$action3 = New-ScheduledTaskAction -Execute "C:\Program Files\Darwinex MT5\terminal64.exe"
$trigger3 = New-ScheduledTaskTrigger -AtStartup
$task3 = Register-ScheduledTask -TaskName "MT5_Autostart" -Action $action3 -Trigger $trigger3 -RunLevel Highest -Force
if ($task3) { Write-Host "[OK] MT5_Autostart registered" -ForegroundColor Green } else { Write-Host "[FAILED] MT5_Autostart" -ForegroundColor Red }

# Summary
Write-Host "---- Registered Tasks ----" -ForegroundColor Cyan
Get-ScheduledTask | Where-Object { $_.TaskName -like 'DARWIN*' -or $_.TaskName -eq 'MT5_Autostart' } | Select-Object TaskName, State | Format-Table -AutoSize
Write-Host "Press any key to close..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")