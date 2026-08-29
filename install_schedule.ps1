# Registers a Windows Scheduled Task that starts the bot each weekday morning.
#
# Run this yourself from an ELEVATED PowerShell (Run as Administrator):
#     powershell -ExecutionPolicy Bypass -File install_schedule.ps1
#
# It only REGISTERS the schedule. It does not start trading now.
# Remove it later with:  Unregister-ScheduledTask -TaskName "ClaudeDaytrader"
#
# NOTE: the trigger time below is LOCAL time. It is set to 09:25 to give the
# bot a few minutes before the 09:30 ET open. If this machine is not on
# Eastern time, change -At accordingly.

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName   = "ClaudeDaytrader"

Write-Host "Project directory: $ProjectDir"

# Resolve python the same way daytrade.bat does.
$Python = "python"
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) { $Python = $VenvPython }
Write-Host "Python: $Python"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Existing task found - removing it first."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "main.py" `
    -WorkingDirectory $ProjectDir

# Weekdays only - the market is closed on weekends anyway, but this avoids
# two pointless days of "market closed" heartbeats.
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At 9:25AM

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Claude Daytrader - starts the paper-trading loop each weekday morning." `
    | Out-Null

Write-Host ""
Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "  Runs weekdays at 9:25 AM local time."
Write-Host "  Restarts up to 3 times if it dies, 5 minutes apart."
Write-Host "  Stops after 12 hours so it cannot linger overnight."
Write-Host ""
Write-Host "Verify with : Get-ScheduledTask -TaskName $TaskName"
Write-Host "Run now with: Start-ScheduledTask -TaskName $TaskName"
Write-Host "Remove with : Unregister-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "IMPORTANT: this places real paper orders on every weekday it runs."
Write-Host "Check logs\bot.log and logs\decisions.jsonl to confirm it is alive."
