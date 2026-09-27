# Make the bot start automatically.
#
#   powershell -ExecutionPolicy Bypass -File install_startup.ps1
#       Registers a Scheduled Task (needs an elevated prompt). Preferred:
#       Task Scheduler restarts the bot if it dies, which plain startup
#       shortcuts cannot do. This project has had two silent multi-day
#       outages; restart-on-failure is the point.
#
#   powershell -ExecutionPolicy Bypass -File install_startup.ps1 -NoAdmin
#       Drops a shortcut in the user's Startup folder instead. No admin
#       needed, starts on logon, but no restart-on-failure.
#
#   powershell -ExecutionPolicy Bypass -File install_startup.ps1 -Remove
#       Removes both.
#
# Triggers registered: at logon (delayed 1 minute so the network is up) AND
# each weekday at 09:25 local. Two triggers would normally risk two bots
# running at once; bot.lock makes the second exit immediately with a clear
# message, so the redundancy is free. That redundancy is deliberate: a logon
# trigger alone never revives a bot that dies mid-week on a machine that
# stays powered on.

[CmdletBinding()]
param(
    [switch]$NoAdmin,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName   = "ClaudeDaytrader"
$Startup    = [Environment]::GetFolderPath("Startup")
$LnkPath    = Join-Path $Startup "ClaudeDaytrader.lnk"

# Resolve python to an ABSOLUTE path that can actually import the project's
# dependencies. "python" on PATH is not good enough for a boot-time task: on
# this machine it resolves to a WindowsApps app-execution alias, which is a
# per-user shim that can break when the Store updates or the alias is toggled
# off in Settings - and a boot task that fails leaves no window to see it in.
# So probe candidates and keep the first that really works.
function Resolve-Python {
    $candidates = @()
    $venv = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (Test-Path $venv) { $candidates += $venv }
    # Direct installs first, app-execution aliases last.
    foreach ($c in (Get-Command python -All -ErrorAction SilentlyContinue)) {
        if ($c.Source -notlike "*\WindowsApps\*") { $candidates += $c.Source }
    }
    foreach ($c in (Get-Command python -All -ErrorAction SilentlyContinue)) {
        if ($c.Source -like "*\WindowsApps\*") { $candidates += $c.Source }
    }
    foreach ($exe in ($candidates | Select-Object -Unique)) {
        try {
            & $exe -c "import alpaca, pandas, dotenv" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return $exe }
        } catch { }
    }
    return $null
}

$Python = Resolve-Python
if (-not $Python) {
    Write-Host "Could not find a python that can import the project's dependencies." -ForegroundColor Red
    Write-Host "Try:  pip install -r requirements.txt"
    exit 1
}

Write-Host "Project : $ProjectDir"
Write-Host "Python  : $Python"
Write-Host ""

# ---------------------------------------------------------------- remove ----
if ($Remove) {
    $did = $false
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Yellow
        $did = $true
    }
    if (Test-Path $LnkPath) {
        Remove-Item $LnkPath
        Write-Host "Removed startup shortcut." -ForegroundColor Yellow
        $did = $true
    }
    if (-not $did) { Write-Host "Nothing to remove." }
    Write-Host "A bot already running is unaffected - stop it with Ctrl+C in its window."
    return
}

# ------------------------------------------------- startup-folder variant ---
if ($NoAdmin) {
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($LnkPath)
    $lnk.TargetPath = $Python
    $lnk.Arguments = "main.py"
    $lnk.WorkingDirectory = $ProjectDir
    $lnk.Description = "Claude Daytrader - paper-trading loop"
    $lnk.WindowStyle = 7            # minimised, so it does not steal focus
    $lnk.Save()
    Write-Host "Created $LnkPath" -ForegroundColor Green
    Write-Host "  Starts on logon, minimised."
    Write-Host "  NOTE: no restart-on-failure. If the bot dies it stays down"
    Write-Host "        until the next logon. Run health_check.py to check on it."
    return
}

# ------------------------------------------------------ scheduled task ------
$admin = ([Security.Principal.WindowsPrincipal] `
          [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "This needs an ELEVATED PowerShell (Run as Administrator)." -ForegroundColor Red
    Write-Host "Either re-run it elevated, or use the no-admin variant:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File install_startup.ps1 -NoAdmin"
    exit 1
}

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Existing task found - replacing it."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute $Python -Argument "main.py" -WorkingDirectory $ProjectDir

# Logon trigger, delayed a minute: at logon the network stack is often not
# ready, and the first thing the bot does is talk to Alpaca and the SEC.
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$logon.Delay = "PT1M"

# Weekday morning trigger as the revival path for a bot that died mid-week.
$morning = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 9:25AM

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 0)   # unlimited: the loop sleeps
                                                    # itself when the market is
                                                    # closed, and the insider
                                                    # strategy holds overnight

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($logon, $morning) `
    -Settings $settings `
    -Principal $principal `
    -Description "Claude Daytrader - paper-trading loop. Starts at logon and each weekday 09:25. bot.lock prevents a second instance." `
    | Out-Null

Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "  Triggers        : at logon (+1 min), and weekdays 09:25 local"
Write-Host "  On failure      : up to 3 restarts, 5 minutes apart"
Write-Host "  Second instance : refused by bot.lock, so the two triggers cannot collide"
Write-Host "  Time limit      : none (the loop sleeps itself when the market is closed)"
Write-Host ""
Write-Host "Start it now   : Start-ScheduledTask -TaskName $TaskName"
Write-Host "Check it       : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "Is it alive?   : python health_check.py"
Write-Host "Remove         : powershell -File install_startup.ps1 -Remove"
Write-Host ""
Write-Host "Trading mode comes from .env, not from this script - check OBSERVE_MODE"
Write-Host "and ALPACA_PAPER there before assuming what it will do."
