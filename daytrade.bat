@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Claude Daytrader

:: Prefer the project venv if one exists, else fall back to system python.
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

where %PY% >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH.
    echo Install Python or activate your venv, then re-run this file.
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo                    CLAUDE DAYTRADER
echo ============================================================
echo   Folder : %CD%
for /f "delims=" %%A in ('%PY% -c "import config;print(config.DECIDER_BACKEND)" 2^>nul') do set "BACKEND=%%A"
for /f "delims=" %%A in ('%PY% -c "import config;print('OBSERVE (no orders)' if config.OBSERVE_MODE else ('PAPER' if config.ALPACA_PAPER else 'LIVE'))" 2^>nul') do set "MODE=%%A"
if not defined BACKEND set "BACKEND=?"
if not defined MODE set "MODE=?"
echo   Backend: !BACKEND!        Mode: !MODE!
if exist "KILL_SWITCH" (
    echo   KILL SWITCH: **ON**  - no orders will be placed
) else (
    echo   Kill switch: off
)
echo ============================================================
echo.
echo   SAFE - no orders, no API cost
echo     1. Connection check       read-only, verifies keys + data
echo     2. Risk gate tests        21 adversarial tests
echo     3. Dry run                full chain, fake data, mock broker
echo     4. Paper P/L report       actual fills from Alpaca
echo     5. Calibration report     stats over the decision log
echo     6. Insider feed check     SEC Form 4 for the watchlist
echo.
echo   ANALYSIS - no orders, reads what already happened
echo    13. Edge report           win rate, expectancy, worst trades
echo    14. Shadow comparison     head-to-head strategy ranking
echo.
echo   TRADING - places real paper orders (unless OBSERVE_MODE is on)
echo     7. Run one cycle
echo     8. Run the loop           until you press Ctrl+C
echo.
echo   UTILITIES
echo     9. Toggle kill switch
echo    10. Tail decision log
echo    11. Open folder
echo    12. Health check          is the bot actually running?
echo.
echo     0. Exit
echo.
set "CHOICE="
set /p "CHOICE=Select: "

if "%CHOICE%"=="1"  goto check
if "%CHOICE%"=="2"  goto gatetests
if "%CHOICE%"=="3"  goto dryrun
if "%CHOICE%"=="4"  goto report
if "%CHOICE%"=="5"  goto calib
if "%CHOICE%"=="6"  goto insider
if "%CHOICE%"=="7"  goto once
if "%CHOICE%"=="8"  goto loop
if "%CHOICE%"=="9"  goto killswitch
if "%CHOICE%"=="10" goto taillog
if "%CHOICE%"=="11" goto openfolder
if "%CHOICE%"=="12" goto health
if "%CHOICE%"=="13" goto edge
if "%CHOICE%"=="14" goto shadowrep
if "%CHOICE%"=="0"  exit /b 0
goto menu

:check
cls
%PY% check_connection.py
goto done

:gatetests
cls
%PY% test_gate.py
goto done

:dryrun
cls
set "N="
set /p "N=How many cycles? [5]: "
if "!N!"=="" set "N=5"
%PY% dryrun.py --cycles !N!
goto done

:report
cls
%PY% paper_report.py
goto done

:calib
cls
%PY% calibration.py
goto done

:insider
cls
echo Querying SEC EDGAR - this can take a moment on a cold cache...
%PY% insider_feed.py
goto done

:once
cls
if exist "KILL_SWITCH" (
    echo KILL SWITCH is ON - no orders will be placed.
    echo Use option 9 to clear it first.
    echo.
)
echo Running ONE cycle. This can place real paper orders.
echo.
set "OK="
set /p "OK=Type y to continue: "
if /i not "!OK!"=="y" goto menu
%PY% main.py --once
goto done

:loop
cls
if exist "KILL_SWITCH" (
    echo KILL SWITCH is ON - no orders will be placed.
    echo Use option 9 to clear it first.
    echo.
)
echo Running the LOOP. This places real paper orders during market hours.
echo Press Ctrl+C in this window to stop it.
echo.
set "OK="
set /p "OK=Type y to continue: "
if /i not "!OK!"=="y" goto menu
%PY% main.py
goto done

:killswitch
if exist "KILL_SWITCH" (
    del "KILL_SWITCH"
    echo.
    echo Kill switch CLEARED - trading is allowed again.
) else (
    type nul > "KILL_SWITCH"
    echo.
    echo Kill switch ON - the risk gate will now refuse every order.
    echo A running loop picks this up on its next cycle. No restart needed.
)
echo.
pause
goto menu

:taillog
cls
if not exist "logs\decisions.jsonl" (
    echo No decision log yet. Run a cycle first.
    goto done
)
echo Last 15 cycles:
echo.
powershell -NoProfile -Command "Get-Content 'logs\decisions.jsonl' -Tail 15 | ForEach-Object { $r = $_ | ConvertFrom-Json; $t = ([datetime]$r.ts).ToLocalTime().ToString('MM-dd HH:mm:ss'); $d = if ($r.decisions.Count) { ($r.decisions | ForEach-Object { $_.action.ToUpper() + ' ' + $_.symbol + $(if ($_.gate_approved) { ' OK' } else { ' rejected' }) }) -join ', ' } else { '-' }; '{0}  [{1}]  {2}' -f $t, $r.backend, $d }"
goto done

:openfolder
start "" "%CD%"
goto menu

:edge
cls
%PY% edge_report.py --days 7
goto done

:shadowrep
cls
%PY% shadow_report.py
goto done

:health
cls
%PY% health_check.py
goto done

:done
echo.
pause
goto menu
