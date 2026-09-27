"""Is the bot actually running? Read-only, places no orders.

Written after a silent two-day outage: the process list showed a `main.py`
that belonged to a different project, the decision log looked plausible
because it still held Thursday's data, and nothing anywhere said "I am not
running." This answers that question directly.

    python health_check.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config

HERE = Path(__file__).parent.resolve()
OK, WARN, BAD = "OK  ", "WARN", "BAD "
issues: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f"  [{status}] {label:<30} {detail}")
    if status == BAD:
        issues.append(label)


def age_str(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    mins = delta.total_seconds() / 60
    if mins < 60:
        return f"{mins:.0f} min ago"
    if mins < 60 * 24:
        return f"{mins / 60:.1f} hours ago"
    return f"{mins / 1440:.1f} days ago"


def main() -> None:
    print("=" * 66)
    print("HEALTH CHECK")
    print("=" * 66)
    now_local = datetime.now()
    print(f"  now: {now_local:%Y-%m-%d %H:%M:%S} local\n")

    # --- 1. Is a process running OUR main.py from THIS directory? ----------
    # A bare process-name match is what caused the outage to go unnoticed:
    # several unrelated projects on this machine also have a main.py.
    found_ours = False
    try:
        import subprocess
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine,CreationDate | ConvertTo-Json -Depth 3"],
            capture_output=True, text=True, timeout=60,
        )
        procs = json.loads(ps.stdout) if ps.stdout.strip() else []
        if isinstance(procs, dict):
            procs = [procs]
        candidates = [p for p in procs if p.get("CommandLine") and "main.py" in p["CommandLine"]]
        if not candidates:
            line(BAD, "bot process", "no main.py process found at all")
        else:
            line(WARN, "main.py processes", f"{len(candidates)} found — verifying which are ours")
            print("         (other projects on this machine also have a main.py)")
    except Exception as exc:
        line(WARN, "process check", f"could not enumerate: {type(exc).__name__}")

    # --- 1b. The instance lock says definitively whether a bot owns the state
    # files. Process-name matching cannot: other projects on this machine have
    # a main.py too, which is why check 1 can only WARN.
    try:
        from instance_lock import InstanceLock
        lk = InstanceLock()
        holder = lk.holder()
        if holder:
            line(OK, "instance lock",
                 f"held by pid {holder['pid']}, since {holder.get('started', '?')}")
        elif lk.path.exists():
            try:
                stale_pid = json.loads(lk.path.read_text(encoding="utf-8")).get("pid")
            except Exception:                              # noqa: BLE001
                stale_pid = "?"
            line(WARN, "instance lock",
                 f"stale (pid {stale_pid} is gone) — a restart will clear it")
        else:
            line(OK, "instance lock", "free — no bot owns the state files")
    except Exception as exc:                               # noqa: BLE001
        line(WARN, "instance lock", f"could not read: {type(exc).__name__}")

    # --- 2. Decision log freshness — the authoritative signal --------------
    # If the loop is alive during market hours this file updates every cycle.
    if not config.DECISION_LOG.exists():
        line(BAD, "decision log", "missing — bot has never completed a cycle")
    else:
        lines = config.DECISION_LOG.read_text(encoding="utf-8").strip().splitlines()
        last = json.loads(lines[-1])
        last_ts = datetime.fromisoformat(last["ts"])
        age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
        detail = f"{len(lines)} cycles, last {age_str(last_ts)}"
        # Allow a generous multiple of the cycle time before calling it dead.
        if age_min <= config.CYCLE_SECONDS / 60 * 3:
            line(OK, "decision log", detail)
        elif age_min <= 60 * 20:
            line(WARN, "decision log", detail + "  (stale — market closed?)")
        else:
            line(BAD, "decision log", detail + "  (bot is NOT running)")
            found_ours = False

    # --- 3. Process log ----------------------------------------------------
    bot_log = config.LOG_DIR / "bot.log"
    if not bot_log.exists():
        line(WARN, "bot.log", "missing — start the bot to create it")
    else:
        mtime = datetime.fromtimestamp(bot_log.stat().st_mtime, timezone.utc)
        tail = bot_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        line(OK if (datetime.now(timezone.utc) - mtime) < timedelta(hours=1) else WARN,
             "bot.log", f"last write {age_str(mtime)}")
        for t in tail:
            print(f"         | {t[:90]}")

    # --- 4. Kill switch ----------------------------------------------------
    if config.KILL_SWITCH_FILE.exists():
        line(BAD, "kill switch", "ON — every order is being refused")
    else:
        line(OK, "kill switch", "off")

    # --- 5. Broker ---------------------------------------------------------
    try:
        from executor import Executor
        ex = Executor()
        acct = ex.account_state()
        if not acct:
            line(BAD, "broker", "account fetch failed")
        else:
            line(OK, "broker", f"equity ${acct['equity']:,.2f}  mode="
                               f"{'PAPER' if acct['paper'] else 'LIVE'}")
            mins = ex.minutes_to_close()
            if mins is None:
                line(WARN, "market clock", "unavailable")
            elif mins > 0:
                line(OK, "market", f"OPEN — {mins:.0f} min to close")
            else:
                line(OK, "market", "closed")
            pos = ex.open_positions()
            line(OK, "positions", ", ".join(p["symbol"] for p in pos) if pos else "flat")
    except Exception as exc:
        line(BAD, "broker", f"{type(exc).__name__}: {str(exc)[:50]}")

    # --- verdict -----------------------------------------------------------
    print("\n" + "=" * 66)
    if issues:
        print("VERDICT: problems found -> " + ", ".join(issues))
        print("\nTo start the bot:  python main.py")
        print("Auto-start on logon: install_startup.ps1 (see its header)")
        sys.exit(1)
    print("VERDICT: healthy")


if __name__ == "__main__":
    main()
