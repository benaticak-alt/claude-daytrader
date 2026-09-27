"""Tests for the single-instance lock.

The failure this prevents is silent: two bots writing the same shadow books
and decision log corrupt the record rather than crashing. The failure it must
NOT cause is also important — a crashed bot leaves a lock behind, and if that
blocked restarts the bot would stay down after every crash.

    python test_lock.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from instance_lock import AlreadyRunning, InstanceLock, process_alive

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


tmp = Path(tempfile.mkdtemp())


def lock(name: str = "bot.lock", **kw) -> InstanceLock:
    return InstanceLock(path=tmp / name, **kw)


# --- basic acquire / release ------------------------------------------------
lk = lock("a.lock")
lk.acquire()
check((tmp / "a.lock").exists(), "acquire writes the lock file")
rec = json.loads((tmp / "a.lock").read_text(encoding="utf-8"))
check(rec["pid"] == os.getpid(), "lock records our pid", str(rec.get("pid")))
check("python" in (rec.get("exe") or "").lower(), "lock records the interpreter",
      rec.get("exe", ""))
lk.release()
check(not (tmp / "a.lock").exists(), "release removes the lock file")
lk.release()          # must be idempotent, not raise
check(True, "release is idempotent")

# --- a live holder blocks a second instance ---------------------------------
# Our own pid is trivially alive, so a lock naming a DIFFERENT live python
# process is what we need. Spawn a real one that outlives the check.
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
try:
    # A real bot writes its lock when it starts, i.e. after the last boot.
    (tmp / "b.lock").write_text(json.dumps(
        {"pid": child.pid, "exe": sys.executable,
         "started": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    blocked = lock("b.lock")
    try:
        blocked.acquire()
        check(False, "a live holder blocks a second instance", "it did NOT block")
    except AlreadyRunning as e:
        check("already running" in str(e).lower(), "a live holder blocks a second instance",
              f"pid {child.pid}")
        check(str(child.pid) in str(e) and "--force" in str(e),
              "the refusal names the pid and the escape hatch")

    # --force takes it over deliberately.
    forced = lock("b.lock", force=True)
    forced.acquire()
    check(json.loads((tmp / "b.lock").read_text(encoding="utf-8"))["pid"] == os.getpid(),
          "--force takes the lock over")
    forced.release()
finally:
    child.terminate()
    child.wait(timeout=10)

# --- a dead holder is stale and must NOT block ------------------------------
# This is the state on disk after the 2026-09-23 crash: a lock left behind by
# a process that no longer exists. Blocking here would keep the bot down.
dead_pid = child.pid                      # just terminated above
(tmp / "c.lock").write_text(json.dumps(
    {"pid": dead_pid, "exe": sys.executable, "started": "2026-09-23T19:12:00"}),
    encoding="utf-8")
stale = lock("c.lock")
stale.acquire()
check(json.loads((tmp / "c.lock").read_text(encoding="utf-8"))["pid"] == os.getpid(),
      "a crashed bot's lock is stale and is taken over",
      f"dead pid {dead_pid} did not block")
stale.release()

# --- a recycled pid belonging to a non-python process does not block --------
(tmp / "d.lock").write_text(json.dumps(
    {"pid": 4, "exe": "C:\\\\Windows\\\\System32\\\\svchost.exe",
     "started": "2026-01-01T00:00:00"}), encoding="utf-8")
recycled = lock("d.lock")
recycled.acquire()          # pid 4 is System on Windows: alive but not python
check(json.loads((tmp / "d.lock").read_text(encoding="utf-8"))["pid"] == os.getpid(),
      "a recycled pid that is not python does not block")
recycled.release()

# --- an uninspectable pid whose lock predates the boot is stale -------------
# pid 4 is the Windows System process: alive, but we are not allowed to read
# its image. With a pre-boot timestamp it must NOT block, or the bot would
# refuse to start after a reboot — exactly when auto-start matters.
(tmp / "d2.lock").write_text(json.dumps(
    {"pid": 4, "exe": sys.executable, "started": "2020-01-01T00:00:00"}),
    encoding="utf-8")
preboot = lock("d2.lock")
preboot.acquire()
check(json.loads((tmp / "d2.lock").read_text(encoding="utf-8"))["pid"] == os.getpid(),
      "uninspectable pid with a pre-boot lock does not block a restart")
preboot.release()

# --- corrupt / empty lock files are treated as stale, not fatal -------------
for name, body in (("e.lock", "not json at all"), ("f.lock", ""),
                   ("g.lock", '{"pid": "garbage"}')):
    (tmp / name).write_text(body, encoding="utf-8")
    lk2 = lock(name)
    try:
        lk2.acquire()
        ok = True
    except Exception as exc:                     # noqa: BLE001
        ok = False
        detail = repr(exc)
    check(ok, f"unreadable lock ({body[:12] or 'empty'!r}) is treated as stale",
          "" if ok else detail)
    lk2.release()

# --- release must not steal a lock another process has since taken ----------
lk3 = lock("h.lock")
lk3.acquire()
(tmp / "h.lock").write_text(json.dumps(
    {"pid": 999_999, "exe": sys.executable, "started": "2026-01-01T00:00:00"}),
    encoding="utf-8")
lk3.release()
check((tmp / "h.lock").exists(),
      "release leaves a lock that now belongs to someone else")

# --- re-entrant: our own pid never blocks us ------------------------------
lk4 = lock("i.lock")
lk4.acquire()
again = lock("i.lock")
try:
    again.acquire()
    check(True, "our own pid does not block us (re-entrant)")
except AlreadyRunning:
    check(False, "our own pid does not block us (re-entrant)", "it blocked")
again.release()

# --- process_alive sanity --------------------------------------------------
alive, exe = process_alive(os.getpid())
check(alive and "python" in exe.lower(), "process_alive finds our own process", exe)
check(not process_alive(999_999)[0], "process_alive reports a missing pid as dead")
check(not process_alive(0)[0], "process_alive rejects pid 0")

# --- context manager ------------------------------------------------------
with lock("j.lock"):
    inside = (tmp / "j.lock").exists()
check(inside and not (tmp / "j.lock").exists(),
      "context manager acquires and releases")

width = max(len(l) for _, l, _ in results)
failures = sum(1 for ok, _, _ in results if not ok)
for ok, label, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")
print(f"\n{len(results) - failures}/{len(results)} passed")
sys.exit(1 if failures else 0)
