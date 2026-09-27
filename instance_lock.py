"""Single-instance lock — only one bot may own the state files.

Two instances of main.py both write cache/shadow_books.json, the insider
deciders' state, and logs/decisions.jsonl. Last writer wins, so concurrent
runs silently corrupt the forward test rather than failing loudly. The
Polymarket bot learned this the hard way in August; this project has had no
lock at all, which was survivable only because the bot was always started by
hand. Auto-start on logon removes that protection, so the lock comes first.

Design notes that matter:

  * The lock stores the PID *and* the executable path. A PID alone is not
    enough: PIDs are recycled, and after a reboot the old number may belong to
    something unrelated. Requiring the image to still be a python process
    makes a false "already running" very unlikely.

  * A crashed bot leaves the file behind. That is fine — the next start sees a
    PID that is not alive, calls the lock stale, and takes over, logging that
    it did so. The bot died on 2026-09-23 without cleaning up; that must not
    prevent a restart.

  * A PID that is alive but uninspectable (a protected system process) is
    ambiguous. Boot time resolves it: a lock written before the current boot
    cannot be held by anything. Within one boot the lock FAILS CLOSED, because
    a bot that does not start is visible in health_check.py within minutes
    while two bots quietly corrupting each other's books is not. `--force` is
    the escape hatch and the refusal message says so.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

LOCK_PATH = Path(__file__).parent / "bot.lock"


_STILL_ACTIVE = 259


def boot_time() -> Optional[datetime]:
    """When this machine last booted, or None if it cannot be determined.

    This is the discriminator that makes auto-start-on-logon safe. A lock
    written before the current boot CANNOT be held by a live process, whatever
    its PID now belongs to — and after a reboot the recorded PID is very likely
    to have been reused by something we are not allowed to inspect (low PIDs go
    to protected system processes). Without this check the bot would refuse to
    start after exactly the event auto-start exists to handle.
    """
    try:
        if sys.platform == "win32":
            ms = ctypes.windll.kernel32.GetTickCount64()
            return datetime.now() - timedelta(milliseconds=ms)
        with open("/proc/uptime", encoding="utf-8") as f:
            return datetime.now() - timedelta(seconds=float(f.read().split()[0]))
    except Exception:                                   # noqa: BLE001
        return None


def _windows_exe(pid: int) -> tuple[bool, str]:
    """(alive, exe path) on Windows, without touching the process.

    OpenProcess succeeding is NOT proof of life. Windows keeps a process object
    queryable for as long as anything holds a handle to it, so a process that
    has already exited still opens fine — a parent shell or the scheduler is
    usually holding one. GetExitCodeProcess is the actual liveness test: only
    STILL_ACTIVE means running.
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied (error 5) means the process EXISTS but belongs to
        # someone else — that is alive-but-unknown, not dead.
        return (k32.GetLastError() == 5, "")
    try:
        code = ctypes.c_ulong()
        if k32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != _STILL_ACTIVE:
            return False, ""          # already exited, handle merely lingering
        size = ctypes.c_uint(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return True, buf.value
        return True, ""
    finally:
        k32.CloseHandle(handle)


def _posix_exe(pid: int) -> tuple[bool, str]:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, ""
    except PermissionError:
        return True, ""          # exists, not ours to inspect
    try:
        return True, os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return True, ""


def process_alive(pid: int) -> tuple[bool, str]:
    """(alive, exe path). exe path is "" when it could not be determined."""
    if pid <= 0:
        return False, ""
    try:
        if sys.platform == "win32":
            return _windows_exe(pid)
        return _posix_exe(pid)
    except Exception:                                   # noqa: BLE001
        log.exception("could not probe pid %s — treating it as alive", pid)
        return True, ""                                 # fail closed


class AlreadyRunning(RuntimeError):
    """Another live instance holds the lock."""


class InstanceLock:
    """Context manager. `with InstanceLock():` or .acquire()/.release()."""

    def __init__(self, path: Path = LOCK_PATH, force: bool = False) -> None:
        self.path = path
        self.force = force
        self._held = False

    # ------------------------------------------------------------------
    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            log.warning("lock file %s is unreadable — treating it as stale", self.path)
            return None

    def holder(self) -> Optional[dict]:
        """The live holder's record, or None if the lock is free or stale.

        Adds `alive` and `exe` to the record so callers (health_check) can
        report on it without duplicating the probing logic.
        """
        rec = self._read()
        if not rec:
            return None
        try:
            pid = int(rec.get("pid") or 0)
        except (TypeError, ValueError):
            log.warning("lock file %s has a non-numeric pid — treating it as stale",
                        self.path)
            return None
        if pid == os.getpid():
            return None                     # our own lock, re-entrant

        alive, exe = process_alive(pid)
        if not alive:
            return None

        # Direct evidence about the image beats every heuristic below.
        recorded = (rec.get("exe") or "").lower()
        current = (exe or "").lower()
        if current:
            if "python" not in current:
                return None                 # PID recycled by something else
            if recorded and Path(current).name != Path(recorded).name:
                return None                 # a different interpreter entirely
            return {**rec, "alive": True, "exe": exe}

        # Ambiguous: the PID is alive but we are not allowed to see what it is
        # (a protected system process, typically). Boot time settles it — a lock
        # written before the current boot cannot belong to a live process, and
        # after a reboot a low PID very often lands on exactly such a process.
        # Getting this wrong would make the bot refuse to start after precisely
        # the event auto-start exists to handle.
        booted, started = boot_time(), rec.get("started")
        if booted and started:
            try:
                if datetime.fromisoformat(started) < booted:
                    log.info("lock from pid %s predates the last boot (%s) — stale",
                             pid, booted.isoformat(timespec="seconds"))
                    return None
            except ValueError:
                pass                        # unparseable timestamp: fail closed
        return {**rec, "alive": True, "exe": exe}

    # ------------------------------------------------------------------
    def acquire(self) -> None:
        held_by = self.holder()
        if held_by and not self.force:
            since = held_by.get("started", "unknown time")
            raise AlreadyRunning(
                f"another bot is already running (pid {held_by['pid']}, started {since}).\n"
                f"Two instances share cache/shadow_books.json and the decision log, so the\n"
                f"second would corrupt the first's record rather than fail loudly.\n"
                f"  - if that process is the bot, leave it alone\n"
                f"  - if you are sure it is not, delete {self.path} or pass --force"
            )
        if held_by and self.force:
            log.warning("--force: taking the lock from pid %s", held_by["pid"])
        stale = self._read()
        if stale and not held_by:
            log.info("clearing stale lock from pid %s (process is gone)", stale.get("pid"))
        self.path.write_text(json.dumps({
            "pid": os.getpid(),
            "exe": sys.executable,
            "started": datetime.now().isoformat(timespec="seconds"),
        }), encoding="utf-8")
        self._held = True
        log.info("instance lock acquired (pid %s)", os.getpid())

    def release(self) -> None:
        if not self._held:
            return
        try:
            rec = self._read()
            # Never delete a lock another process has since taken.
            if rec and int(rec.get("pid") or 0) == os.getpid():
                self.path.unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove lock file %s", self.path)
        self._held = False

    # ------------------------------------------------------------------
    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
