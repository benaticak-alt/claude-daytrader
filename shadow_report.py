"""Head-to-head strategy comparison from the logs.

Ranks strategies by EXPECTANCY — per-trade profit after losses — which is the
number that decides whether an edge survives itself. Win rate alone is
meaningless: 70% winners still bleeds if losses are four times the wins.

Everything is computed from individual trade events, never from a running
summary. The virtual books reset every time the bot restarts, so a summary
reflects only the newest process while the events persist across all of them.

    python shadow_report.py
    python shadow_report.py --day 2026-08-21
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime

import config

# bot.log line: "shadow[orb] EOD flatten: TSLA pnl=+26.31; SPY pnl=-0.47"
FLATTEN_LINE = re.compile(
    r"^(?P<ts>\S+ \S+?),\d+ .*?shadow\[(?P<name>\w+)\] EOD flatten: (?P<body>.+)$"
)
FLATTEN_ITEM = re.compile(r"(?P<sym>[A-Z.]+) pnl=(?P<pnl>[-+][\d.]+)")


def load_events(day: str | None) -> tuple[dict[str, list], int]:
    """Trade events per strategy, from the decision log plus recovered flattens."""
    trades: dict[str, list] = defaultdict(list)
    seen: set[tuple] = set()
    cycles = 0

    if config.DECISION_LOG.exists():
        with open(config.DECISION_LOG, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                stamp = datetime.fromisoformat(rec["ts"]).astimezone()
                if day and stamp.strftime("%Y-%m-%d") != day:
                    continue
                shadow = rec.get("shadow")
                if not shadow:
                    continue
                cycles += 1
                for name, data in shadow.items():
                    if data.get("error"):
                        continue
                    for ev in data.get("events", []):
                        if "pnl" not in ev:
                            continue
                        key = (name, ev["symbol"], round(ev["pnl"], 2),
                               stamp.strftime("%Y-%m-%d %H:%M"))
                        if key in seen:
                            continue
                        seen.add(key)
                        trades[name].append({**ev, "day": stamp.strftime("%Y-%m-%d")})

    # Recover end-of-day exits from bot.log. Before the fix these were never
    # written to the decision log, and they are often the largest trades of the
    # day — omitting them understates every strategy.
    botlog = config.LOG_DIR / "bot.log"
    if botlog.exists():
        for raw in botlog.read_text(encoding="utf-8", errors="replace").splitlines():
            m = FLATTEN_LINE.match(raw)
            if not m:
                continue
            d = m.group("ts")[:10]
            if day and d != day:
                continue
            hhmm = m.group("ts")[11:16]
            for item in FLATTEN_ITEM.finditer(m.group("body")):
                pnl = float(item.group("pnl"))
                key = (m.group("name"), item.group("sym"), round(pnl, 2), f"{d} {hhmm}")
                if key in seen:
                    continue
                seen.add(key)
                trades[m.group("name")].append({
                    "action": "eod_flatten", "symbol": item.group("sym"),
                    "pnl": pnl, "day": d,
                })
    return trades, cycles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day", help="filter to one local date, YYYY-MM-DD")
    args = p.parse_args()

    trades, cycles = load_events(args.day)
    if not trades:
        print("no shadow trades logged yet.\n"
              "Set SHADOW_BACKENDS (and OBSERVE_MODE=true) and run main.py.")
        return

    print("=" * 76)
    print(f"SHADOW STRATEGY COMPARISON — {cycles} cycles"
          + (f" on {args.day}" if args.day else ""))
    print("=" * 76)

    rows = []
    for name, ts in trades.items():
        wins = [t["pnl"] for t in ts if t["pnl"] > 0]
        losses = [t["pnl"] for t in ts if t["pnl"] <= 0]
        n = len(ts)
        wr = len(wins) / n if n else 0.0
        avg_w = sum(wins) / len(wins) if wins else 0.0
        avg_l = abs(sum(losses) / len(losses)) if losses else 0.0
        exp = wr * avg_w - (1 - wr) * avg_l
        rows.append((exp, name, n, wr, avg_w, avg_l, sum(t["pnl"] for t in ts)))

    rows.sort(reverse=True)
    print(f"  {'strategy':10s} {'expect':>8s} {'realized':>10s} {'trades':>7s} "
          f"{'win%':>6s} {'avg W':>8s} {'avg L':>8s}")
    print("  " + "-" * 72)
    for exp, name, n, wr, avg_w, avg_l, total in rows:
        print(f"  {name:10s} {exp:+8.2f} {total:+10.2f} {n:7d} "
              f"{100*wr:5.1f}% {avg_w:+8.2f} {-avg_l:+8.2f}")

    print("\n  Expectancy = per-trade profit after losses. Positive survives;")
    print("  negative bleeds however good the win rate looks.")

    # Concentration check: one huge trade can carry an entire strategy.
    print("\n  largest single trades:")
    for exp, name, n, *_rest in rows:
        ts = sorted(trades[name], key=lambda t: -abs(t["pnl"]))[:2]
        if not ts:
            continue
        share = abs(ts[0]["pnl"]) / max(sum(abs(t["pnl"]) for t in trades[name]), 1e-9)
        print(f"    {name:10s} " + ", ".join(
            f"{t['symbol']} {t['pnl']:+.2f} ({t['day']})" for t in ts)
            + f"   biggest = {100*share:.0f}% of all P&L moved")

    best = rows[0]
    print()
    if best[0] > 0:
        print(f"  BEST: {best[1]} at {best[0]:+.2f}/trade over {best[2]} trades.")
        if best[2] < 30:
            print(f"  {best[2]} trades is too few to act on — keep observing.")
    else:
        print("  No strategy has positive expectancy. Keep observing, do not trade.")


if __name__ == "__main__":
    main()
