"""Offline calibration report over the decision log.

Answers the question that decides whether this system has edge: does Claude's
stated confidence predict outcomes? Run after accumulating paper-trading
history:  python calibration.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import config


def main() -> None:
    if not config.DECISION_LOG.exists():
        print("no decision log yet — run main.py first")
        return

    # Keyed by (backend, confidence bucket) so Claude can be compared directly
    # against the free rule baseline on the same watchlist and period.
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    cycles_by_backend: dict[str, int] = defaultdict(int)
    total_cycles = 0
    gate_rejections: dict[str, int] = defaultdict(int)

    with open(config.DECISION_LOG, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total_cycles += 1
            backend = record.get("backend", "unknown")
            cycles_by_backend[backend] += 1
            for d in record["decisions"]:
                if d["action"] in ("buy", "sell"):
                    bucket = f"{int(d['confidence'] * 10) / 10:.1f}"
                    by_bucket[(backend, bucket)].append(d)
                if not d["gate_approved"]:
                    for reason in d["gate_reasons"]:
                        gate_rejections[reason.split("(")[0].strip()] += 1

    print(f"cycles logged: {total_cycles}")
    for backend, count in sorted(cycles_by_backend.items()):
        print(f"  {backend}: {count} cycles")
    print()

    print("proposals by backend and confidence bucket:")
    for backend, bucket in sorted(by_bucket):
        ds = by_bucket[(backend, bucket)]
        approved = sum(1 for d in ds if d["gate_approved"])
        print(f"  {backend:8s} {bucket}: {len(ds)} proposed, {approved} approved")

    print("\ntop gate rejection reasons:")
    for reason, count in sorted(gate_rejections.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {count:4d}  {reason}")

    print(
        "\nNote: P&L attribution needs order fills from the broker — join "
        "decisions.jsonl order_ids against Alpaca's order/position history to "
        "compute win rate per confidence bucket."
    )


if __name__ == "__main__":
    main()
