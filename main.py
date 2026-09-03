"""Main decision loop.

data feed -> context builder -> Claude -> risk gate -> executor -> log
Run:  python main.py            (loops during market hours)
      python main.py --once     (single cycle, easiest way to smoke-test)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import config
from context_builder import build_context
from data_feed import DataFeed
from decider import Decider
from models import CycleDecisions
from executor import Executor
from insider_feed import InsiderFeed
from risk_gate import RiskGate
from trade_log import log_cycle

def _setup_logging() -> None:
    """Log to console AND to logs/bot.log.

    Without a persistent process log there is no way to tell a clean shutdown
    from a crash from "never started" — the decision log only records cycles
    that actually completed, so a bot that died at startup leaves no trace.
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        config.LOG_DIR / "bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)


_setup_logging()
log = logging.getLogger("main")


def _make_decider(name: str):
    """Construct a decider by name. Raises if the name is unknown."""
    if name == "rule":
        from decider_rule import RuleDecider
        return RuleDecider()
    if name == "orb":
        from decider_orb import ORBDecider
        return ORBDecider()
    if name == "meanrev":
        from decider_meanrev import MeanReversionDecider
        return MeanReversionDecider()
    if name == "scheduled":
        from decider_scheduled import ScheduledDecider, parse_schedule
        schedule = parse_schedule(config.STRATEGY_SCHEDULE)
        # Build only the sub-strategies the schedule actually references.
        names = sorted({n for _, _, n in schedule})
        subs = {n: _make_decider(n) for n in names}
        log.info("scheduled routing: %s", config.STRATEGY_SCHEDULE)
        return ScheduledDecider(subs, schedule)
    if name == "insider":
        from decider_insider import InsiderDecider
        return InsiderDecider()
    if name == "ml":
        from decider_ml import MLDecider
        return MLDecider()
    if name == "ollama":
        from decider_ollama import OllamaDecider
        return OllamaDecider()
    if name == "claude":
        return Decider()
    raise ValueError(f"unknown decider backend: {name!r}")


def build_shadow_runner():
    """Non-executing strategies run alongside the live one, for comparison.

    In observe mode the live backend is folded in too: with no orders placed the
    account stays flat forever, so a decider reading it would propose entries
    endlessly and never manage an exit. Giving it a virtual book like every
    other strategy is the only way its record means anything.
    """
    if config.OBSERVE_MODE:
        names = sorted(set(config.SHADOW_BACKENDS) | {config.DECIDER_BACKEND})
    else:
        names = [n for n in config.SHADOW_BACKENDS if n != config.DECIDER_BACKEND]
    if not names:
        return None
    deciders = {}
    for name in names:
        try:
            deciders[name] = _make_decider(name)
        except Exception:
            log.exception("could not build shadow decider %r — skipping", name)
    if not deciders:
        return None
    # Deliberately excluded: 'claude' as a shadow would bill on every cycle
    # while placing no trades. Run it live if you want to pay for it.
    if "claude" in deciders:
        log.warning("shadow 'claude' costs money every cycle with no trades placed")
    from shadow import ShadowRunner
    log.info("shadow strategies: %s", ", ".join(deciders))
    return ShadowRunner(deciders, config.FLAT_POSITION_NOTIONAL)


def build_decider():
    """Select the decision backend. Only 'claude' costs money."""
    backend = config.DECIDER_BACKEND
    try:
        decider = _make_decider(backend)
    except ValueError:
        log.warning("unknown DECIDER_BACKEND=%r, falling back to claude", backend)
        backend, decider = "claude", Decider()
    labels = {
        "rule": "RULE baseline (free, deterministic)",
        "orb": "OPENING RANGE BREAKOUT (free, deterministic)",
        "meanrev": "VWAP MEAN REVERSION (free, deterministic)",
        "scheduled": "SCHEDULED time-of-day routing (free)",
        "ollama": f"OLLAMA local model {config.OLLAMA_MODEL} (free)",
        "claude": f"CLAUDE {config.CLAUDE_MODEL} (billed per cycle)",
    }
    log.info("decider: %s", labels.get(backend, backend))
    return decider


def run_cycle(
    feed: DataFeed,
    decider: Decider,
    gate: RiskGate,
    executor: Executor,
    insider_feed: InsiderFeed | None,
    shadow_runner=None,
) -> None:
    account = executor.account_state()
    positions = executor.open_positions()

    # Time-to-close drives both the entry cutoff (enforced in the risk gate)
    # and the hard flatten below.
    mins_left = executor.minutes_to_close()
    account["minutes_to_close"] = mins_left

    # End-of-day flatten. Runs before anything else and returns immediately:
    # this strategy must never carry an overnight gap.
    # A multi-day strategy must not be flattened nightly — that would destroy
    # the effect it was measured on. Such deciders declare holds_overnight.
    live_holds_overnight = getattr(decider, "holds_overnight", False)

    if mins_left is not None and 0 < mins_left <= config.FLATTEN_BEFORE_CLOSE_MIN:
        if positions and not config.OBSERVE_MODE and not live_holds_overnight:
            log.warning("EOD flatten: %.0f min to close, closing %d position(s)",
                        mins_left, len(positions))
            for pos in positions:
                executor.close_position(pos["symbol"])
        else:
            log.info("EOD window (%.0f min to close) — no live positions", mins_left)

        # Shadow books must flatten too, or their P&L includes overnight gaps
        # the live bot could never have taken.
        #
        # These results MUST be logged. End-of-day exits are a large share of
        # all outcomes — and often the largest single ones — so dropping them
        # makes every downstream expectancy figure silently wrong.
        if shadow_runner is not None:
            try:
                snaps = [feed.snapshot(s) for s in config.WATCHLIST]
                ctx = build_context(snaps, account, positions)
                flat = shadow_runner.flatten_all(ctx)
                log_cycle(
                    ctx,
                    CycleDecisions(market_read="EOD flatten", decisions=[]),
                    [],
                    flat,
                )
            except Exception:
                log.exception("shadow EOD flatten failed")
        return

    snapshots = []
    for symbol in config.WATCHLIST:
        try:
            snapshots.append(feed.snapshot(symbol))
        except Exception:
            log.exception("snapshot failed for %s", symbol)

    usable = [s for s in snapshots if s.data_ok]
    if not usable and not positions:
        log.warning("no usable symbols and no open positions — skipping cycle")
        return

    # Insider data is cached (default 60min TTL) so this is a no-op most cycles.
    # A failure here must never block trading — it is enrichment, not a dependency.
    insider: dict = {}
    if insider_feed is not None:
        for snap in usable:
            try:
                insider[snap.symbol] = insider_feed.summary(snap.symbol).to_context()
            except Exception:
                log.exception("insider lookup failed for %s", snap.symbol)

    context_json = build_context(snapshots, account, positions, insider or None)

    # OBSERVE MODE: every strategy runs through the shadow runner (including the
    # configured backend) and nothing reaches the broker. No live decider runs,
    # so the primary is not evaluated twice.
    if config.OBSERVE_MODE:
        shadow_out = shadow_runner.run(context_json) if shadow_runner else None
        observed = CycleDecisions(
            market_read="OBSERVE MODE — all strategies logged, no orders placed",
            decisions=[],
        )
        log_cycle(context_json, observed, [], shadow_out)
        return

    cycle = decider.decide(context_json)
    log.info("market read: %s", cycle.market_read)

    # Parallel strategies — logged only, never executed. Each keeps its own
    # virtual book, so their records are independent of what the live one does.
    shadow_out = shadow_runner.run(context_json) if shadow_runner else None

    results = []
    approved_entries = 0
    for decision in cycle.decisions:
        verdict = gate.evaluate(decision, account, positions, approved_entries)
        order_id = None
        if verdict.approved:
            if decision.action == "close":
                ok = executor.close_position(decision.symbol)
                order_id = "closed" if ok else None
            else:
                order_id = executor.submit_entry(decision, verdict.capped_notional)
                if order_id:
                    approved_entries += 1
        log.info(
            "%s %s conf=%.2f -> %s (%s)",
            decision.action.upper(), decision.symbol, decision.confidence,
            "APPROVED" if verdict.approved else "REJECTED", "; ".join(verdict.reasons),
        )
        results.append((decision, verdict, order_id))

    log_cycle(context_json, cycle, results, shadow_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args()

    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env (see .env.example)")

    feed = DataFeed()
    decider = build_decider()
    shadow_runner = build_shadow_runner()
    gate = RiskGate()
    executor = Executor()
    insider_feed = (
        InsiderFeed(config.INSIDER_LOOKBACK_DAYS, config.INSIDER_CACHE_MINUTES)
        if config.INSIDER_ENABLED
        else None
    )

    mode = "OBSERVE (no orders)" if config.OBSERVE_MODE else (
        "PAPER" if config.ALPACA_PAPER else "LIVE")
    log.info("=" * 60)
    log.info("STARTUP pid=%s mode=%s backend=%s", os.getpid(), mode, config.DECIDER_BACKEND)
    log.info("watchlist=%s cycle=%ss insider_feed=%s",
             config.WATCHLIST, config.CYCLE_SECONDS, "on" if insider_feed else "off")
    log.info("=" * 60)

    if args.once:
        run_cycle(feed, decider, gate, executor, insider_feed, shadow_runner)
        return

    cycles = 0
    heartbeat = 0
    try:
        while True:
            if config.KILL_SWITCH_FILE.exists():
                log.warning("KILL_SWITCH present — sleeping, no decisions")
            elif not executor.market_open():
                # Only log this occasionally; otherwise an overnight run buries
                # the interesting lines under hundreds of identical ones.
                if heartbeat % 12 == 0:
                    log.info("market closed — sleeping (heartbeat %d)", heartbeat)
                heartbeat += 1
            else:
                heartbeat = 0
                try:
                    run_cycle(feed, decider, gate, executor, insider_feed, shadow_runner)
                    cycles += 1
                except Exception:
                    log.exception("cycle failed — continuing")
            time.sleep(config.CYCLE_SECONDS)
    except KeyboardInterrupt:
        log.info("SHUTDOWN: interrupted by user after %d cycles", cycles)
    except Exception:
        # A crash here previously left no trace at all.
        log.exception("FATAL: loop died after %d cycles", cycles)
        raise
    finally:
        log.info("bot exiting after %d cycles", cycles)


if __name__ == "__main__":
    main()
