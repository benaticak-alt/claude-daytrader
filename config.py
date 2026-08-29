"""Central configuration for the Claude day-trading agent.

Everything the deterministic risk gate enforces lives here, NOT in the prompt.
Claude never sees or controls these limits.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Broker (Alpaca) — PAPER ONLY by default.
# ---------------------------------------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
# Live trading requires BOTH this flag and paper=False to be set deliberately.
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() != "false"

# ---------------------------------------------------------------------------
# Universe & cadence
# ---------------------------------------------------------------------------
WATCHLIST = [s.strip().upper() for s in os.getenv("WATCHLIST", "SPY,QQQ,NVDA,TSLA,AAPL").split(",")]
CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS", "300"))  # 5-min decision cadence
BAR_TIMEFRAME_MINUTES = 5
BAR_LOOKBACK = 120  # bars kept for rolling indicators (spans prior sessions)

# Fetch several calendar days so RSI/ATR have history at the opening bell.
# Without this the bot is blind until ~20 bars into the session (about 11:10am),
# missing the open entirely — where most intraday momentum actually happens.
# VWAP is still computed from TODAY'S session only; see SymbolSnapshot.session_bars.
BAR_HISTORY_DAYS = int(os.getenv("BAR_HISTORY_DAYS", "5"))
MIN_BARS_FOR_INDICATORS = 20   # rolling-window minimum (may span sessions)
MIN_SESSION_BARS = 2           # VWAP needs at least a little of today to mean anything

# ---------------------------------------------------------------------------
# Data-quality gate (lesson from the Polymarket bot's Gamma-fallback artifact:
# never let the model reason over stale/synthetic quotes)
# ---------------------------------------------------------------------------
MAX_QUOTE_AGE_SECONDS = 10
MAX_SPREAD_PCT = 0.5          # reject symbols with >0.5% bid/ask spread
REQUIRE_LIVE_QUOTE = True     # hard requirement, analog of REQUIRE_CLOB_BOOK

# Data feed: "iex" (free tier) or "sip" (paid, full consolidated tape).
#
# IMPORTANT: IEX is a single exchange carrying only a few percent of national
# volume. Its quotes are REAL but represent a partial view — spreads read wider
# and volume far lower than the consolidated NBBO that actually fills your
# orders. Decisions made on IEX data and filled at NBBO prices are a genuine
# source of paper-vs-live divergence. Fine for validating machinery; upgrade to
# SIP before trusting any P&L conclusion.
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
# IEX spreads run wider than NBBO, so the spread gate is loosened on that feed
# to avoid rejecting every symbol for a spread the real market doesn't have.
MAX_SPREAD_PCT_IEX = float(os.getenv("MAX_SPREAD_PCT_IEX", "1.5"))

# ---------------------------------------------------------------------------
# Deterministic risk gate — Claude's output cannot override any of these
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 0.65             # gate auto-rejects theses below this
MAX_POSITION_NOTIONAL = 2_000.0   # hard $ ceiling per position
MAX_CONCURRENT_POSITIONS = 3

# --- Position sizing -------------------------------------------------------
# OFF: every entry is FLAT_POSITION_NOTIONAL. Flat sizing is what makes
#      "win rate by confidence bucket" interpretable — with variable sizes a
#      single large winner masquerades as edge. Keep this off until a
#      flat-sized baseline proves confidence actually predicts outcomes.
# ON:  notional = equity * BASE_POSITION_PCT * confidence multiplier,
#      then floored by MAX_POSITION_NOTIONAL and by half the available cash.
#
# Do not flip this mid-session: it splits the sample into two halves that
# cannot be compared.
CONFIDENCE_SIZING = os.getenv("CONFIDENCE_SIZING", "false").lower() == "true"
FLAT_POSITION_NOTIONAL = float(os.getenv("FLAT_POSITION_NOTIONAL", "1000"))

BASE_POSITION_PCT = float(os.getenv("BASE_POSITION_PCT", "0.01"))  # 1% of equity
# Confidence maps linearly from MIN_CONFIDENCE..1.0 onto these multipliers,
# so a floor-confidence trade is half-size and a maximal one is double.
SIZE_MIN_MULT = float(os.getenv("SIZE_MIN_MULT", "0.5"))
SIZE_MAX_MULT = float(os.getenv("SIZE_MAX_MULT", "2.0"))
# Never commit more than this fraction of available cash to one position.
MAX_CASH_FRACTION = float(os.getenv("MAX_CASH_FRACTION", "0.5"))
MAX_DAILY_LOSS = 200.0            # $ realized+unrealized; kill switch when breached
PDT_MIN_EQUITY = 25_000.0         # accounts under this are PDT-limited
PDT_MAX_DAY_TRADES = 3            # per rolling 5 business days
KILL_SWITCH_FILE = Path(__file__).parent / "KILL_SWITCH"  # touch this file to halt

# End-of-day handling. This strategy's risk model assumes intraday moves of
# ~0.1%; an overnight gap can be 5%+ on news or earnings. Holding through the
# close would expose the account to a risk the position sizing never priced in,
# so positions are force-flattened before the bell and new entries stop earlier.
NO_ENTRY_BEFORE_CLOSE_MIN = 20    # stop opening new positions this close to the bell
FLATTEN_BEFORE_CLOSE_MIN = 10     # force-close everything at this point

# ---------------------------------------------------------------------------
# SEC Form 4 insider-transaction feed (public disclosure data, no API key)
# ---------------------------------------------------------------------------
INSIDER_ENABLED = os.getenv("INSIDER_ENABLED", "true").lower() != "false"
INSIDER_LOOKBACK_DAYS = int(os.getenv("INSIDER_LOOKBACK_DAYS", "90"))
INSIDER_CACHE_MINUTES = int(os.getenv("INSIDER_CACHE_MINUTES", "60"))
INSIDER_MAX_FILINGS_PER_SYMBOL = 40  # bound work; mega-caps file constantly
# SEC requires a descriptive User-Agent with real contact info on every request.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "claude-daytrader research benaticak@gmail.com")

# ---------------------------------------------------------------------------
# Decision backend:
#   "claude" — Opus 5 forms theses (costs money)
#   "rule"   — VWAP/RSI momentum baseline, free, the control group
#   "orb"    — opening range breakout, free
#   "ollama" — local model, free
# ---------------------------------------------------------------------------
DECIDER_BACKEND = os.getenv("DECIDER_BACKEND", "claude").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

# Strategies to run in PARALLEL without trading them. Each keeps its own virtual
# position book and logs what it would have done, so you get a head-to-head on
# identical data at zero risk. Comma-separated; the live backend is skipped
# automatically if listed. Set empty to disable.
#
# Why shadows rather than trading several at once: with a shared position book
# there is no way to attribute P&L, and attribution is the entire question.
SHADOW_BACKENDS = [
    s.strip().lower()
    for s in os.getenv("SHADOW_BACKENDS", "rule,orb").split(",")
    if s.strip()
]

# OBSERVE MODE: run every strategy, place ZERO orders.
#
# The kill switch cannot do this — it skips the whole cycle, so no strategy
# runs and no data is gathered. Observe mode instead runs the full pipeline
# (data gate, indicators, every strategy, virtual books, logging) and simply
# never calls the broker. Costs nothing, risks nothing, and produces the same
# comparison data a live run would.
#
# In this mode DECIDER_BACKEND is folded into the shadow set: a "live" decider
# reading the real (permanently flat) account would propose entries forever and
# never manage an exit, making its record meaningless.
OBSERVE_MODE = os.getenv("OBSERVE_MODE", "false").lower() == "true"

# Round-trip cost applied to every virtual fill, per side, in basis points.
# Without it shadow strategies trade at the midpoint and look better than any
# real fill could — measured losses here average ~0.17%, so a spread of a few
# bps is a material share of the result, not a rounding error.
SHADOW_SLIPPAGE_BPS = float(os.getenv("SHADOW_SLIPPAGE_BPS", "2.0"))

# --- Time-of-day strategy routing (backend "scheduled") --------------------
# Intraday character is not constant: directional at the open, choppy midday,
# directional again into the close. Format "HH:MM-HH:MM=backend", ET.
# Entries come only from the active strategy; exits always come from whichever
# strategy opened the position (see decider_scheduled.py).
STRATEGY_SCHEDULE = os.getenv(
    "STRATEGY_SCHEDULE",
    "09:30-11:00=orb,11:00-14:30=meanrev,14:30-15:40=rule",
)

# --- VWAP mean reversion (meanrev) -----------------------------------------
# Entry band: too shallow and there is no edge to capture; too deep and the
# name is trending down rather than stretched.
MEANREV_MIN_STRETCH_PCT = float(os.getenv("MEANREV_MIN_STRETCH_PCT", "0.6"))
MEANREV_MAX_STRETCH_PCT = float(os.getenv("MEANREV_MAX_STRETCH_PCT", "2.5"))
MEANREV_RSI_MAX = float(os.getenv("MEANREV_RSI_MAX", "38"))
MEANREV_TARGET_PCT = float(os.getenv("MEANREV_TARGET_PCT", "0.10"))  # exit near VWAP
MEANREV_ATR_STOP = float(os.getenv("MEANREV_ATR_STOP", "1.5"))
# A spread widening during a drop signals real stress, not a stretch worth fading.
MEANREV_MAX_SPREAD_PCT = float(os.getenv("MEANREV_MAX_SPREAD_PCT", "0.8"))

# --- Opening range breakout (orb) ------------------------------------------
# The opening range is the high/low established in the first N minutes after
# the bell. Breakouts from it are one of the oldest documented intraday
# patterns: the range represents overnight-order absorption, and a decisive
# move beyond it on volume signals genuine directional commitment.
OPENING_RANGE_MINUTES = int(os.getenv("OPENING_RANGE_MINUTES", "15"))
# A breakout without volume is noise — this is the confirmation the plain
# momentum rule lacked, and whose absence produced the Aug 6 whipsaws.
ORB_VOLUME_MULT = float(os.getenv("ORB_VOLUME_MULT", "1.5"))
# Stop distance in ATRs below the entry price.
ORB_ATR_STOP_MULT = float(os.getenv("ORB_ATR_STOP_MULT", "1.0"))
# Refuse to chase: beyond this far past the range high, the move is already made.
ORB_MAX_EXTENSION_PCT = float(os.getenv("ORB_MAX_EXTENSION_PCT", "2.0"))

# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
CLAUDE_MAX_TOKENS = 16000  # Opus 5 thinks by default; max_tokens caps thinking + answer
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "high")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
DECISION_LOG = LOG_DIR / "decisions.jsonl"
