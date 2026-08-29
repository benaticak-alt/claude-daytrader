# claude-daytrader

An autonomous intraday equities trading agent: Claude (Opus 5) is the decision-maker,
a **deterministic risk gate** sits between Claude and the broker, and Alpaca paper
trading is the execution venue.

> **This is an experiment, not financial advice.** It starts in paper mode and should
> stay there until the decision log proves the model's confidence is calibrated.
> Nothing here guarantees profit; intraday trading usually loses money.

## Architecture

```
DataFeed ──> ContextBuilder ──> Decider (Claude) ──> RiskGate ──> Executor (Alpaca)
   │                                                    │              │
   └── data-quality gate                                └──────────────┴──> trade_log (JSONL)
```

Design principles (several imported from a prior prediction-market bot):

1. **Claude proposes, code disposes.** Claude outputs a structured thesis
   (direction, confidence, invalidation, size). The risk gate — plain Python with
   limits from `config.py` — approves, shrinks, or rejects. Claude cannot raise
   its own limits, and the gate fails closed when account data is missing.
2. **Data-quality gate before the model.** Stale quotes, degenerate bid/ask, wide
   spreads, or thin bar history exclude a symbol from the cycle entirely. A model
   reasoning over bad data produces fictional edge that looks real in the P&L.
3. **Log everything for calibration.** Every cycle's context, thesis, confidence,
   gate verdict, and order ID goes to `logs/decisions.jsonl`. The question that
   decides whether to ever go live: *does stated confidence predict outcomes?*
   (`python calibration.py` for a first-pass report.)
4. **Hard safety rails**: daily loss kill switch, PDT compliance for sub-$25k
   accounts, position caps, long-only v1, end-of-day flatten, and a manual
   `KILL_SWITCH` file that halts everything instantly.

## Is it actually running?

```bash
python health_check.py
```

Answers the one question the process list cannot. Several projects on this
machine have a `main.py`, so seeing one in Task Manager proves nothing — a
two-day outage went unnoticed exactly that way. The authoritative signal is
**decision log freshness**: during market hours it updates every cycle, so a
log older than a few cycles means the bot is dead regardless of what processes
appear to be running.

`logs/bot.log` records startup, shutdown, and crashes, so a silent death
leaves a trace. It rotates at 5MB, keeping 3 backups.

To have it start on its own each weekday, run **from an elevated PowerShell**:

```
powershell -ExecutionPolicy Bypass -File install_schedule.ps1
```

That registers a Scheduled Task at 9:25 AM local on weekdays, restarting up to
3 times if it dies. It only registers the schedule — it does not start trading
immediately. Remove it with `Unregister-ScheduledTask -TaskName ClaudeDaytrader`.

## Running multiple strategies

`SHADOW_BACKENDS` (default `rule,orb`) runs strategies **in parallel without
trading them**. Each keeps its own virtual position book, sees the identical
market snapshot every cycle, and logs what it would have done. Results land
under a `shadow` key in `logs/decisions.jsonl`.

**Why shadows instead of trading several at once.** With a shared position book
there is no answer to "who owns AAPL" when two strategies want it, no rule for
when one says close and another says hold, and — fatally — **no way to attribute
P&L**. A +$40 day tells you nothing if three strategies were mixed into it, and
attribution is the entire question you are trying to answer.

Shadow mode gives the head-to-head on identical data at zero risk. Once one
strategy demonstrably wins, promote it to `DECIDER_BACKEND` and demote the rest.

`claude` is deliberately awkward as a shadow: it bills on every cycle while
placing no trades. Run it live if you want to pay for it.

## Strategies

| Backend | Logic |
|---|---|
| `rule` | RSI 52–68 and 0.05–1.50% above session VWAP. Exit on RSI < 45 or price below VWAP. The deliberately dumb control group. |
| `orb` | Opening range breakout — see below. |
| `claude` | Opus 5 forms theses from the full context. |
| `ollama` | Local model, free, weak judgment. |

### Time-of-day routing (`scheduled`)

Intraday character is not constant. Volume and volatility trace a U-curve:
directional at the open, choppy and thin through midday, directional again into
the close. One strategy across all of it is the wrong strategy for hours a day —
and both Aug 6 losses were momentum entries into a flat afternoon tape.

```
STRATEGY_SCHEDULE=09:30-11:00=orb,11:00-14:30=meanrev,14:30-15:40=rule
```

**The hard part is ownership, not routing.** If ORB opens NVDA at 10:00 and the
clock rolls past 11:00, mean reversion now holds a position it never took and
whose exit rule ("has it reverted to VWAP?") is meaningless for a breakout
trade. Naive routing strands positions under a strategy that cannot manage them.

So the two directions route differently:

- **Entries** — only from the strategy active for the current time.
- **Exits** — always from the strategy that *opened* the position, whatever
  time it is.

Ownership is in-memory and lost on restart; an unknown position falls back to
the active strategy, and the end-of-day flatten backstops either way.

### VWAP mean reversion (`meanrev`)

```
ENTRY   price MEANREV_MIN..MAX % BELOW session VWAP (0.6%-2.5% default)
        AND RSI below MEANREV_RSI_MAX
        AND spread still tight
EXIT    price back within MEANREV_TARGET_PCT of VWAP, or MEANREV_ATR_STOP ATRs down
```

The floor on the band matters as much as the ceiling: barely below VWAP has no
edge to capture, and 4% below is a downtrend, not a stretch. This trades the
middle.

### Opening range breakout (`orb`)

```
ENTRY   opening range window has CLOSED (first OPENING_RANGE_MINUTES, default 15)
        AND price breaks above the range high
        AND breakout-bar volume > ORB_VOLUME_MULT x the 20-bar average
        AND price above session VWAP
        AND not more than ORB_MAX_EXTENSION_PCT past the high (no chasing)

EXIT    price falls back inside the range (breakout failed)
        OR price drops ORB_ATR_STOP_MULT ATRs below entry
```

The volume test is the guard the plain momentum rule lacked — both Aug 6 losers
were unconfirmed entries that reverted within minutes. `python test_orb.py`
covers 16 scenarios.

## Position sizing

**The risk gate owns sizing outright.** A decider's `suggested_notional` is
advisory — logged so you can see what it wanted, never used to execute. A model
must not be able to influence its own position size; that's the point of a gate.

Two modes, via `CONFIDENCE_SIZING` in `.env`:

| Mode | Formula |
|---|---|
| `false` (default) | flat `FLAT_POSITION_NOTIONAL` every trade |
| `true` | `equity × BASE_POSITION_PCT × confidence multiplier` |

The multiplier maps confidence from `MIN_CONFIDENCE`→1.0 onto
`SIZE_MIN_MULT`→`SIZE_MAX_MULT` (0.5x–2.0x by default), so a floor-confidence
trade is half-size and a maximal one double. Both modes are then floored by
`MAX_POSITION_NOTIONAL` **and by half of available cash** — no single position
can ever exceed 50% of cash. Unknown equity or cash sizes to zero and the trade
is refused.

**Leave it off until you have a flat-sized baseline.** "Win rate by confidence
bucket" is only interpretable when every trade is the same size — with variable
sizing, one large winner in a high-confidence bucket looks like edge that isn't
there. Measure first, then scale. Never flip it mid-session: that splits the
sample into two halves you can't compare.

**Still missing: volatility normalization.** TSLA's daily range ran ~10x SPY's
(0.67% vs 0.07% on 2026-08-11), so equal notionals carry very unequal risk and
P&L will cluster in high-volatility names regardless of signal quality.
ATR-normalized sizing is the fix; not built yet.

## End-of-day handling

This strategy sizes for ~0.1% intraday moves. An overnight gap can be 5%+ on
news or earnings — a risk the position sizing never priced in. So:

- **20 minutes before the close**: the risk gate refuses all new entries.
- **10 minutes before the close**: the loop force-closes every open position
  before doing anything else.

Both thresholds are in `config.py`. Closes remain permitted inside the window
(the flatten depends on it), and an unknown time-to-close fails closed.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env    # then fill in Alpaca paper keys
```

Anthropic auth: `ANTHROPIC_API_KEY` in the environment, or `ant auth login`.

## Run

**Easiest: double-click `daytrade.bat`.** A menu launcher for everything below.
Its header shows the active backend, paper/live mode, and kill-switch state, so
you can see at a glance what a run would actually do. Safe options are grouped
separately from the ones that place orders, and both trading options ask for
confirmation first.

Or run the scripts directly:

**No keys needed** — these validate the machinery offline, for free:

```bash
python test_gate.py       # 17 adversarial tests against the risk gate
```

```bash
python dryrun.py --cycles 5   # full decision chain, synthetic data, mock broker
```

**With Alpaca keys** in `.env`:

```bash
python main.py --once     # single decision cycle (smoke test)
python main.py            # loop every CYCLE_SECONDS during market hours
```

To halt immediately at any time: create an empty file named `KILL_SWITCH` in this
directory. The loop stops placing orders until you delete it.

## Data feed caveat (free tier)

Alpaca's free plan provides **IEX** data only; the consolidated **SIP** tape is
paid. `ALPACA_DATA_FEED` defaults to `iex` so the free tier works out of the box.

IEX is a single exchange carrying a low-single-digit percentage of national
volume. Its quotes are real, but they are a *partial view*: spreads read much
wider and volume much lower than the NBBO your orders actually fill against.
Observed live: NVDA showed a **0.721%** spread on IEX against a true NBBO spread
of roughly 0.01%. `MAX_SPREAD_PCT_IEX` (default 1.5%) exists so the data gate
doesn't reject every symbol over a spread the real market doesn't have.

This matters for conclusions, not just plumbing: decisions made on IEX prices
and filled at NBBO prices diverge, so P&L from a free-tier paper run is not a
clean estimate of live performance. Fine for validating machinery. Upgrade to
`ALPACA_DATA_FEED=sip` with a paid data plan before trusting any P&L result.

## Decision backends (`DECIDER_BACKEND` in `.env`)

Only the Claude backend costs money. The other two exist so you can validate
the machinery for free before spending anything.

| Backend  | Cost | What it's for |
|----------|------|---------------|
| `claude` | ~$5–6/day at 5-min cadence | The real decider. The only one whose results mean anything. |
| `rule`   | $0 | Deterministic VWAP + RSI baseline. Exercises every stage of the pipeline, **and** is the control group. |
| `ollama` | $0 | Local LLM (`OLLAMA_MODEL`). Validates the structured-output path without an API key. |

**Start with `rule`.** It costs nothing, runs instantly, and tests everything
that can actually break at 3am: the data gate, the risk gate, order routing,
position accounting, logging. If the bot is going to crash or misroute an
order, you want to find out for free.

`rule` is also the **baseline you have to beat**. "Claude has edge" is an empty
claim without a control group. If Opus 5 can't outperform a dumb VWAP+RSI rule
on the same watchlist over the same period, the LLM is adding nothing — and
that is a genuinely possible outcome worth knowing early. `calibration.py`
keys results by backend so the comparison is direct.

**On `ollama`:** it proves the plumbing handles schema-constrained decisions,
but do not read its trades as a preview of Claude's. In testing, `gemma4`
reasoned sensibly about which symbol to pick and then filled the
`invalidation` field with raw control tokens. The schema guaranteed the
response *parsed*; it guaranteed nothing about meaning. Local models are for
finding bugs, not for forming beliefs about strategy.

## Validation path (in order)

0. **Run `DECIDER_BACKEND=rule` first**, for free, until the plumbing is boring.
1. **Paper trade ≥ 4–6 weeks.** Accumulate decisions.jsonl history.
2. **Calibration analysis.** Win rate by confidence bucket must be monotonic and
   the ≥0.65 buckets must clear costs. If 0.65-confidence trades win 50%, there is
   no edge — stop here.
3. **Only then consider small real capital**, and only by explicitly setting
   `ALPACA_PAPER=false`.

## Insider transactions (SEC Form 4)

`insider_feed.py` pulls **public** SEC Form 4 disclosures — the filings that
corporate officers, directors, and 10%+ owners must submit within two business
days of trading their own company's stock. Free, official, no API key. It folds
an `insider_form4` block into each symbol's context.

Test it standalone: `python insider_feed.py`

**Transaction-code filtering is the whole game.** Most Form 4 rows carry zero
directional information — only `P` (open-market purchase) and `S` (open-market
sale) are kept; grants (`A`), option exercises (`M`), tax withholding (`F`), and
gifts (`G`) are discarded. A real example from NVDA's filings: an insider
"disposed" of 500,000 shares under code `G` — a gift, not a sale. Counted
naively that reads as a huge bearish signal and is in fact noise.

**Two honest caveats**, both encoded in the system prompt:

1. **Wrong timescale.** Insider-buying alpha is documented over weeks to months,
   not 5-minute bars. It's used as a directional *tilt* on trades the technicals
   already justify — never as a trigger. The one genuinely intraday element is
   `filings_today`, since a fresh disclosure can move the tape.
2. **Mega-caps are nearly silent.** Over 90 days, SPY/QQQ/NVDA/TSLA/AAPL showed
   **zero** open-market insider buys (only routine selling). Buying is the
   informative side and it's rare. This feed earns its keep on small- and
   mid-caps, so it argues for widening `WATCHLIST` beyond mega-caps.

Turn it off with `INSIDER_ENABLED=false`. Results cache for 60 minutes, so it
costs one SEC round-trip per symbol per hour, and a failure never blocks trading.

## Not yet built (deliberate v1 cuts)

- **Options** — the models support the reasoning, but assignment/expiration
  handling in the gate needs to exist first.
- **News/sentiment feed** — Claude's actual comparative advantage over a pure
  quant model; add a headlines section to the context builder when a news API
  key is available.
- **Short entries** — long-only until long-side calibration is proven.
- **Intra-cycle stop-losses** — the invalidation condition is logged but only
  re-evaluated once per cycle; a faster watcher thread would enforce it in
  real time.
- **Insider-driven watchlist scanner** — scanning all of EDGAR for cluster buys
  to *discover* symbols is a daily batch job, not an intraday concern. That one
  genuinely is a separate program.
