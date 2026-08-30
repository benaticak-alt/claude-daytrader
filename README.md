# claude-daytrader

An autonomous US-equities trading system built to answer one question honestly:
**is there retail-accessible intraday edge on free data — and can I trust my own
measurement of it?**

The answer to the first question turned out to be **no**, measured across four
hand-written strategies, three machine-learned models, ~25,000 backtested rule
trades and ~700,000 out-of-sample model predictions. The answer to the second
question is the actual product: a validation pipeline that caught, before any
capital was at risk, every one of the illusions that usually make trading bots
look profitable — including two lookahead leaks in my own feature code, a
wall-clock bug that silently invalidated a two-year backtest, and a skill-free
model that passed the deployment gate on pure bull-market drift.

> Paper trading only. Nothing here is financial advice; the headline finding is
> that these strategies **lose** money after costs.

## Findings

| Approach | Sample | Result (per $1k trade, after costs) |
|---|---|---|
| VWAP/RSI momentum rule | 4,506 backtested + 90 live/shadow trades | **−$0.30 to −$1.13** |
| Opening range breakout | 3,422 backtested (12-symbol universe) | **−$0.51** |
| VWAP mean reversion | 2,182 backtested | **−$0.47** |
| Time-of-day strategy router | 2,820 backtested | **−$0.41** |
| GBM, ±1 ATR / 1h label | 331k OOS predictions, 6 walk-forward folds | AUC 0.527 (stable) — **−$0.47** best threshold |
| GBM, ±2 ATR / 4h label | 331k OOS predictions | AUC 0.564 — **−$0.43**: learned *volatility*, not direction |
| GBM, daily ±1.5 ATR / 10d | 32.5k OOS predictions | AUC 0.504 — "+$6.25/trade" that was **beta, not alpha** (below) |

Three results worth the whole project:

1. **A real signal that cannot pay its costs.** The intraday models found a
   genuinely stable signal — AUC above 0.5 in all twelve walk-forward folds —
   worth ≈ $0.07/trade gross against a $0.40 round-trip cost. Edge exists;
   economics don't.
2. **Sample size is everything.** The breakout strategy measured **+$2.59/trade
   over 19 live shadow trades** and **−$0.23 over 1,984 backtested trades**.
   Every strategy's apparent edge shrank monotonically as its sample grew.
3. **Beta masquerading as alpha.** The daily-horizon model passed the original
   deployment gate at +$6.25/trade — until the every-bar baseline revealed that
   buying *everything* earned +$5.66/trade over the same bull-market window.
   The model's picks were actually −$2.50/trade *worse* than indiscriminate
   buying (t = −3.5). The gate now tests **excess over the every-bar baseline
   at t ≥ 2**, not profit over zero.

## Architecture

```
DataFeed ──> ContextBuilder ──> Decider (rule|orb|meanrev|scheduled|ml|claude) ──> RiskGate ──> Executor (Alpaca paper)
   │                                                                                  │              │
   └── data-quality gate                                                              └──────────────┴──> JSONL logs
```

Principles that did the real work:

- **Deciders propose, deterministic code disposes.** Every strategy — including
  the LLM — emits a structured thesis (direction, confidence, invalidation,
  size). A deterministic risk gate approves, shrinks, or refuses: confidence
  floor, position caps, daily-loss kill switch, PDT compliance, end-of-day
  flatten, half-of-cash sizing ceiling. The gate **fails closed** on any
  missing input, and a decider cannot influence its own position size.
- **Data-quality gate before the model.** Stale quotes, zero asks (which
  Alpaca's free feed really does return after hours), wide spreads, or thin
  history exclude a symbol from the cycle entirely. A model reasoning over bad
  data produces fictional edge that looks real in the P&L.
- **The model's own verdict is binding.** `train_model.py` embeds
  `deployable: false` in the artifact when walk-forward evaluation fails, and
  the live decider refuses to trade any model whose own evaluation said no —
  and refuses any model whose feature schema the live context cannot supply
  (no silent NaN-filling).
- **Shadow mode / observe mode.** Any set of strategies runs in parallel on
  identical live data with isolated virtual books and modeled slippage,
  logging what they *would* have done while one (or zero) strategies trades.
  P&L attribution stays clean; a broken experiment cannot disturb execution.
- **Backtests replay the production code.** Same decider classes, same context
  builder, same risk gate, same slippage model, same end-of-day flatten — a
  ranking device, not a parallel invention that can flatter itself.

## Rigor mechanics (the part that caught the bugs)

- **Leakage auditor** (`audit_dataset.py`) — flagged an opening-range feature
  visible before the range closed, and a mislabeled overnight-gap feature.
- **Walk-forward only**, expanding window, embargo ≥ label lookahead
  (1 day intraday, 14 days for the 10-day daily label). Never random CV.
- **Triple-barrier labels** with realized per-trade outcomes (a stop-out and a
  flat expiry are both "label 0" but cost different money), ATR-scaled so the
  model can't just learn which ticker is volatile; daily labels model
  **gap-through fills** (a gap past your stop fills at the open, not the stop).
- **Evaluation in dollars after slippage**, pooled out-of-sample — never
  accuracy.
- **Snapshot-time routing.** The time-of-day router once read the wall clock;
  in a backtest that applied one strategy to two years of bars. The tell:
  a router whose results were byte-identical to one of its components.
  Everything time-dependent now reads the snapshot's own timestamp.
- **80 tests across five suites**, including adversarial risk-gate tests
  (oversized orders, missing account data, kill-switch, PDT edge cases) and a
  proof that observe mode places zero orders (the test executor raises on any
  order attempt).

## Running it

Requires a free Alpaca paper account. Copy `.env.example` → `.env`, add keys.

```
pip install -r requirements.txt
python check_connection.py     # read-only: keys, data feed, quality gate
python test_gate.py            # 28 adversarial risk-gate tests
python dryrun.py --cycles 5    # full chain, synthetic data, mock broker
python main.py                 # the loop (observe mode by default = no orders)
daytrade.bat                   # menu launcher for all of the above
```

Research pipeline:

```
python build_dataset.py --years 2          # intraday dataset, triple-barrier labels
python audit_dataset.py                    # leakage audit — run before believing anything
python train_model.py                      # walk-forward training + deployment verdict
python backtest.py --years 2               # replay history through the live code path
python edge_report.py / shadow_report.py   # expectancy, ranked, from real fills / shadows
```

`SEC Form 4` insider-transaction enrichment (`insider_feed.py`) is public EDGAR
data — set `SEC_USER_AGENT` in `.env` with your contact info as the SEC requires.

## Honest limitations

- Free-tier IEX data is a partial view of national volume; fills at the
  signal bar's close + 2bp are optimistic; no borrow, halts, or overnight risk
  beyond the modeled gaps. All biases here *flatter* the strategies — which
  strengthens the negative result.
- The daily universe is currently-listed symbols only (survivorship bias —
  again flattering, again survived by the conclusion).
- One LLM decider (Claude) exists and runs through the same gate; it was not
  part of the measured comparison because per-cycle inference costs money and
  the free baselines never earned the right to a paid challenger.
