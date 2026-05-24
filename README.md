# 0dte-usindex

0DTE (same-day expiry) options strategies on US index ETFs and SPX.

## Strategies

### Iron Condor — SPY + QQQ
[`strategies/iron_condor/run_0dte_ic.py`](strategies/iron_condor/run_0dte_ic.py)

Sells a daily iron condor on SPY and QQQ using:

- **Dynamic sigma wings** — short/long strikes placed at multiples of the daily
  1-sigma move (`VIX/100/√252`), keeping probability-of-touch approximately
  constant across vol regimes. Put wings are widened proportionally to the CBOE
  SKEW z-score for asymmetric downside protection.

- **Multi-signal gate** — skips days when any of five conditions hold:

  | Signal | Condition | Rationale |
  |---|---|---|
  | VRP z-score | < −0.5 | Selling vol too cheap vs realised |
  | VIX 252d %-rank | > 80th pct | Extreme fear regime |
  | VVIX 252d %-rank | > 85th pct | Vol-of-vol spiking |
  | CBOE SKEW z-score | > 1.5 | Extreme tail-put demand |
  | VIX9D / VIX | > 1.01 | Inverted near-term term structure |

#### Backtest results (2023-01-12 → 2026-05-22)

| Metric | Value |
|---|---|
| Trades | 473 |
| Win rate | 80.1% |
| Total P&L | +$4.84M |
| Sharpe | 2.03 |
| Max drawdown | −$1.27M |
| Capital per ticker/day | $10,000 |

## 0DTE availability by ticker

| Ticker | Mon/Wed/Fri | Full Mon–Fri |
|---|---|---|
| SPY | ~2016 | 2023-01-01 |
| QQQ | ~2021 | 2023-01-01 |
| SPXW | ~2021 | ~2022-05-16 |
| IWM | ~2022 | ~2024-05-06 |

## Requirements

```
yfinance
pandas
numpy
matplotlib
```

A running **Theta Data terminal** is required for intraday option quote data
(`http://127.0.0.1:25503/v3`). Quotes are cached to disk after the first fetch.

## Usage

```bash
pip install -r requirements.txt

# Run with defaults (SPY+QQQ, 2023-01-01 to latest)
python strategies/iron_condor/run_0dte_ic.py

# Custom date range and output directory
python strategies/iron_condor/run_0dte_ic.py \
  --start 2024-01-01 \
  --end   2025-12-31 \
  --out   /path/to/results
```

Output written to `--out` directory:
- `trades.csv` — per-trade log with strikes, premiums, P&L, sigma parameters
- `summary.json` — aggregated metrics
- `equity_curve.png` — cumulative P&L chart
- `run.log` — timestamped progress log
- `quote_cache/` — cached Theta Data responses (reused on re-runs)
