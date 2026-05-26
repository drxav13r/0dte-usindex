#!/usr/bin/env python3
"""
Per-ticker parameter grid-search for 0DTE Iron Condor.

Pre-loads all cached option CSVs into memory once, then sweeps parameter
combinations in pure Python (no IO in the inner loop). Reports top-N
combinations per ticker ranked by Sharpe ratio.

Two sweep modes:
  --mode combo  : max_breach_prob × min_credit_risk × rvol_mult × n_short grid
  --mode gate   : vix_pct_max × vix_spike_ratio (requires wide-gate cache)

Usage:
  python optimize_ic.py [--cache-dir DIR] [--top N] [--min-trades M] [--mode combo|gate]
"""
import sys, math, itertools, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_0dte_ic import (
    TICKERS_CONFIG, CAPITAL_PER_TICKER_DAY, DATA_LOOKBACK_YEARS,
    build_quote_lookup, select_best_combo, apply_gate,
    dl_ohlc, dl_close, compute_gate_signals,
)

DEFAULT_CACHE = Path('/tmp/0dte_v3_results/quote_cache')
MIN_TRADES    = 30

# ── Per-ticker parameter grids ────────────────────────────────────────────────
# 'combo' mode: breach-prob / credit-quality / rvol_mult — all use cached dates
COMBO_GRIDS = {
    'SPY': {
        'max_breach_prob': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        'min_credit_risk': [0.02, 0.03, 0.04, 0.05],
        'rvol_mult':       [0.50, 0.75, 1.00, 1.25],
        'n_short_sigma':   [
            [1.0, 1.25, 1.5, 1.75, 2.0],          # v3 default
            [1.25, 1.5, 1.75, 2.0],                # drop tightest
            [1.0, 1.25, 1.5, 1.75],                # drop widest
        ],
    },
    'QQQ': {
        'max_breach_prob': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'min_credit_risk': [0.02, 0.03, 0.04, 0.05],
        'rvol_mult':       [0.50, 0.75, 1.00, 1.25],
        'n_short_sigma':   [
            [1.25, 1.5, 1.75, 2.0, 2.25],          # v3 default
            [1.5, 1.75, 2.0, 2.25],                # drop tightest
            [1.25, 1.5, 1.75, 2.0],                # drop widest
        ],
    },
}

# 'gate' mode: environment thresholds — only useful if cache covers gated-out dates
GATE_GRIDS = {
    'SPY': {
        'vix_pct_max':     [0.70, 0.75, 0.80, 0.85, 0.90],
        'vix_spike_ratio': [1.15, 1.20, 1.25, 1.30, 1.35],
        'gap_skip_pct':    [0.005, 0.007, 0.010, 0.015],
    },
    'QQQ': {
        'vix_pct_max':     [0.65, 0.70, 0.75, 0.80, 0.85],
        'vix_spike_ratio': [1.10, 1.15, 1.20, 1.25, 1.30],
        'gap_skip_pct':    [0.005, 0.007, 0.010, 0.015],
    },
}


# ── Cache loader ──────────────────────────────────────────────────────────────

def load_cache(cache_dir):
    """Pre-load all cached option CSVs → {(ticker, date): (lkp, strikes)}."""
    data  = {}
    files = sorted(cache_dir.glob('*_*_quote_1h.csv'))
    print(f'Loading {len(files)} cached CSVs ...', flush=True)
    for f in files:
        stem  = f.stem                      # e.g. 'SPY_2023-01-12_quote_1h'
        parts = stem.split('_')
        if len(parts) < 2:
            continue
        ticker = parts[0]
        dt     = parts[1]
        if ticker not in TICKERS_CONFIG:
            continue
        try:
            df = pd.read_csv(f, parse_dates=['timestamp'])
        except Exception:
            continue
        if 'expiration' in df.columns:
            df = df[df['expiration'].astype(str) == dt]
        if df.empty:
            continue
        lkp     = build_quote_lookup(df)
        strikes = np.sort(df['strike'].dropna().astype(float).unique())
        data[(ticker, dt)] = (lkp, strikes)
    print(f'Loaded {len(data)} (ticker, date) pairs.', flush=True)
    return data


# ── Per-combination runner ────────────────────────────────────────────────────

def run_combo(ticker, cfg, all_dates, px_map, signals,
              prev_close, rvol5, cache_data):
    """Run one (ticker, cfg) combo. Returns metrics dict or None if < MIN_TRADES."""
    daily_pnl = {}
    daily_ml  = {}
    wins = total = 0

    for dt in all_dates:
        if dt < cfg['start']:
            continue
        prices = px_map.get(dt)
        if prices is None:
            continue
        S_open, _ = prices
        base_sig  = signals.get(dt, {})
        vix_today = base_sig.get('vix', 20.0)
        skew_z    = base_sig.get('skew_z') or 0.0

        prev_c = prev_close.get(dt)
        gap    = (S_open / prev_c - 1.0) if prev_c else 0.0
        rv5    = rvol5.get(dt)
        sig    = {**base_sig, 'gap_pct': gap,
                  'rvol5': rv5 if rv5 is not None else vix_today / 100 / math.sqrt(252)}

        skip, _ = apply_gate(sig, cfg)
        if skip:
            continue

        if (ticker, dt) not in cache_data:
            continue
        lkp, strikes = cache_data[(ticker, dt)]
        best = select_best_combo(lkp, strikes, S_open, vix_today, skew_z, cfg, sig=sig)
        if best is None:
            continue

        ksc, ksp, klc, klp = best['ksc'], best['ksp'], best['klc'], best['klp']
        sca_x = lkp.get((ksc, 'CALL'), {}).get('ask_exit', float('nan'))
        spa_x = lkp.get((ksp, 'PUT'),  {}).get('ask_exit', float('nan'))
        lcb_x = lkp.get((klc, 'CALL'), {}).get('bid_exit', float('nan'))
        lpb_x = lkp.get((klp, 'PUT'),  {}).get('bid_exit', float('nan'))
        if any(math.isnan(v) for v in [sca_x, spa_x, lcb_x, lpb_x]):
            continue

        close_cost = (sca_x + spa_x) - (lcb_x + lpb_x)
        pnl_pc     = max(best['premium'] - close_cost, -best['max_loss'])
        n_c        = CAPITAL_PER_TICKER_DAY / (S_open * 0.01)
        pnl_usd    = pnl_pc * n_c * 100
        ml_usd     = best['max_loss'] * n_c * 100

        daily_pnl[dt] = daily_pnl.get(dt, 0.0) + pnl_usd
        daily_ml[dt]  = daily_ml.get(dt, 0.0)  + ml_usd
        total += 1
        wins  += int(pnl_usd > 0)

    if total < 1:
        return None

    s_pnl       = pd.Series(daily_pnl).sort_index()
    cum         = s_pnl.cumsum()
    dd          = float((cum - cum.cummax()).min())
    peak_margin = max(daily_ml.values()) if daily_ml else 1.0
    total_pnl   = float(s_pnl.sum())

    daily_vals  = list(daily_pnl.values())
    mean_d, std_d = float(np.mean(daily_vals)), float(np.std(daily_vals))
    sharpe      = mean_d / (std_d + 1e-9) * math.sqrt(252) if std_d > 1e-9 else 0.0

    dates_list = sorted(daily_pnl)
    years = max((pd.Timestamp(dates_list[-1]) - pd.Timestamp(dates_list[0])).days / 365.25, 1e-6)
    try:
        cagr = round(((1 + total_pnl / peak_margin) ** (1 / years) - 1) * 100, 1)
    except Exception:
        cagr = None

    return dict(
        trades    = total,
        win_rate  = round(total and wins / total, 3),
        total_pnl = round(total_pnl),
        sharpe    = round(sharpe, 3),
        max_dd_pct= round(dd / peak_margin * 100, 1) if peak_margin > 0 else None,
        cagr_pct  = cagr,
        peak_margin = round(peak_margin),
    )


# ── Market data loader (shared) ───────────────────────────────────────────────

def load_market_data(tickers):
    earliest = min(TICKERS_CONFIG[t]['start'] for t in tickers)
    data_start = str(pd.Timestamp(earliest) - pd.DateOffset(years=DATA_LOOKBACK_YEARS))[:10]
    print(f'Downloading market data from {data_start} ...', flush=True)

    px        = dl_ohlc(tickers, data_start, None)
    spx_close = dl_close('^GSPC',  data_start, None)
    vix       = dl_close('^VIX',   data_start, None)
    vix9d     = dl_close('^VIX9D', data_start, None)
    vvix      = dl_close('^VVIX',  data_start, None)
    skew      = dl_close('^SKEW',  data_start, None)

    all_dates = sorted({d for sym_px in px.values() for d in sym_px})
    signals   = compute_gate_signals(all_dates, spx_close, vix, vix9d, vvix, skew)

    close_by_sym = {
        sym: pd.Series({d: c for d, (o, c) in dates.items()}).sort_index()
        for sym, dates in px.items()
    }
    rvol5_lookup = {
        sym: {pd.Timestamp(ts).strftime('%Y-%m-%d'): float(v)
              for ts, v in np.log(s / s.shift(1)).rolling(5, min_periods=3).std().items()
              if pd.notna(v)}
        for sym, s in close_by_sym.items()
    }
    prev_close_lookup = {
        sym: {pd.Timestamp(ts).strftime('%Y-%m-%d'): float(s.iloc[i - 1])
              for i, ts in enumerate(s.index) if i > 0}
        for sym, s in close_by_sym.items()
    }
    return px, signals, rvol5_lookup, prev_close_lookup, all_dates


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',   type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--top',         type=int,  default=15)
    parser.add_argument('--min-trades',  type=int,  default=MIN_TRADES)
    parser.add_argument('--mode',        choices=['combo', 'gate'], default='combo',
                        help='"combo": breach-prob/credit/rvol params (uses cached dates). '
                             '"gate": env-gate params (requires wide-gate cache).')
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ'])
    args = parser.parse_args()

    cache_data = load_cache(args.cache_dir)
    px, signals, rvol5, prev_close, all_dates = load_market_data(args.tickers)

    grids = COMBO_GRIDS if args.mode == 'combo' else GATE_GRIDS
    all_results = {}

    for ticker in args.tickers:
        base_cfg = TICKERS_CONFIG[ticker].copy()
        px_map   = px.get(ticker, {})
        rv5      = rvol5.get(ticker, {})
        prc      = prev_close.get(ticker, {})
        grid     = grids[ticker]

        keys   = list(grid)
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f'\n=== {ticker} | {args.mode} sweep | {len(combos)} combinations ===', flush=True)

        rows = []
        for i, vals in enumerate(combos, 1):
            overrides = dict(zip(keys, vals))
            cfg = {**base_cfg, **overrides}
            m   = run_combo(ticker, cfg, all_dates, px_map, signals, prc, rv5, cache_data)
            if m and m['trades'] >= args.min_trades:
                # Stringify list params for display
                display = {k: (str(v) if isinstance(v, list) else v) for k, v in overrides.items()}
                rows.append({**display, **m})
            if i % 100 == 0 or i == len(combos):
                print(f'  {i}/{len(combos)} done, {len(rows)} valid', flush=True)

        if not rows:
            print(f'  No valid combinations for {ticker}', flush=True)
            continue

        df = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
        print(f'\n--- Top {args.top} for {ticker} (Sharpe, min {args.min_trades} trades) ---')
        pd.set_option('display.width', 200)
        pd.set_option('display.max_columns', 20)
        print(df.head(args.top).to_string(index=False))
        all_results[ticker] = df.head(args.top).to_dict('records')

    out = args.cache_dir.parent / f'optimize_{args.mode}_results.json'
    out.write_text(json.dumps(all_results, indent=2))
    print(f'\nResults → {out}')


if __name__ == '__main__':
    main()
