#!/usr/bin/env python3
"""
0DTE Adaptive Iron Condor — SPY + QQQ
======================================
Sells a same-day-expiry iron condor on SPY and QQQ every eligible trading day.

Strike selection
----------------
Short and long strikes are placed at multiples of the daily 1-sigma move
(derived from VIX each morning), so probability-of-touch stays approximately
constant regardless of the prevailing volatility level.  The (N_short × N_wing)
grid is evaluated and the combination with the best premium / max_loss
reward-to-risk ratio is selected.  Put wings are widened proportionally to the
CBOE SKEW z-score for extra downside protection on skewed days.

Dynamic gate (any one signal skips the day)
--------------------------------------------
  vrp_low      : IV−RV z-score < threshold  (premium too cheap vs realised vol)
  vix_high     : VIX 252d %-rank > threshold (extreme fear regime)
  vvix_high    : VVIX 252d %-rank > threshold (vol-of-vol spiking)
  skew_extreme : CBOE SKEW z-score > threshold (extreme tail-put demand)
  term_inv     : VIX9D / VIX > threshold    (near-term stress / inverted term structure)

Data source
-----------
Intraday option quotes are fetched from a local Theta Data terminal
(http://127.0.0.1:25503/v3) with a file-based cache to avoid redundant calls.
Underlying OHLC and vol-surface signals are pulled from yfinance.

Availability
------------
Full Mon-Fri daily 0DTE expirations confirmed from:
  SPY  : 2023-01-01
  QQQ  : 2023-01-01
  SPXW : ~2022-05-16
  IWM  : ~2024-05-06

Usage
-----
  python run_0dte_ic.py [--out /path/to/output] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import argparse, csv, io, json, math
from pathlib import Path
from urllib.parse import urlencode
import urllib.request, urllib.error

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("yfinance required: pip install yfinance") from e

# ── Theta Data local terminal ────────────────────────────────────────────────
THETA_BASE = 'http://127.0.0.1:25503/v3'

# ── Defaults (overridable via CLI) ───────────────────────────────────────────
DEFAULT_OUT   = Path(__file__).parent / 'results'
DEFAULT_START = '2023-01-01'   # full Mon-Fri 0DTE available for SPY + QQQ from this date
DEFAULT_END   = None           # None = up to latest available

TICKERS    = ['SPY', 'QQQ']
DATA_START_LOOKBACK = 2        # years of extra history before START for rolling calculations
CAPITAL_PER_TICKER_DAY = 10_000.0

# ── Dynamic wing grid (in units of daily 1-sigma) ───────────────────────────
# Short legs placed N_short × (VIX/100/√252) OTM from spot.
# Long legs placed an additional N_wing × (VIX/100/√252) beyond the short legs.
N_SHORT_SIGMA = [1.0, 1.25, 1.5, 1.75, 2.0]
N_WING_SIGMA  = [0.5, 0.75, 1.0, 1.25]

# Put-wing asymmetry: widen put wing when SKEW z-score is positive
SKEW_PUT_MULT_MAX   = 1.30
SKEW_PUT_MULT_SLOPE = 0.20    # mult = 1 + slope × clamp(skew_z, 0, 1.5)

# ── Gate thresholds ──────────────────────────────────────────────────────────
VRP_ZSCORE_MIN  = -0.5
VIX_PCT_MAX     =  0.80
VVIX_PCT_MAX    =  0.85
SKEW_ZSCORE_MAX =  1.5
TERM_INV_RATIO  =  1.01


# ── Logging ──────────────────────────────────────────────────────────────────

def make_logger(log_path):
    def log(msg):
        line = f"{pd.Timestamp.utcnow().isoformat()} {msg}"
        print(line, flush=True)
        with log_path.open('a') as f:
            f.write(line + '\n')
    return log


# ── Theta Data helpers ───────────────────────────────────────────────────────

def http_csv(path, params, timeout=45):
    url = THETA_BASE + path + '?' + urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode('utf-8', 'replace'), None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace'), None
    except Exception as e:
        return None, '', repr(e)


def fetch_quote_history(symbol, dt, cache_dir):
    cp = cache_dir / f'{symbol}_{dt}_quote_1h.csv'
    if cp.exists() and cp.stat().st_size > 50:
        try:
            return pd.read_csv(cp, parse_dates=['timestamp']), 'cache'
        except Exception:
            pass
    status, text, err = http_csv('/option/history/quote',
                                  {'symbol': symbol, 'date': dt, 'expiration': dt, 'interval': '1h'})
    if status != 200:
        return None, f'http_{status}:{(text or err)[:160]}'
    cp.write_text(text)
    try:
        return pd.read_csv(io.StringIO(text), parse_dates=['timestamp']), 'fetched'
    except Exception as e:
        return None, f'parse_error:{e}'


def build_quote_lookup(df):
    """Pre-compute entry (first nonzero) and exit (last nonzero) bid/ask per (strike, right)."""
    lookup = {}
    for key, grp in df.groupby(['strike', 'right']):
        grp = grp.sort_values('timestamp')

        def first_nz(col):
            v = grp[pd.to_numeric(grp[col], errors='coerce').fillna(0) > 0]
            return float(v.iloc[0][col]) if not v.empty else np.nan

        def last_nz(col):
            v = grp[pd.to_numeric(grp[col], errors='coerce').fillna(0) > 0]
            return float(v.iloc[-1][col]) if not v.empty else np.nan

        lookup[(float(key[0]), str(key[1]).upper())] = {
            'bid_entry': first_nz('bid'), 'ask_entry': first_nz('ask'),
            'bid_exit':  last_nz('bid'),  'ask_exit':  last_nz('ask'),
        }
    return lookup


def nearest_strike(arr, target):
    if len(arr) == 0:
        return None
    return float(arr[np.argmin(np.abs(arr - target))])


# ── Strike selection ─────────────────────────────────────────────────────────

def select_best_combo(lkp, strikes, S_open, vix_today, skew_z_today):
    """
    Scan N_SHORT_SIGMA × N_WING_SIGMA grid with sigma-scaled targets.
    Short strikes placed at n_short × daily_sigma OTM; long legs an additional
    n_wing × daily_sigma beyond (put side widened by skew multiplier).
    Returns the combo with best premium / max_loss, or None.
    """
    daily_sigma    = (vix_today / 100.0) / math.sqrt(252)
    skew_z_clamped = max(0.0, min(skew_z_today or 0.0, 1.5))
    put_wing_mult  = min(SKEW_PUT_MULT_MAX, 1.0 + SKEW_PUT_MULT_SLOPE * skew_z_clamped)

    best       = None
    best_ratio = -np.inf

    for n_short in N_SHORT_SIGMA:
        short_w = n_short * daily_sigma
        ksc = nearest_strike(strikes, S_open * (1 + short_w))
        ksp = nearest_strike(strikes, S_open * (1 - short_w))
        if ksc is None or ksp is None or ksc <= S_open or ksp >= S_open:
            continue

        scb = lkp.get((ksc, 'CALL'), {}).get('bid_entry', np.nan)
        spb = lkp.get((ksp, 'PUT'),  {}).get('bid_entry', np.nan)
        if np.isnan(scb) or np.isnan(spb) or scb <= 0 or spb <= 0:
            continue

        for n_wing in N_WING_SIGMA:
            klc = nearest_strike(strikes, S_open * (1 + short_w + n_wing * daily_sigma))
            klp = nearest_strike(strikes, S_open * (1 - short_w - n_wing * daily_sigma * put_wing_mult))
            if klc is None or klp is None or klc <= ksc or klp >= ksp:
                continue

            lca = lkp.get((klc, 'CALL'), {}).get('ask_entry', np.nan)
            lpa = lkp.get((klp, 'PUT'),  {}).get('ask_entry', np.nan)
            if np.isnan(lca) or np.isnan(lpa):
                continue

            premium  = (scb + spb) - (lca + lpa)
            if premium <= 0:
                continue
            spread_w = max(klc - ksc, ksp - klp)
            max_loss = spread_w - premium
            if max_loss <= 0:
                continue

            ratio = premium / max_loss
            if ratio > best_ratio:
                best_ratio = ratio
                best = dict(ksc=ksc, ksp=ksp, klc=klc, klp=klp,
                            scb=scb, spb=spb, lca=lca, lpa=lpa,
                            premium=premium, max_loss=max_loss, rr_ratio=ratio,
                            n_short=n_short, n_wing=n_wing,
                            daily_sigma_pct=round(daily_sigma * 100, 3),
                            put_wing_mult=round(put_wing_mult, 3))
    return best


# ── Per-day trade execution ───────────────────────────────────────────────────

def run_one(symbol, dt, S_open, S_close, vix_today, skew_z_today, cache_dir):
    df, src = fetch_quote_history(symbol, dt, cache_dir)
    if df is None or df.empty:
        return {'ticker': symbol, 'date': dt, 'traded': False, 'reason': src, 'pnl_usd': 0.0}

    if 'expiration' in df.columns:
        df = df[df['expiration'].astype(str) == dt]
    if df.empty:
        return {'ticker': symbol, 'date': dt, 'traded': False, 'reason': 'no_0dte_rows', 'pnl_usd': 0.0}

    lkp     = build_quote_lookup(df)
    strikes = np.sort(df['strike'].dropna().astype(float).unique())
    best    = select_best_combo(lkp, strikes, S_open, vix_today, skew_z_today)

    if best is None:
        return {'ticker': symbol, 'date': dt, 'traded': False, 'reason': 'no_valid_combo', 'pnl_usd': 0.0}

    ksc, ksp, klc, klp = best['ksc'], best['ksp'], best['klc'], best['klp']
    sca_x = lkp.get((ksc, 'CALL'), {}).get('ask_exit', np.nan)
    spa_x = lkp.get((ksp, 'PUT'),  {}).get('ask_exit', np.nan)
    lcb_x = lkp.get((klc, 'CALL'), {}).get('bid_exit', np.nan)
    lpb_x = lkp.get((klp, 'PUT'),  {}).get('bid_exit', np.nan)

    if any(np.isnan(v) for v in [sca_x, spa_x, lcb_x, lpb_x]):
        return {'ticker': symbol, 'date': dt, 'traded': False, 'reason': 'missing_exit_quotes',
                'pnl_usd': 0.0, 'ksc': ksc, 'ksp': ksp, 'klc': klc, 'klp': klp}

    close_cost = (sca_x + spa_x) - (lcb_x + lpb_x)
    pnl_pc     = max(best['premium'] - close_cost, -best['max_loss'])
    n_c        = CAPITAL_PER_TICKER_DAY / (S_open * 0.01)
    pnl_usd    = float(pnl_pc * n_c * 100)

    return {'ticker': symbol, 'date': dt, 'traded': True, 'reason': 'trade',
            'pnl_usd': pnl_usd, 'won': pnl_usd > 0,
            'S_open': S_open, 'S_close': S_close,
            'premium': best['premium'], 'close_cost': close_cost,
            'ksc': ksc, 'ksp': ksp, 'klc': klc, 'klp': klp,
            'n_short': best['n_short'], 'n_wing': best['n_wing'],
            'daily_sigma_pct': best['daily_sigma_pct'],
            'put_wing_mult': best['put_wing_mult'],
            'rr_ratio': best['rr_ratio'], 'source': src}


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades):
    df = pd.DataFrame(trades)
    t  = df[df.get('traded', False) == True].copy() if not df.empty else pd.DataFrame()
    if t.empty:
        return {}
    t['date'] = pd.to_datetime(t['date'])
    daily  = t.groupby('date')['pnl_usd'].sum().sort_index()
    cum    = daily.cumsum()
    dd     = cum - cum.cummax()
    sharpe = float(daily.mean() / (daily.std() + 1e-9) * math.sqrt(252)) if len(daily) > 1 else 0.0
    return dict(rows=len(df), trades=len(t),
                start=str(t['date'].min().date()), end=str(t['date'].max().date()),
                win_rate=float((t['pnl_usd'] > 0).mean()),
                total_pnl=float(t['pnl_usd'].sum()),
                sharpe=sharpe, max_dd=float(dd.min()),
                skipped=int((df.get('traded', False) != True).sum()))


# ── Market data helpers ───────────────────────────────────────────────────────

def dl_close(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    s = df['Close']
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


# ── Gate computation ──────────────────────────────────────────────────────────

def compute_gate_signals(dates, spy_close, vix, vix9d, vvix, skew):
    """
    Returns {date_str: (skip, reasons, vix_val, skew_z_val)}.

    Five signals — any one triggers a skip:
      vrp_low      : VRP z-score < VRP_ZSCORE_MIN   (selling vol too cheap vs realised)
      vix_high     : VIX 252d %-rank > VIX_PCT_MAX   (extreme fear)
      vvix_high    : VVIX 252d %-rank > VVIX_PCT_MAX (vol-of-vol spiking)
      skew_extreme : SKEW z-score > SKEW_ZSCORE_MAX  (extreme tail-put demand)
      term_inv     : VIX9D/VIX > TERM_INV_RATIO      (inverted term structure)

    vix_val and skew_z_val are passed through to select_best_combo for
    dynamic wing sizing.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in dates])

    def align(s):
        if s is None or s.empty:
            return pd.Series(np.nan, index=idx)
        return s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)

    vix_a   = align(vix)
    vix9d_a = align(vix9d)
    vvix_a  = align(vvix)
    skew_a  = align(skew)

    rv21 = (np.log(spy_close / spy_close.shift(1))
            .rolling(21, min_periods=10).std() * math.sqrt(252) * 100)
    vrp  = vix_a - align(rv21)

    def zscore(s, w=252):
        return (s - s.rolling(w, min_periods=20).mean()) / (s.rolling(w, min_periods=20).std() + 1e-9)

    def pct_rank(s, w=252):
        return s.rolling(w, min_periods=20).rank(pct=True)

    vrp_z    = zscore(vrp)
    vix_pct  = pct_rank(vix_a)
    vvix_pct = pct_rank(vvix_a)
    skew_z   = zscore(skew_a)
    term_inv = (vix9d_a / (vix_a + 1e-9)) > TERM_INV_RATIO

    gate = {}
    for ts, (d_str, _) in zip(idx, dates):
        def get(s):
            val = s.loc[ts] if ts in s.index else np.nan
            return None if pd.isna(val) else float(val)

        reasons = []
        vz  = get(vrp_z);    vz  is not None and vz  < VRP_ZSCORE_MIN  and reasons.append('vrp_low')
        vp  = get(vix_pct);  vp  is not None and vp  > VIX_PCT_MAX     and reasons.append('vix_high')
        vvp = get(vvix_pct); vvp is not None and vvp > VVIX_PCT_MAX    and reasons.append('vvix_high')
        sz  = get(skew_z);   sz  is not None and sz  > SKEW_ZSCORE_MAX and reasons.append('skew_extreme')
        ti  = term_inv.loc[ts] if ts in term_inv.index else False
        bool(ti) and reasons.append('term_inv')

        gate[d_str] = (bool(reasons), ','.join(reasons),
                       get(vix_a) or 20.0,
                       get(skew_z) or 0.0)
    return gate


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='0DTE Adaptive Iron Condor Backtest')
    parser.add_argument('--out',   type=Path, default=DEFAULT_OUT,   help='Output directory')
    parser.add_argument('--start', default=DEFAULT_START,            help='First trading date (YYYY-MM-DD)')
    parser.add_argument('--end',   default=DEFAULT_END,              help='Last trading date (YYYY-MM-DD), default=latest')
    args = parser.parse_args()

    out   = args.out
    start = args.start
    end   = args.end

    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / 'quote_cache'
    cache_dir.mkdir(exist_ok=True)

    log = make_logger(out / 'run.log')
    (out / 'run.log').write_text('')

    # Extra lookback so 252d rolling windows are warm by START
    data_start = str(pd.Timestamp(start) - pd.DateOffset(years=DATA_START_LOOKBACK))[:10]

    log(f'Downloading market data from {data_start}, trading from {start}')
    px     = yf.download(TICKERS,  start=data_start, end=end, auto_adjust=False, progress=False, group_by='ticker')
    spy_cl = dl_close('SPY',       data_start, end)
    vix    = dl_close('^VIX',      data_start, end)
    vix9d  = dl_close('^VIX9D',    data_start, end)
    vvix   = dl_close('^VVIX',     data_start, end)
    skew   = dl_close('^SKEW',     data_start, end)

    if px.empty:
        raise RuntimeError('No underlying data from yfinance')

    log(f'VIX={len(vix)} VIX9D={len(vix9d)} VVIX={len(vvix)} SKEW={len(skew)} rows')

    dates = []
    for dt in px.index:
        ok, row = True, {}
        for sym in TICKERS:
            try:
                o = float(px[(sym, 'Open')].loc[dt])
                c = float(px[(sym, 'Close')].loc[dt])
                if not np.isfinite(o + c):
                    ok = False
                row[sym] = (o, c)
            except Exception:
                ok = False
        if ok:
            dates.append((pd.Timestamp(dt).strftime('%Y-%m-%d'), row))

    log(f'Total yfinance dates: {len(dates)}')

    gate = compute_gate_signals(dates, spy_cl, vix, vix9d, vvix, skew)

    trade_dates = [(d, r) for d, r in dates if d >= start]
    gated_count = sum(1 for d, _ in trade_dates if gate.get(d, (False,))[0])
    log(f'Trading dates: {len(trade_dates)}, gated out: {gated_count}')

    fields = ['ticker', 'date', 'traded', 'reason', 'pnl_usd', 'won',
              'S_open', 'S_close', 'premium', 'close_cost',
              'ksc', 'ksp', 'klc', 'klp',
              'n_short', 'n_wing', 'daily_sigma_pct', 'put_wing_mult', 'rr_ratio', 'source']
    trades_path = out / 'trades.csv'
    with trades_path.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields, extrasaction='ignore').writeheader()

    all_rows = []
    for i, (dt, row) in enumerate(trade_dates, 1):
        skip, reason, vix_today, skew_z_today = gate.get(dt, (False, '', 20.0, 0.0))
        if skip:
            for sym in TICKERS:
                all_rows.append({'ticker': sym, 'date': dt, 'traded': False,
                                 'reason': f'gate:{reason}', 'pnl_usd': 0.0})
        else:
            for sym in TICKERS:
                all_rows.append(run_one(sym, dt, *row[sym], vix_today, skew_z_today, cache_dir))

        if i % 10 == 0:
            pd.DataFrame(all_rows).to_csv(trades_path, index=False)
            m = metrics(all_rows)
            (out / 'summary.json').write_text(json.dumps(m, indent=2))
            log(f'progress dates={i}/{len(trade_dates)} rows={len(all_rows)} {m}')

    pd.DataFrame(all_rows).to_csv(trades_path, index=False)
    m = metrics(all_rows)
    (out / 'summary.json').write_text(json.dumps(m, indent=2))
    log(f'FINAL {m}')

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = pd.DataFrame(all_rows)
        t = t[t['traded'] == True].copy()
        t['date'] = pd.to_datetime(t['date'])
        daily = t.groupby('date')['pnl_usd'].sum().sort_index()
        cum   = daily.cumsum()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cum.index, cum.values)
        ax.set_title('Adaptive 0DTE IC SPY+QQQ — dynamic sigma wings + multi-signal gate')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Cumulative PnL ($)')
        fig.tight_layout()
        fig.savefig(out / 'equity_curve.png', dpi=150)
        plt.close(fig)
        log('plot saved')
    except Exception as e:
        log(f'plot_failed={e}')


if __name__ == '__main__':
    main()
