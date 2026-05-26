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

Dynamic gate (any one signal skips the day — thresholds are per-ticker)
------------------------------------------------------------------------
  vrp_low      : IV−RV z-score < threshold  (premium too cheap vs realised vol)
  vix_high     : VIX 252d %-rank > threshold (extreme fear regime)
  vvix_high    : VVIX 252d %-rank > threshold (vol-of-vol spiking)
  skew_extreme : CBOE SKEW z-score > threshold (extreme tail-put demand)
  term_inv     : VIX9D / VIX > threshold    (near-term stress / inverted term structure)

Ticker configuration
--------------------
Each ticker has its own start date, wing-grid, and gate thresholds:

  SPY  start=2023-01-01  standard wings + standard gate
  QQQ  start=2023-01-01  wider wings (higher realised vol) + stricter gate

0DTE availability (confirmed via Theta Data API)
-------------------------------------------------
  SPY   Mon/Wed/Fri ~2016  →  full Mon-Fri 2023-01-01
  QQQ   Mon/Wed/Fri ~2021  →  full Mon-Fri 2023-01-01

Data source
-----------
Intraday option quotes are fetched from a local Theta Data terminal
(http://127.0.0.1:25503/v3) with a file-based cache to avoid redundant calls.
Underlying OHLC and vol-surface signals are pulled from yfinance.

Usage
-----
  python run_0dte_ic.py [--out DIR] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                        [--tickers SPY QQQ]
"""
import argparse, csv, io, json, math
from pathlib import Path
from urllib.parse import urlencode
import urllib.request, urllib.error

import pandas as pd
import numpy as np
from scipy.stats import norm

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("yfinance required: pip install yfinance") from e

# ── Theta Data local terminal ────────────────────────────────────────────────
THETA_BASE = 'http://127.0.0.1:25503/v3'

# ── Per-ticker configuration ─────────────────────────────────────────────────
# underlying          : yfinance symbol for open/close price used in strike sizing
# start               : first date with confirmed full Mon-Fri 0DTE expirations
# n_short_sigma       : candidate short-strike distances (× daily 1-sigma)
# n_wing_sigma        : candidate wing widths (× daily 1-sigma)
# skew_put_mult_max   : max put-wing widening multiplier (SKEW asymmetry)
# skew_put_mult_slope : slope of put-wing mult vs SKEW z-score
#
# Gate thresholds — any one triggered skips the day:
#   vrp_zscore_min    : VRP z-score < this → vol too cheap vs realised
#   vix_pct_max       : VIX 252d %-rank > this → extreme fear regime
#   vix_spike_ratio   : VIX / VIX[5d ago] > this → rapid vol spike (relative)
#   vvix_pct_max      : VVIX 252d %-rank > this → vol-of-vol spike
#   skew_zscore_max   : SKEW z-score > this → extreme tail-put demand
#   term_inv_ratio    : VIX9D / VIX > this → inverted near-term structure
#   gap_skip_pct      : |open/prev_close - 1| > this → large opening gap
#
# Strike-level filters (applied inside select_best_combo per n_short candidate):
#   max_breach_prob   : skip n_short if P(breach) > this using effective sigma
#                       effective_sigma = max(vix_implied, rvol5_daily × rvol_mult)
#   rvol_mult         : realized-vol multiplier for effective sigma
#   min_credit_risk   : skip combo if premium/max_loss < this (quality filter)
TICKERS_CONFIG = {
    'SPY': {
        'underlying': 'SPY', 'start': '2023-01-01',
        'n_short_sigma': [1.0, 1.25, 1.5, 1.75, 2.0],
        # Adaptive wings: calm regime uses tight wings (low peak_margin → high CAGR)
        # elevated regime widens protection (lower per-trade loss → lower DD)
        # Calm = VIX_pct < wing_calm_vix_pct AND rvol5 < vix_sigma
        'n_wing_sigma':       [0.75, 1.0],  # elevated: wider protection
        'n_wing_sigma_calm':  [0.5],        # calm: tightest → min peak_margin
        'wing_calm_vix_pct':  0.50,         # split at VIX 252d median
        'skew_put_mult_max':   1.30,
        'skew_put_mult_slope': 0.20,
        # Environment gate
        'vrp_zscore_min':  -0.5,
        'vix_pct_max':      0.90,
        'vix_spike_ratio':  1.25,
        'vvix_pct_max':     0.85,
        'skew_zscore_max':  1.5,
        'term_inv_ratio':   1.01,
        'gap_skip_pct':     0.005,
        # Breach-probability / credit-quality filters (grid-optimised)
        'max_breach_prob':  0.30,
        'rvol_mult':        1.25,
        'min_credit_risk':  0.03,
        # Risk-budget sizing: target fixed max_loss_usd per trade so peak_margin
        # ≈ 2×cap (both tickers on same day) instead of being VIX-driven (~$1.9M).
        # Lower cap → smaller denominator → higher CAGR.  Set None to use legacy sizing.
        'max_loss_usd_cap': 400_000,
    },
    'QQQ': {
        'underlying': 'QQQ', 'start': '2023-01-01',
        'n_short_sigma': [1.25, 1.5, 1.75, 2.0, 2.25],
        # Adaptive wings
        'n_wing_sigma':       [1.0, 1.25],  # elevated
        'n_wing_sigma_calm':  [0.75],       # calm: tighter → lower peak_margin
        'wing_calm_vix_pct':  0.50,
        'skew_put_mult_max':   1.40,
        'skew_put_mult_slope': 0.25,
        # Environment gate — stricter for tech/macro sensitivity
        'vrp_zscore_min':  -0.3,
        'vix_pct_max':      0.75,
        'vix_spike_ratio':  1.10,
        'vvix_pct_max':     0.80,
        'skew_zscore_max':  1.5,
        'term_inv_ratio':   1.01,
        'gap_skip_pct':     0.007,
        # Breach-probability / credit-quality filters (grid-optimised)
        'max_breach_prob':  0.25,
        'rvol_mult':        1.25,
        'min_credit_risk':  0.03,
        'max_loss_usd_cap': 400_000,
    },
}

DEFAULT_TICKERS = ['SPY', 'QQQ']
DEFAULT_OUT     = Path(__file__).parent / 'results'
DEFAULT_END     = None

DATA_LOOKBACK_YEARS = 2      # extra history before earliest start for rolling calcs
CAPITAL_PER_TICKER_DAY = 10_000.0


# ── Logging ───────────────────────────────────────────────────────────────────

def make_logger(log_path):
    def log(msg):
        line = f"{pd.Timestamp.utcnow().isoformat()} {msg}"
        print(line, flush=True)
        with log_path.open('a') as f:
            f.write(line + '\n')
    return log


# ── Theta Data helpers ────────────────────────────────────────────────────────

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


# ── Strike selection ──────────────────────────────────────────────────────────

def select_best_combo(lkp, strikes, S_open, vix_today, skew_z_today, cfg, sig=None):
    """
    Scan cfg['n_short_sigma'] × cfg['n_wing_sigma'] grid.

    Each n_short candidate is pre-screened by breach probability:
      P(breach) = P(call struck) + P(put struck)
               = norm.sf(n_eff_call) + norm.sf(n_eff_put / put_wing_mult)
    where effective_sigma = max(vix_implied, rvol5 × rvol_mult) blends implied
    and realised vol.  n_eff = short_w / effective_sigma in sigma units.

    Combos with premium/max_loss < min_credit_risk are also rejected.
    Returns combo with best premium/max_loss, or None.
    """
    sig            = sig or {}
    vix_sigma      = (vix_today / 100.0) / math.sqrt(252)
    rvol5_daily    = sig.get('rvol5', vix_sigma)
    eff_sigma      = max(vix_sigma, rvol5_daily * cfg.get('rvol_mult', 1.0))
    # 0DTE breach probability uses intraday sigma only (open→close, ~65% of daily var)
    INTRADAY_VOL_FRAC   = 0.65
    intraday_eff_sigma  = eff_sigma * math.sqrt(INTRADAY_VOL_FRAC)
    max_bp         = cfg.get('max_breach_prob', 1.0)
    min_cr         = cfg.get('min_credit_risk', 0.0)

    skew_z_clamped = max(0.0, min(skew_z_today or 0.0, 1.5))
    put_wing_mult  = min(cfg['skew_put_mult_max'],
                         1.0 + cfg['skew_put_mult_slope'] * skew_z_clamped)

    # Adaptive wing grid: calm regime → tight wings (lower peak_margin, higher CAGR)
    #                     elevated regime → wider protection (lower per-trade loss, lower DD)
    # Calm = VIX percentile below threshold AND realized vol below implied
    vix_pct_sig = sig.get('vix_pct')
    calm_vix_threshold = cfg.get('wing_calm_vix_pct', 0.50)
    is_calm = (
        (vix_pct_sig is None or vix_pct_sig < calm_vix_threshold) and
        rvol5_daily < vix_sigma  # realised vol below implied
    )
    wing_grid = (cfg.get('n_wing_sigma_calm') or cfg['n_wing_sigma']) if is_calm \
                else cfg['n_wing_sigma']

    best       = None
    best_ratio = -np.inf

    for n_short in cfg['n_short_sigma']:
        short_w = n_short * vix_sigma   # distance to short strike as fraction of spot
        # Breach probability using intraday effective sigma (open→close only)
        n_eff_call = short_w / intraday_eff_sigma
        n_eff_put  = short_w / (intraday_eff_sigma * put_wing_mult)
        p_breach   = norm.sf(n_eff_call) + norm.sf(n_eff_put)
        if p_breach > max_bp:
            continue   # too risky given today's realised vol / implied vol
        ksc = nearest_strike(strikes, S_open * (1 + short_w))
        ksp = nearest_strike(strikes, S_open * (1 - short_w))
        if ksc is None or ksp is None or ksc <= S_open or ksp >= S_open:
            continue

        scb = lkp.get((ksc, 'CALL'), {}).get('bid_entry', np.nan)
        spb = lkp.get((ksp, 'PUT'),  {}).get('bid_entry', np.nan)
        if np.isnan(scb) or np.isnan(spb) or scb <= 0 or spb <= 0:
            continue

        for n_wing in wing_grid:
            klc = nearest_strike(strikes, S_open * (1 + short_w + n_wing * vix_sigma))
            klp = nearest_strike(strikes, S_open * (1 - short_w - n_wing * vix_sigma * put_wing_mult))
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
            if premium / max_loss < min_cr:
                continue   # credit too small relative to risk

            ratio = premium / max_loss
            if ratio > best_ratio:
                best_ratio = ratio
                best = dict(ksc=ksc, ksp=ksp, klc=klc, klp=klp,
                            scb=scb, spb=spb, lca=lca, lpa=lpa,
                            premium=premium, max_loss=max_loss, rr_ratio=ratio,
                            n_short=n_short, n_wing=n_wing,
                            daily_sigma_pct=round(vix_sigma * 100, 3),
                            put_wing_mult=round(put_wing_mult, 3),
                            wing_regime='calm' if is_calm else 'elevated')
    return best


# ── Per-day trade execution ───────────────────────────────────────────────────

def run_one(symbol, dt, S_open, S_close, vix_today, skew_z_today, cache_dir, cfg, sig=None):
    df, src = fetch_quote_history(symbol, dt, cache_dir)
    if df is None or df.empty:
        return {'ticker': symbol, 'date': dt, 'traded': False, 'reason': src, 'pnl_usd': 0.0}

    if 'expiration' in df.columns:
        df = df[df['expiration'].astype(str) == dt]
    if df.empty:
        return {'ticker': symbol, 'date': dt, 'traded': False, 'reason': 'no_0dte_rows', 'pnl_usd': 0.0}

    lkp     = build_quote_lookup(df)
    strikes = np.sort(df['strike'].dropna().astype(float).unique())
    best    = select_best_combo(lkp, strikes, S_open, vix_today, skew_z_today, cfg, sig=sig)

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

    close_cost   = (sca_x + spa_x) - (lcb_x + lpb_x)
    pnl_pc       = max(best['premium'] - close_cost, -best['max_loss'])
    # Dynamic position sizing: if max_loss_usd_cap is configured, back-solve n_c so
    # that max_loss_usd ≤ cap on every trade.  This keeps peak_margin ≈ 2×cap
    # regardless of the VIX regime, unlocking higher CAGR from a smaller denominator.
    # Without a cap, n_c = CAPITAL / (spot × 1%) and peak_margin is set by the
    # highest-VIX day we trade (~$1.9M currently).
    ml_cap = cfg.get('max_loss_usd_cap')
    if ml_cap and best['max_loss'] > 0:
        n_c = ml_cap / (best['max_loss'] * 100)
    else:
        n_c = CAPITAL_PER_TICKER_DAY / (S_open * 0.01)
    pnl_usd      = float(pnl_pc * n_c * 100)
    max_loss_usd = float(best['max_loss'] * n_c * 100)   # margin required for this position

    return {'ticker': symbol, 'date': dt, 'traded': True, 'reason': 'trade',
            'pnl_usd': pnl_usd, 'won': pnl_usd > 0,
            'max_loss_usd': max_loss_usd,
            'S_open': S_open, 'S_close': S_close,
            'premium': best['premium'], 'close_cost': close_cost,
            'ksc': ksc, 'ksp': ksp, 'klc': klc, 'klp': klp,
            'n_short': best['n_short'], 'n_wing': best['n_wing'],
            'daily_sigma_pct': best['daily_sigma_pct'],
            'put_wing_mult': best['put_wing_mult'],
            'rr_ratio': best['rr_ratio'], 'source': src}


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(trades):
    df = pd.DataFrame(trades)
    t  = df[df.get('traded', False) == True].copy() if not df.empty else pd.DataFrame()
    if t.empty:
        return {}
    t['date'] = pd.to_datetime(t['date'])

    daily     = t.groupby('date')['pnl_usd'].sum().sort_index()
    cum       = daily.cumsum()
    dd        = cum - cum.cummax()

    # Sharpe (raw $) — mean/std of daily dollar P&L × sqrt(252); depends on sizing scale
    sharpe_dollar = float(daily.mean() / (daily.std() + 1e-9) * math.sqrt(252)) if len(daily) > 1 else 0.0

    # Sharpe (margin-based) — daily return = daily_pnl / peak_margin_deployed
    # peak_margin = max single-day sum of max_loss_usd across all tickers
    if 'max_loss_usd' in t.columns:
        daily_margin = t.groupby('date')['max_loss_usd'].sum().sort_index()
        # reindex to all trading days (non-traded days have 0 margin deployed)
        daily_margin = daily_margin.reindex(daily.index, fill_value=0)
        peak_margin  = float(daily_margin.max())
        if peak_margin > 0:
            daily_ret    = daily / peak_margin
            sharpe_margin = float(daily_ret.mean() / (daily_ret.std() + 1e-9) * math.sqrt(252))
        else:
            sharpe_margin = 0.0
    else:
        peak_margin   = None
        sharpe_margin = None

    by_ticker = (t.groupby('ticker')
                  .agg(trades=('pnl_usd', 'count'),
                       win_rate=('won', 'mean'),
                       total_pnl=('pnl_usd', 'sum'))
                  .round(4).to_dict('index'))
    years = max((t['date'].max() - t['date'].min()).days / 365.25, 1e-6)
    total_pnl = float(t['pnl_usd'].sum())
    if peak_margin and peak_margin > 0 and years > 0:
        total_return   = total_pnl / peak_margin
        base = 1 + total_return
        try:
            cagr_pct = round((base ** (1 / years) - 1) * 100, 2) if base > 0 else None
        except (OverflowError, ValueError):
            cagr_pct = None
        ann_return_pct = round(total_return / years * 100, 2)
    else:
        cagr_pct = ann_return_pct = None

    return dict(rows=len(df), trades=len(t),
                start=str(t['date'].min().date()), end=str(t['date'].max().date()),
                win_rate=float((t['pnl_usd'] > 0).mean()),
                total_pnl=round(total_pnl, 2),
                sharpe=round(sharpe_dollar, 4),
                sharpe_margin=round(sharpe_margin, 4) if sharpe_margin is not None else None,
                peak_margin_usd=round(peak_margin, 2) if peak_margin is not None else None,
                cagr_pct=cagr_pct,
                ann_return_pct=ann_return_pct,
                max_dd=float(dd.min()),
                max_dd_pct=round(float(dd.min()) / peak_margin * 100, 2) if peak_margin else None,
                skipped=int((df.get('traded', False) != True).sum()),
                by_ticker=by_ticker)


# ── Market data helpers ────────────────────────────────────────────────────────

def dl_close(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    s = df['Close']
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


def dl_ohlc(tickers, start, end):
    """Download Open+Close for a list of yfinance tickers. Returns {ticker: {date_str: (O,C)}}."""
    unique = list(set(tickers))
    raw = yf.download(unique, start=start, end=end,
                      auto_adjust=False, progress=False, group_by='ticker')
    if raw.empty:
        return {}
    result = {}
    for sym in unique:
        try:
            o_s = raw[(sym, 'Open')]  if len(unique) > 1 else raw['Open']
            c_s = raw[(sym, 'Close')] if len(unique) > 1 else raw['Close']
            result[sym] = {
                pd.Timestamp(dt).strftime('%Y-%m-%d'): (float(o), float(c))
                for dt, o, c in zip(o_s.index, o_s.values, c_s.values)
                if np.isfinite(float(o)) and np.isfinite(float(c))
            }
        except Exception:
            result[sym] = {}
    return result


# ── Gate computation ───────────────────────────────────────────────────────────

def compute_gate_signals(all_dates, spx_close, vix, vix9d, vvix, skew):
    """
    Returns {date_str: dict} with raw signal values for per-ticker gate evaluation.

    Keys per date: vrp_z, vix_pct, vvix_pct, skew_z, term_inv (bool), vix, skew_z.
    Uses SPX (^GSPC) realised vol for VRP — close proxy for SPY/QQQ.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])

    def align(s):
        if s is None or s.empty:
            return pd.Series(np.nan, index=idx)
        return s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)

    vix_a   = align(vix)
    vix9d_a = align(vix9d)
    vvix_a  = align(vvix)
    skew_a  = align(skew)

    rv21 = (np.log(spx_close / spx_close.shift(1))
            .rolling(21, min_periods=10).std() * math.sqrt(252) * 100)
    vrp  = vix_a - align(rv21)

    def zscore(s, w=252):
        return (s - s.rolling(w, min_periods=20).mean()) / (s.rolling(w, min_periods=20).std() + 1e-9)

    def pct_rank(s, w=252):
        return s.rolling(w, min_periods=20).rank(pct=True)

    vrp_z       = zscore(vrp)
    vix_pct     = pct_rank(vix_a)
    vvix_pct    = pct_rank(vvix_a)
    skew_z      = zscore(skew_a)
    term_inv    = (vix9d_a / (vix_a + 1e-9))           # ratio; threshold applied per-ticker
    vix_5d_ratio = vix_a / (vix_a.shift(5) + 1e-9)    # relative 5-session VIX change

    signals = {}
    for ts in idx:
        d_str = ts.strftime('%Y-%m-%d')

        def get(s):
            val = s.loc[ts] if ts in s.index else np.nan
            return None if pd.isna(val) else float(val)

        signals[d_str] = {
            'vrp_z':          get(vrp_z),
            'vix_pct':        get(vix_pct),
            'vix_5d_ratio':   get(vix_5d_ratio),
            'vvix_pct':       get(vvix_pct),
            'skew_z':         get(skew_z),
            'term_inv_ratio': get(term_inv),
            'vix':            get(vix_a) or 20.0,
        }
    return signals


def apply_gate(sig, cfg):
    """Apply per-ticker gate thresholds to a signal dict. Returns (skip, reason_str).

    Environment signals (computed once for all dates, macro-level):
      vrp_low       : IV−RV premium too cheap
      vix_high      : VIX extreme %-rank
      vix_spike     : VIX jumped >X% relative to 5 sessions ago
      vvix_high     : vol-of-vol spike
      skew_extreme  : tail-put demand extreme
      term_inv      : near-term term structure inverted
    Intraday signals (per-ticker, injected by main loop each day):
      gap           : opening gap too large vs previous close
    """
    reasons = []
    vz  = sig.get('vrp_z');          vz  is not None and vz  < cfg['vrp_zscore_min']   and reasons.append('vrp_low')
    vp  = sig.get('vix_pct');        vp  is not None and vp  > cfg['vix_pct_max']      and reasons.append('vix_high')
    vsr = sig.get('vix_5d_ratio');   vsr is not None and vsr > cfg['vix_spike_ratio']  and reasons.append('vix_spike')
    vvp = sig.get('vvix_pct');       vvp is not None and vvp > cfg['vvix_pct_max']     and reasons.append('vvix_high')
    sz  = sig.get('skew_z');         sz  is not None and sz  > cfg['skew_zscore_max']  and reasons.append('skew_extreme')
    ti  = sig.get('term_inv_ratio'); ti  is not None and ti  > cfg['term_inv_ratio']   and reasons.append('term_inv')
    gp  = sig.get('gap_pct');        gp  is not None and abs(gp) > cfg['gap_skip_pct'] and reasons.append('gap')
    return bool(reasons), ','.join(reasons)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='0DTE Adaptive Iron Condor Backtest')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS,
                        choices=list(TICKERS_CONFIG), metavar='T',
                        help=f'Tickers to trade (default: {DEFAULT_TICKERS})')
    parser.add_argument('--out',   type=Path, default=DEFAULT_OUT)
    parser.add_argument('--start', default=None,
                        help='Override start date for ALL tickers (YYYY-MM-DD)')
    parser.add_argument('--end',   default=DEFAULT_END)
    args = parser.parse_args()

    active = {t: TICKERS_CONFIG[t] for t in args.tickers}
    # --start pushes each ticker's start later but never earlier than its config start
    ticker_starts  = {
        t: (max(args.start, cfg['start']) if args.start else cfg['start'])
        for t, cfg in active.items()
    }
    earliest_start = min(ticker_starts.values())
    data_start     = str(pd.Timestamp(earliest_start) - pd.DateOffset(years=DATA_LOOKBACK_YEARS))[:10]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / 'quote_cache'
    cache_dir.mkdir(exist_ok=True)

    log = make_logger(out / 'run.log')
    (out / 'run.log').write_text('')

    log(f'Tickers: {list(active)} | data from {data_start} | starts: {ticker_starts}')

    underlying_syms = list({cfg['underlying'] for cfg in active.values()})
    log(f'Downloading underlyings {underlying_syms} + signal series from {data_start}')
    px = dl_ohlc(underlying_syms, data_start, args.end)

    spx_close = dl_close('^GSPC', data_start, args.end)
    vix       = dl_close('^VIX',   data_start, args.end)
    vix9d     = dl_close('^VIX9D', data_start, args.end)
    vvix      = dl_close('^VVIX',  data_start, args.end)
    skew      = dl_close('^SKEW',  data_start, args.end)

    log(f'SPX={len(spx_close)} VIX={len(vix)} VIX9D={len(vix9d)} VVIX={len(vvix)} SKEW={len(skew)} rows')

    all_dates = sorted({d for sym_px in px.values() for d in sym_px})
    log(f'Total unique dates: {len(all_dates)}')

    signals = compute_gate_signals(all_dates, spx_close, vix, vix9d, vvix, skew)

    # ── Per-underlying: RVOL5 (5-day rolling daily sigma) and prev-close ────────
    # Used in breach-probability calc and gap gate inside apply_gate.
    close_by_sym = {
        sym: pd.Series({d: c for d, (o, c) in dates.items()}).sort_index()
        for sym, dates in px.items()
    }
    rvol5_by_sym = {
        sym: np.log(s / s.shift(1)).rolling(5, min_periods=3).std()
        for sym, s in close_by_sym.items()
    }
    # Map date_str → rvol5 daily sigma per underlying
    rvol5_lookup = {
        sym: {
            pd.Timestamp(ts).strftime('%Y-%m-%d'): float(v)
            for ts, v in rv.items() if pd.notna(v)
        }
        for sym, rv in rvol5_by_sym.items()
    }
    # Map date_str → prev-close per underlying
    prev_close_lookup = {
        sym: {
            pd.Timestamp(ts).strftime('%Y-%m-%d'): float(s.iloc[i - 1])
            for i, ts in enumerate(s.index) if i > 0
        }
        for sym, s in close_by_sym.items()
    }

    trade_dates = [d for d in all_dates if d >= earliest_start]
    ref_cfg = next(iter(active.values()))
    gated = sum(1 for d in trade_dates if apply_gate(signals.get(d, {}), ref_cfg)[0])
    log(f'Trade-eligible dates: {len(trade_dates)}, gated out (SPY ref, env only): {gated}')

    fields = ['ticker', 'date', 'traded', 'reason', 'pnl_usd', 'won',
              'max_loss_usd', 'S_open', 'S_close', 'premium', 'close_cost',
              'ksc', 'ksp', 'klc', 'klp',
              'n_short', 'n_wing', 'daily_sigma_pct', 'put_wing_mult', 'rr_ratio',
              'wing_regime', 'source']
    trades_path = out / 'trades.csv'
    with trades_path.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields, extrasaction='ignore').writeheader()

    all_rows = []
    for i, dt in enumerate(trade_dates, 1):
        base_sig     = signals.get(dt, {})
        vix_today    = base_sig.get('vix', 20.0)
        skew_z_today = base_sig.get('skew_z') or 0.0

        for sym, cfg in active.items():
            if dt < ticker_starts[sym]:
                continue
            prices = px.get(cfg['underlying'], {}).get(dt)
            if prices is None:
                all_rows.append({'ticker': sym, 'date': dt, 'traded': False,
                                 'reason': 'no_underlying_px', 'pnl_usd': 0.0})
                continue
            S_open, S_close = prices
            und = cfg['underlying']

            # Inject per-ticker intraday signals into a copy of base_sig
            prev_c = prev_close_lookup.get(und, {}).get(dt)
            gap_pct = (S_open / prev_c - 1.0) if prev_c else 0.0
            rvol5   = rvol5_lookup.get(und, {}).get(dt)
            sig = {**base_sig,
                   'gap_pct': gap_pct,
                   'rvol5':   rvol5 if rvol5 is not None else vix_today / 100 / math.sqrt(252)}

            skip, reason = apply_gate(sig, cfg)
            if skip:
                all_rows.append({'ticker': sym, 'date': dt, 'traded': False,
                                 'reason': f'gate:{reason}', 'pnl_usd': 0.0})
            else:
                all_rows.append(run_one(sym, dt, S_open, S_close,
                                        vix_today, skew_z_today, cache_dir, cfg, sig=sig))

        if i % 10 == 0:
            pd.DataFrame(all_rows).to_csv(trades_path, index=False)
            m = metrics(all_rows)
            (out / 'summary.json').write_text(json.dumps(m, indent=2))
            log(f'progress dates={i}/{len(trade_dates)} rows={len(all_rows)} '
                f'trades={m.get("trades", 0)} pnl={m.get("total_pnl", 0):.0f}')

    pd.DataFrame(all_rows).to_csv(trades_path, index=False)
    m = metrics(all_rows)
    (out / 'summary.json').write_text(json.dumps(m, indent=2))
    log(f'FINAL {json.dumps(m)}')

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = pd.DataFrame(all_rows)
        t = t[t['traded'] == True].copy()
        t['date'] = pd.to_datetime(t['date'])

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        daily_all = t.groupby('date')['pnl_usd'].sum().sort_index().cumsum()
        axes[0].plot(daily_all.index, daily_all.values, linewidth=1.5)
        axes[0].set_title('Combined cumulative PnL — all tickers')
        axes[0].set_ylabel('Cumulative PnL ($)')
        axes[0].grid(True, alpha=0.3)

        for sym in active:
            sub = t[t['ticker'] == sym]
            if sub.empty:
                continue
            daily = sub.groupby('date')['pnl_usd'].sum().sort_index().cumsum()
            axes[1].plot(daily.index, daily.values, label=sym, linewidth=1.2)
        axes[1].set_title('Per-ticker cumulative PnL')
        axes[1].set_ylabel('Cumulative PnL ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('Adaptive 0DTE Iron Condor — dynamic sigma wings + multi-signal gate',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(out / 'equity_curve.png', dpi=150)
        plt.close(fig)
        log('plot saved')
    except Exception as e:
        log(f'plot_failed={e}')


if __name__ == '__main__':
    main()
