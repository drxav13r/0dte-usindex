#!/usr/bin/env python3
"""
compute_skew_signals.py
=======================
Signal computation pipeline for 0DTE iron condor gating and dynamic wing sizing.

Signals computed per (trade_date, ticker):
  rr_1w        : Risk Reversal 1w = IV(25Δ put, 1w) − IV(25Δ call, 1w)
  bf_1w        : Butterfly 1w     = [IV(25Δ put, 1w) + IV(25Δ call, 1w)] / 2 − IV(ATM, 1w)
  atm_iv_1w    : ATM implied vol at 1w expiry
  atm_iv_1m    : ATM implied vol at 1m expiry
  ts_slope     : Term-structure slope = atm_iv_1w / atm_iv_1m
  corr_spot_iv_21d : 21d rolling corr(SPY daily return, VIX daily change)

Usage:
  python compute_skew_signals.py --ticker SPY [--cache_dir /tmp/0dte_skew_cache]
  python compute_skew_signals.py --ticker QQQ
  python compute_skew_signals.py --all  # process SPY and QQQ
"""

import os
import re
import math
import warnings
import argparse
import logging
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path("/tmp/0dte_skew_cache")
RISK_FREE_RATE = 0.05          # approximate annual risk-free rate
TARGET_DELTA = 0.25            # 25-delta targets for RR / BF
IV_LOWER = 0.01                # brentq lower bound for IV solve
IV_UPPER = 5.0                 # brentq upper bound for IV solve
MAX_DTE_1W = 14                # days: <= 14d → 1w expiry bucket
MIN_DTE_1W = 2                 # must have at least 2 calendar days to expiry
MIN_DTE_1M = 15                # minimum days for 1m bucket
MAX_DTE_1M = 50                # maximum days for 1m bucket


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS call delta in (0, 1)."""
    return norm.cdf(_d1(S, K, T, r, sigma))


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS put delta in (-1, 0); return abs value so 0.25 means 25-delta."""
    return abs(norm.cdf(_d1(S, K, T, r, sigma)) - 1.0)


def implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    market_mid: float,
    right: str,
) -> Optional[float]:
    """
    Solve BS IV via Brentq.

    Returns None if market_mid is too small/large for a valid IV in [IV_LOWER, IV_UPPER].
    """
    if T <= 0 or market_mid <= 0:
        return None

    price_fn = bs_call_price if right.upper() == "CALL" else bs_put_price

    try:
        # Check that root is bracketed
        lo_price = price_fn(S, K, T, r, IV_LOWER) - market_mid
        hi_price = price_fn(S, K, T, r, IV_UPPER) - market_mid
        if lo_price * hi_price > 0:
            return None
        sigma = brentq(
            lambda v: price_fn(S, K, T, r, v) - market_mid,
            IV_LOWER,
            IV_UPPER,
            xtol=1e-6,
            maxiter=100,
        )
        return sigma if 0 < sigma < IV_UPPER else None
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# 25-delta strike finder
# ---------------------------------------------------------------------------

def find_25delta_strike(
    strikes_with_iv: list[tuple[float, float]],
    S: float,
    T: float,
    r: float,
    right: str,
) -> Optional[tuple[float, float]]:
    """
    Given a list of (strike, iv) pairs, find the strike whose BS delta
    is closest to 0.25 (for puts, use |put_delta|; for calls use call_delta).

    Returns (best_strike, best_iv) or None.
    """
    delta_fn = bs_put_delta if right.upper() == "PUT" else bs_call_delta
    best = None
    best_dist = float("inf")
    for K, iv in strikes_with_iv:
        if iv is None or iv <= 0:
            continue
        try:
            d = delta_fn(S, K, T, r, iv)
            dist = abs(d - TARGET_DELTA)
            if dist < best_dist:
                best_dist = dist
                best = (K, iv)
        except Exception:
            continue
    return best


# ---------------------------------------------------------------------------
# Per-date, per-expiry IV surface computation
# ---------------------------------------------------------------------------

def compute_surface_signals(
    snap: pd.DataFrame,
    spot: float,
    trade_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
) -> dict:
    """
    Given a single-timestamp option snapshot for one expiry, compute:
      atm_iv, iv_25d_call, iv_25d_put, rr, bf

    snap columns: strike, right, bid, ask   (already filtered: bid>0, ask>0)
    Returns dict with keys: atm_iv, iv_25d_call, iv_25d_put, rr, bf
    """
    T = (expiry_date - trade_date).days / 365.0
    if T < 1 / 365:
        return {}

    r = RISK_FREE_RATE
    snap = snap.copy()
    snap["mid"] = (snap["bid"] + snap["ask"]) / 2.0

    # ATM strike: closest to spot
    all_strikes = snap["strike"].unique()
    atm_strike = float(all_strikes[np.argmin(np.abs(all_strikes - spot))])

    # Build IV curve: for each (strike, right) compute IV
    iv_map = {}  # (strike, right) -> iv
    for _, row in snap.iterrows():
        K = float(row["strike"])
        right = row["right"].upper()
        iv = implied_vol(spot, K, T, r, row["mid"], right)
        if iv is not None:
            iv_map[(K, right)] = iv

    if not iv_map:
        return {}

    # ATM IV: prefer call/put average at ATM strike
    atm_call_iv = iv_map.get((atm_strike, "CALL"))
    atm_put_iv = iv_map.get((atm_strike, "PUT"))
    if atm_call_iv is not None and atm_put_iv is not None:
        atm_iv = (atm_call_iv + atm_put_iv) / 2.0
    elif atm_call_iv is not None:
        atm_iv = atm_call_iv
    elif atm_put_iv is not None:
        atm_iv = atm_put_iv
    else:
        return {}

    # Build sorted lists for 25-delta search
    # Calls: OTM calls (K > spot), Puts: OTM puts (K < spot)
    # But use all strikes to avoid missing the 25-delta region
    call_ivs = [(K, iv) for (K, rt), iv in iv_map.items() if rt == "CALL"]
    put_ivs = [(K, iv) for (K, rt), iv in iv_map.items() if rt == "PUT"]

    result_25d_call = find_25delta_strike(call_ivs, spot, T, r, "CALL")
    result_25d_put = find_25delta_strike(put_ivs, spot, T, r, "PUT")

    if result_25d_call is None or result_25d_put is None:
        return {}

    iv_25d_call = result_25d_call[1]
    iv_25d_put = result_25d_put[1]

    rr = iv_25d_put - iv_25d_call
    bf = (iv_25d_put + iv_25d_call) / 2.0 - atm_iv

    return {
        "atm_iv": atm_iv,
        "iv_25d_call": iv_25d_call,
        "iv_25d_put": iv_25d_put,
        "rr": rr,
        "bf": bf,
    }


# ---------------------------------------------------------------------------
# Main computation: load all CSVs for a ticker, compute per-date signals
# ---------------------------------------------------------------------------

def compute_skew_signals(
    ticker: str,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Load all skew cache files for `ticker` and compute per-date signals.

    Returns DataFrame with columns:
        date, rr_1w, bf_1w, atm_iv_1w, atm_iv_1m, ts_slope
    """
    cache_dir = Path(cache_dir)
    pattern = re.compile(
        rf"^{ticker}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})_quote_1h\.csv$"
    )

    # Index files by (trade_date, expiry)
    file_index: dict[tuple[str, str], Path] = {}
    for fname in cache_dir.iterdir():
        m = pattern.match(fname.name)
        if m:
            file_index[(m.group(1), m.group(2))] = fname

    if not file_index:
        raise FileNotFoundError(
            f"No files found for ticker {ticker} in {cache_dir}"
        )

    # Group by trade_date → list of (expiry, path)
    by_date: dict[str, list[tuple[str, Path]]] = {}
    for (td, exp), path in file_index.items():
        by_date.setdefault(td, []).append((exp, path))

    trade_dates = sorted(by_date.keys())
    log.info(
        f"{ticker}: {len(trade_dates)} trade dates, "
        f"{len(file_index)} files  ({trade_dates[0]} → {trade_dates[-1]})"
    )

    # Fetch spot prices via yfinance for the full range
    log.info(f"{ticker}: fetching spot prices via yfinance …")
    yf_ticker = yf.Ticker(ticker)
    spot_df = yf_ticker.history(
        start=trade_dates[0],
        end=(pd.to_datetime(trade_dates[-1]) + timedelta(days=2)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )
    spot_df.index = pd.to_datetime(spot_df.index).tz_localize(None).normalize()
    # Use open price as morning spot approximation
    spot_series = spot_df["Open"]

    rows = []

    for td_str in trade_dates:
        trade_date = pd.to_datetime(td_str)
        exps = by_date[td_str]

        # Categorise expirations
        w1_candidates, m1_candidates = [], []
        for exp_str, path in exps:
            exp_date = pd.to_datetime(exp_str)
            dte = (exp_date - trade_date).days
            if MIN_DTE_1W <= dte <= MAX_DTE_1W:
                w1_candidates.append((dte, exp_str, path))
            elif MIN_DTE_1M <= dte <= MAX_DTE_1M:
                m1_candidates.append((dte, exp_str, path))

        if not w1_candidates or not m1_candidates:
            continue  # need both expiry buckets

        # Pick the shortest-DTE within each bucket
        w1_candidates.sort()
        m1_candidates.sort()
        _, exp_1w_str, path_1w = w1_candidates[0]
        _, exp_1m_str, path_1m = m1_candidates[0]

        # Spot price for this date
        if trade_date not in spot_series.index:
            # Try nearest prior business day
            prior = spot_series.index[spot_series.index <= trade_date]
            if len(prior) == 0:
                continue
            spot = float(spot_series.loc[prior[-1]])
        else:
            spot = float(spot_series.loc[trade_date])

        if spot <= 0 or np.isnan(spot):
            continue

        # Load and process each expiry
        signals: dict = {}
        for label, path, exp_str in [
            ("1w", path_1w, exp_1w_str),
            ("1m", path_1m, exp_1m_str),
        ]:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                log.warning(f"  {td_str} {label}: read error: {e}")
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # First non-zero snapshot (skip 09:30 which is often all zeros)
            valid = df[(df["bid"] > 0) & (df["ask"] > 0)]
            if valid.empty:
                continue
            first_ts = valid["timestamp"].min()
            snap = valid[valid["timestamp"] == first_ts].copy()

            exp_date = pd.to_datetime(exp_str)
            surf = compute_surface_signals(snap, spot, trade_date, exp_date)
            if surf:
                signals[label] = surf

        if "1w" not in signals or "1m" not in signals:
            continue

        atm_iv_1w = signals["1w"]["atm_iv"]
        atm_iv_1m = signals["1m"]["atm_iv"]

        if atm_iv_1m <= 0:
            continue

        row = {
            "date": trade_date,
            "rr_1w": signals["1w"]["rr"],
            "bf_1w": signals["1w"]["bf"],
            "atm_iv_1w": atm_iv_1w,
            "atm_iv_1m": atm_iv_1m,
            "ts_slope": atm_iv_1w / atm_iv_1m,
            # extra detail columns
            "iv_25d_put_1w": signals["1w"]["iv_25d_put"],
            "iv_25d_call_1w": signals["1w"]["iv_25d_call"],
            "spot": spot,
        }
        rows.append(row)

        if len(rows) % 50 == 0:
            log.info(f"  processed {len(rows)} dates …")

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info(
        f"{ticker}: computed {len(result)} rows  "
        f"({result['date'].min().date()} → {result['date'].max().date()})"
    )
    return result


# ---------------------------------------------------------------------------
# corr(spot, IV) — 21-day rolling correlation, no skew cache needed
# ---------------------------------------------------------------------------

def compute_corr_spot_iv(
    ticker: str = "SPY",
    start: str = "2022-01-01",
    end: Optional[str] = None,
    window: int = 21,
) -> pd.Series:
    """
    Compute 21-day rolling correlation between:
      - daily log return of `ticker`
      - daily change in VIX

    Returns a pd.Series indexed by date.
    """
    if end is None:
        end = (pd.Timestamp.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    log.info(f"Fetching {ticker} and VIX for corr computation …")
    spy = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)

    spy_ret = spy["Close"].squeeze().pct_change()
    vix_chg = vix["Close"].squeeze().diff()

    combined = pd.DataFrame({"spy_ret": spy_ret, "vix_chg": vix_chg}).dropna()
    corr = combined["spy_ret"].rolling(window).corr(combined["vix_chg"])
    corr.name = "corr_spot_iv_21d"
    corr.index = pd.to_datetime(corr.index).tz_localize(None).normalize()
    return corr


# ---------------------------------------------------------------------------
# integrate_skew_signals — merge with existing gate-signals dict
# ---------------------------------------------------------------------------

def integrate_skew_signals(
    signals: dict,
    ticker: str = "SPY",
    cache_dir: Path = CACHE_DIR,
    trade_date: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Load pre-computed skew signals parquet and enrich a signals dict.

    Parameters
    ----------
    signals   : existing dict from compute_gate_signals (or similar)
    ticker    : 'SPY' or 'QQQ'
    cache_dir : directory where skew_signals_{ticker}.parquet lives
    trade_date: the date to look up; defaults to today

    Returns
    -------
    enriched signals dict with additional keys:
        rr_1w, bf_1w, ts_slope, atm_iv_1w, atm_iv_1m, corr_spot_iv_21d
    """
    if trade_date is None:
        trade_date = pd.Timestamp.today().normalize()

    out = dict(signals)  # copy, don't mutate

    # --- skew surface signals ---
    parquet_path = Path(cache_dir) / f"skew_signals_{ticker}.parquet"
    if parquet_path.exists():
        sk = pd.read_parquet(parquet_path)
        sk["date"] = pd.to_datetime(sk["date"]).dt.normalize()
        row = sk[sk["date"] == trade_date]
        if row.empty:
            # fall back to most recent available date
            past = sk[sk["date"] <= trade_date]
            row = past.tail(1) if not past.empty else sk.tail(1)
            if not row.empty:
                log.warning(
                    f"integrate_skew_signals: no exact match for {trade_date.date()}, "
                    f"using {row.iloc[0]['date'].date()}"
                )

        if not row.empty:
            r = row.iloc[0]
            out["rr_1w"] = float(r["rr_1w"])
            out["bf_1w"] = float(r["bf_1w"])
            out["ts_slope"] = float(r["ts_slope"])
            out["atm_iv_1w"] = float(r["atm_iv_1w"])
            out["atm_iv_1m"] = float(r["atm_iv_1m"])
        else:
            log.warning(f"integrate_skew_signals: no skew data found for {ticker}")
            out.update(rr_1w=None, bf_1w=None, ts_slope=None, atm_iv_1w=None, atm_iv_1m=None)
    else:
        log.warning(
            f"integrate_skew_signals: parquet not found at {parquet_path}. "
            "Run compute_skew_signals() first."
        )
        out.update(rr_1w=None, bf_1w=None, ts_slope=None, atm_iv_1w=None, atm_iv_1m=None)

    # --- rolling corr(spot, IV) from yfinance (always fresh) ---
    try:
        start = (trade_date - timedelta(days=60)).strftime("%Y-%m-%d")
        end = (trade_date + timedelta(days=2)).strftime("%Y-%m-%d")
        corr_series = compute_corr_spot_iv(ticker=ticker, start=start, end=end)
        # Most recent value on or before trade_date
        valid_corr = corr_series.dropna()
        past_corr = valid_corr[valid_corr.index <= trade_date]
        if not past_corr.empty:
            out["corr_spot_iv_21d"] = float(past_corr.iloc[-1])
        else:
            out["corr_spot_iv_21d"] = None
    except Exception as e:
        log.warning(f"integrate_skew_signals: corr computation failed: {e}")
        out["corr_spot_iv_21d"] = None

    return out


# ---------------------------------------------------------------------------
# adjust_wings_for_skew — dynamic wing sizing
# ---------------------------------------------------------------------------

def adjust_wings_for_skew(
    cfg: dict,
    sig: dict,
    bf_1w_threshold: float = None,
    rr_1w_threshold: float = None,
    ts_inversion_threshold: float = 1.0,
    corr_unusual_threshold: float = -0.3,
) -> dict:
    """
    Adjust iron condor wing widths and position size based on skew signals.

    Parameters
    ----------
    cfg : strategy config dict containing:
          - n_wing_sigma        : list/array of wing widths to try (both sides)
          - n_wing_sigma_calm   : tighter grid used on calm days
          - (optional) bf_1w_p75  : 75th percentile BF (overrides bf_1w_threshold)
          - (optional) rr_1w_p75  : 75th percentile RR (overrides rr_1w_threshold)

    sig : signals dict (output of integrate_skew_signals); expects:
          - bf_1w, rr_1w, ts_slope, corr_spot_iv_21d

    bf_1w_threshold : float; if None, use cfg["bf_1w_p75"] or default 0.02
    rr_1w_threshold : float; if None, use cfg["rr_1w_p75"] or default 0.03

    Returns
    -------
    dict with keys:
        n_wing_sigma         : adjusted wing grid (both sides)
        n_wing_sigma_put     : adjusted put-side wing grid (wider when RR high)
        n_wing_sigma_call    : adjusted call-side wing grid
        n_wing_sigma_calm    : adjusted calm-day grid
        position_size_mult   : 1.0 or 0.5 (regime health multiplier)
        skip_day             : bool — True if TS inversion is severe enough to skip
        adjustments          : list of human-readable strings explaining changes
    """
    # --- resolve base grids ---
    base_wings = list(cfg.get("n_wing_sigma", [1.5, 1.75, 2.0, 2.25, 2.5]))
    base_calm = list(cfg.get("n_wing_sigma_calm", [1.25, 1.5, 1.75, 2.0]))

    # --- resolve thresholds ---
    if bf_1w_threshold is None:
        bf_1w_threshold = float(cfg.get("bf_1w_p75", 0.02))
    if rr_1w_threshold is None:
        rr_1w_threshold = float(cfg.get("rr_1w_p75", 0.03))

    # --- read signals (handle None gracefully) ---
    bf_1w = sig.get("bf_1w")
    rr_1w = sig.get("rr_1w")
    ts_slope = sig.get("ts_slope")
    corr = sig.get("corr_spot_iv_21d")

    # Working copies — put/call side adjustments tracked separately
    put_add = 0.0     # extra sigma added to put wing
    call_add = 0.0    # extra sigma added to call wing
    both_add = 0.0    # extra sigma added to both wings
    position_size_mult = 1.0
    skip_day = False
    adjustments = []

    # ----------------------------------------------------------------
    # Rule 1: High butterfly → fat tails priced → widen BOTH wings
    #   BF > 75th pct: +0.25σ both wings
    # ----------------------------------------------------------------
    if bf_1w is not None and bf_1w > bf_1w_threshold:
        both_add += 0.25
        adjustments.append(
            f"bf_1w={bf_1w:.4f} > threshold={bf_1w_threshold:.4f}: +0.25σ both wings"
        )

    # ----------------------------------------------------------------
    # Rule 2: High risk reversal → put skew elevated → widen PUT wing
    #   RR > 75th pct: +0.25σ put wing specifically
    # ----------------------------------------------------------------
    if rr_1w is not None and rr_1w > rr_1w_threshold:
        put_add += 0.25
        adjustments.append(
            f"rr_1w={rr_1w:.4f} > threshold={rr_1w_threshold:.4f}: +0.25σ put wing"
        )

    # ----------------------------------------------------------------
    # Rule 3: Term structure inverted → near-term stress
    #   TS > 1.0: +0.5σ both wings (or skip if TS > 1.1)
    #   TS < 0.85: calm contango → no change (normal/good regime)
    # ----------------------------------------------------------------
    if ts_slope is not None:
        if ts_slope > 1.10:
            skip_day = True
            adjustments.append(
                f"ts_slope={ts_slope:.3f} > 1.10 (severe inversion): SKIP DAY"
            )
        elif ts_slope > ts_inversion_threshold:
            both_add += 0.50
            adjustments.append(
                f"ts_slope={ts_slope:.3f} > {ts_inversion_threshold}: +0.5σ both wings (mild inversion)"
            )
        elif ts_slope < 0.85:
            adjustments.append(
                f"ts_slope={ts_slope:.3f} < 0.85: steep contango — favourable IC environment"
            )

    # ----------------------------------------------------------------
    # Rule 4: Unusual corr(spot, IV) regime → reduce size
    #   Normally strong negative (-0.6 to -0.8)
    #   Near zero or positive → unusual → 50% size
    # ----------------------------------------------------------------
    if corr is not None and corr > corr_unusual_threshold:
        position_size_mult = 0.5
        adjustments.append(
            f"corr_spot_iv_21d={corr:.3f} > {corr_unusual_threshold}: "
            "unusual regime → 50% position size"
        )

    # ----------------------------------------------------------------
    # Apply adjustments
    # ----------------------------------------------------------------
    def _add_to_grid(grid: list, add: float) -> list:
        """Round to nearest 0.25σ step after adding."""
        return [round(w + add, 4) for w in grid]

    def _round25(v: float) -> float:
        return round(round(v / 0.25) * 0.25, 4)

    total_put = both_add + put_add
    total_call = both_add + call_add

    # Symmetric grid: take the larger of put/call additions for both sides
    # (conservative — if we need a wider put, the overall wing grid gets
    # the put adjustment so we don't accidentally sell too narrow on either side)
    symmetric_add = max(total_put, total_call)

    n_wing_sigma_adjusted = _add_to_grid(base_wings, symmetric_add)
    n_wing_sigma_calm_adjusted = _add_to_grid(base_calm, symmetric_add)
    n_wing_sigma_put = _add_to_grid(base_wings, total_put)
    n_wing_sigma_call = _add_to_grid(base_wings, total_call)

    return {
        "n_wing_sigma": n_wing_sigma_adjusted,
        "n_wing_sigma_put": n_wing_sigma_put,
        "n_wing_sigma_call": n_wing_sigma_call,
        "n_wing_sigma_calm": n_wing_sigma_calm_adjusted,
        "position_size_mult": position_size_mult,
        "skip_day": skip_day,
        "adjustments": adjustments,
        # pass-through raw signals for downstream logging
        "bf_1w": bf_1w,
        "rr_1w": rr_1w,
        "ts_slope": ts_slope,
        "corr_spot_iv_21d": corr,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute skew signals from option quote cache.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ticker", choices=["SPY", "QQQ"], help="Single ticker to process")
    grp.add_argument("--all", action="store_true", help="Process both SPY and QQQ")
    parser.add_argument(
        "--cache_dir",
        default=str(CACHE_DIR),
        help=f"Path to skew cache directory (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for parquet files (default: same as cache_dir)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cache_dir

    tickers = ["SPY", "QQQ"] if args.all else [args.ticker]

    for ticker in tickers:
        log.info(f"=== Processing {ticker} ===")
        df = compute_skew_signals(ticker, cache_dir=cache_dir)
        out_path = out_dir / f"skew_signals_{ticker}.parquet"
        df.to_parquet(out_path, index=False)
        log.info(f"  saved → {out_path}  ({len(df)} rows)")
        print(df.tail(5).to_string(index=False))
        print()

    log.info("Done.")


if __name__ == "__main__":
    main()
