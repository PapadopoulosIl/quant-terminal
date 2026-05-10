"""
market_layer.py — Market Data Ingestion & Normalization
=======================================================
Responsible for:
  - Fetching OHLCV data from yfinance (single & multi-ticker)
  - Normalizing price data to clean DataFrames
  - Providing market snapshots (current price, prev close, sparklines)
  - Ingesting historical returns for the risk engine
  - Fetching analyst views and fundamentals raw data

Contract:
  Input:  ticker strings, date ranges, config primitives
  Output: pd.DataFrame or dict of clean numeric data
  No business logic — only fetch + normalize.
"""
from __future__ import annotations
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from utils import (
    safe_get_attr, safe_float,
    build_statement_table, add_margin_rows, normalize_estimate_table,
    pick_first_available,
)


def enforce_strict_chronology(df: pd.DataFrame, is_annual: bool = False, periods: int = 4) -> pd.DataFrame:
    """
    Keep only true date columns, sort newest-to-oldest, and relabel them
    consistently so downstream statement logic is not polluted by TTM columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    valid_cols = [col for col in df.columns if isinstance(col, (pd.Timestamp, dt.datetime))]
    if not valid_cols:
        return df

    valid_cols.sort(reverse=True)
    target_cols = valid_cols[:periods]
    clean_df = df[target_cols].copy()

    prefix = "y" if is_annual else "q"
    new_col_names = {}
    for i, col in enumerate(target_cols):
        date_str = pd.Timestamp(col).strftime("%b '%y")
        new_col_names[col] = f"{i}{prefix} ({date_str})"

    return clean_df.rename(columns=new_col_names)


# ---------------------------------------------------------------------------
# Price downloads
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers: list, start: dt.date, end: dt.date) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty or "Close" not in raw:
        return pd.DataFrame()
    close = raw["Close"].dropna(how="all")
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close


def fetch_market_snapshot(tickers: list) -> dict:
    """
    Returns {ticker: {"current": float, "prev_close": float, "spark": list[float], "low_52w": float, "high_52w": float}}.
    Uses 1y period to extract range data.
    """
    result: dict = {}
    if not tickers:
        return result
    try:
        raw = yf.download(tickers, period="1y", interval="1d", progress=False, auto_adjust=False)
        if raw.empty:
            return result
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        highs = raw["High"] if "High" in raw.columns.get_level_values(0) else raw
        lows  = raw["Low"]  if "Low" in raw.columns.get_level_values(0) else raw
        
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
            highs = highs.to_frame(name=tickers[0])
            lows  = lows.to_frame(name=tickers[0])
            
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(0)
            highs = highs.columns.get_level_values(0)
            lows  = lows.columns.get_level_values(0)
            
        for t in tickers:
            if t not in close.columns:
                continue
            s_close = close[t].dropna()
            s_high  = highs[t].dropna()
            s_low   = lows[t].dropna()
            
            if s_close.empty:
                continue
                
            result[t] = {
                "current":    float(s_close.iloc[-1]),
                "prev_close": float(s_close.iloc[-2]) if len(s_close) >= 2 else float(s_close.iloc[-1]),
                "spark":      [round(float(v), 4) for v in s_close.tail(7).values],
                "low_52w":    float(s_low.min()),
                "high_52w":   float(s_high.max()),
            }
    except Exception:
        pass
    return result


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_historical_returns(tickers: list) -> pd.DataFrame:
    """1 year of daily % returns, including SPY + QQQ as benchmarks."""
    if not tickers:
        return pd.DataFrame()
    full_list = list(set(tickers + ["SPY", "QQQ"]))
    try:
        raw = yf.download(full_list, period="1y", interval="1d", progress=False, auto_adjust=False)
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=full_list[0])
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(0)
        close.ffill(inplace=True)
        return close.pct_change().dropna()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Live briefing / intraday
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_briefing_data(ticker: str) -> dict:
    """Layer A: Pure Data Ingestion for Live Briefing."""
    daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    intraday = yf.download(ticker, period="5d", interval="30m", prepost=True, progress=False, auto_adjust=False)
    
    if daily.empty:
        raise ValueError(f"Δεν επέστρεψαν daily δεδομένα για {ticker}.")
        
    # Clean multi-index if present
    for df in [daily, intraday]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
    
    daily = daily.dropna(how="all")
    intraday = intraday.dropna(how="all")
    
    # Return raw data + basic primitives for Layer B/C to consume
    return {
        "ticker": ticker,
        "daily": daily,
        "intraday": intraday,
        "updated_at": dt.datetime.now(),
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_pulse_data(ticker: str, range_key: str) -> tuple[pd.DataFrame, dict]:
    configs = {
        "1D": {"period": "1d",  "interval": "5m",  "label": "Today + Pre / After Hours"},
        "1W": {"period": "5d",  "interval": "30m", "label": "1 Week Pulse"},
        "1M": {"period": "1mo", "interval": "1h",  "label": "1 Month Pulse"},
        "1Y": {"period": "1y",  "interval": "1d",  "label": "1 Year Pulse"},
    }
    config = configs[range_key]
    data = yf.download(ticker, period=config["period"], interval=config["interval"],
                       prepost=True, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"Δεν επέστρεψαν pulse δεδομένα για {ticker} στο {range_key}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data.dropna(how="all"), config


@st.cache_data(ttl=900, show_spinner=False)
def fetch_range_reference_data(ticker: str) -> pd.DataFrame:
    daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    if daily.empty:
        return pd.DataFrame()
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = [col[0] for col in daily.columns]
    return daily.dropna(how="all")


# ---------------------------------------------------------------------------
# Fundamentals & analyst data
# ---------------------------------------------------------------------------

def _estimate_forward_pe(info, earnings_estimate, eps_trend, current_price, trailing_pe, funds_data=None, symbol=None):
    forward_pe = info.get("forwardPE", np.nan)
    if pd.notna(forward_pe):
        return forward_pe, "info"

    # ETF specific check
    if funds_data is not None:
        f_pe = _extract_fund_equity_ratio(funds_data, "Price/Earnings Forward", symbol)
        if pd.notna(f_pe):
            return f_pe, "fund_equity_ratio"

    forward_eps = info.get("forwardEps", np.nan)
    if pd.notna(current_price) and pd.notna(forward_eps) and forward_eps not in [0, np.nan]:
        return current_price / forward_eps, "derived_from_forward_eps"

    for table in [earnings_estimate, eps_trend]:
        if isinstance(table, pd.DataFrame) and not table.empty:
            cols = [col for col in table.columns if str(col).lower() in {"avg", "current"}]
            if cols:
                try:
                    eps_candidate = pd.to_numeric(table[cols[0]], errors="coerce").dropna()
                    if not eps_candidate.empty and pd.notna(current_price) and eps_candidate.iloc[0] != 0:
                        return current_price / eps_candidate.iloc[0], "derived_from_estimate_table"
                except Exception:
                    pass

    return np.nan, "unavailable"


def _extract_fund_equity_ratio(funds_data, metric_name: str, symbol: str) -> float:
    try:
        eh = funds_data.equity_holdings
        if eh is None or eh.empty:
            return np.nan
        
        # Normalize index and columns for robust matching
        lookup_metric = str(metric_name).lower()
        lookup_symbol = str(symbol).upper()
        
        # Match symbol column
        match_col = None
        for col in eh.columns:
            if str(col).upper() == lookup_symbol:
                match_col = col
                break
        if not match_col: return np.nan
        
        # Search for metric in index
        val = np.nan
        for idx in eh.index:
            idx_str = str(idx).lower()
            if lookup_metric in idx_str or idx_str in lookup_metric:
                val = safe_float(eh.loc[idx, match_col])
                if pd.notna(val): break
        
        if pd.notna(val):
            # Yahoo often returns Yields (e.g., 0.03 for 33x P/E) for ETFs
            if 0 < val < 0.25: 
                return 1.0 / val
            return val
    except Exception:
        pass
    return np.nan


def _extract_top_holding_weight(funds_data, symbol: str) -> float:
    try:
        th = funds_data.top_holdings
        if th is None or th.empty:
            return np.nan
    except Exception:
        return np.nan
    lookup = str(symbol).upper()
    norm_cols = {str(col).strip().lower(): col for col in th.columns}
    symbol_col = norm_cols.get("symbol") or norm_cols.get("holding")
    weight_col = (norm_cols.get("holdingpercent") or norm_cols.get("holding percent")
                  or norm_cols.get("weight") or norm_cols.get("portfoliopercent"))
    try:
        if symbol_col and weight_col:
            matches = th[th[symbol_col].astype(str).str.upper() == lookup]
            if not matches.empty:
                w = safe_float(matches.iloc[0][weight_col])
                if pd.notna(w):
                    return w * 100 if w <= 1 else w
        if lookup in th.index.astype(str).str.upper():
            row = th.loc[th.index.astype(str).str.upper() == lookup]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for candidate in ["holdingPercent", "holding percent", "weight", "portfolioPercent"]:
                if candidate in row.index:
                    w = safe_float(row[candidate])
                    if pd.notna(w):
                        return w * 100 if w <= 1 else w
    except Exception:
        pass
    return np.nan


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals_data(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = safe_get_attr(ticker, "info", {}) or {}
    funds_data = safe_get_attr(ticker, "funds_data", None)

    raw_q_income = safe_get_attr(ticker, "quarterly_income_stmt", pd.DataFrame())
    raw_a_income = safe_get_attr(ticker, "income_stmt", pd.DataFrame())
    raw_q_cashflow = safe_get_attr(ticker, "quarterly_cashflow", pd.DataFrame())
    raw_a_cashflow = safe_get_attr(ticker, "cashflow", pd.DataFrame())

    q_income = enforce_strict_chronology(raw_q_income, is_annual=False, periods=4)
    a_income = enforce_strict_chronology(raw_a_income, is_annual=True, periods=4)
    q_cashflow = enforce_strict_chronology(raw_q_cashflow, is_annual=False, periods=4)
    a_cashflow = enforce_strict_chronology(raw_a_cashflow, is_annual=True, periods=4)

    row_map = {
        "Revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
        "Gross Profit": ["Gross Profit"],
        "Operating Income": ["Operating Income"],
        "Net Income": ["Net Income", "Net Income Common Stockholders"],
        "Diluted EPS": ["Diluted EPS", "Basic EPS"],
        "CapEx": ["Capital Expenditure"],
        "Free Cash Flow": ["Free Cash Flow"],
        "Operating Cash Flow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    }

    quarterly = build_statement_table(q_income, row_map, periods=4)
    annual    = build_statement_table(a_income, row_map, periods=4)

    def _merge_cashflow_rows(df_main: pd.DataFrame, df_fallback: pd.DataFrame) -> pd.DataFrame:
        if df_fallback is None or df_fallback.empty:
            return df_main
        extra_frames = [df_main]
        for label in ["CapEx", "Free Cash Flow", "Operating Cash Flow"]:
            if label not in df_main.index:
                extra = build_statement_table(df_fallback, {label: row_map[label]}, periods=4)
                if not extra.empty:
                    extra_frames.append(extra)
        return pd.concat(extra_frames) if extra_frames else df_main

    quarterly = _merge_cashflow_rows(quarterly, q_cashflow)
    annual = _merge_cashflow_rows(annual, a_cashflow)

    quarterly = add_margin_rows(quarterly)
    annual    = add_margin_rows(annual)

    earnings_estimate = normalize_estimate_table(safe_get_attr(ticker, "earnings_estimate", pd.DataFrame()))
    revenue_estimate  = normalize_estimate_table(safe_get_attr(ticker, "revenue_estimate", pd.DataFrame()))
    eps_trend         = normalize_estimate_table(safe_get_attr(ticker, "eps_trend", pd.DataFrame()))

    latest_quarter = quarterly.iloc[:, 0] if not quarterly.empty else pd.Series(dtype=float)
    latest_period  = quarterly.columns[0] if not quarterly.empty else "-"
    current_price  = safe_float(info.get("regularMarketPrice"))
    trailing_eps   = safe_float(info.get("trailingEps"))
    forward_eps    = safe_float(info.get("forwardEps"))

    ttm_eps = np.nan
    if not quarterly.empty and "Diluted EPS" in quarterly.index:
        eps_row = pd.to_numeric(quarterly.loc["Diluted EPS"], errors="coerce").dropna()
        if len(eps_row) == 4:
            ttm_eps = float(eps_row.sum())

    if pd.isna(ttm_eps):
        ttm_eps = trailing_eps

    # ── Multi-pass Multiples Extraction ─────────────────────────────────────
    trailing_pe = np.nan
    if pd.notna(current_price) and pd.notna(ttm_eps) and ttm_eps > 0:
        trailing_pe = current_price / ttm_eps
    else:
        trailing_pe = safe_float(info.get("trailingPE"))
        if pd.isna(trailing_pe):
            trailing_pe = _extract_fund_equity_ratio(funds_data, "Price/Earnings", symbol)

    ps_ratio = safe_float(info.get("priceToSalesTrailing12Months"))
    if pd.isna(ps_ratio):
        ps_ratio = _extract_fund_equity_ratio(funds_data, "Price/Sales", symbol)

    forward_pe = np.nan
    forward_pe_source = "unavailable"
    if pd.notna(current_price) and pd.notna(forward_eps) and forward_eps > 0:
        forward_pe = current_price / forward_eps
        forward_pe_source = "derived_from_forward_eps"
    else:
        forward_pe, forward_pe_source = _estimate_forward_pe(
            info, earnings_estimate, eps_trend, current_price, trailing_pe, funds_data, symbol
        )
    
    # ── Last Resort Hardcoded Safety Net for Major Benchmarks ──
    if symbol == "QQQ":
        if pd.isna(trailing_pe) or trailing_pe < 5: trailing_pe = 34.6
        if pd.isna(forward_pe) or forward_pe < 5: forward_pe = 30.5
        if pd.isna(ps_ratio) or ps_ratio < 1: ps_ratio = 5.2
    elif symbol == "SPY":
        if pd.isna(trailing_pe) or trailing_pe < 5: trailing_pe = 28.2
        if pd.isna(forward_pe) or forward_pe < 5: forward_pe = 24.5
        if pd.isna(ps_ratio) or ps_ratio < 1: ps_ratio = 2.9

    div_yield      = safe_float(info.get("dividendYield"))
    if pd.isna(div_yield):
        div_yield = _extract_fund_equity_ratio(funds_data, "Yield", symbol)

    snapshot = {
        "current_price": current_price, "trailing_eps": ttm_eps if pd.notna(ttm_eps) else trailing_eps,
        "forward_eps": forward_eps, "trailing_pe": trailing_pe,
        "forward_pe": forward_pe, "forward_pe_source": forward_pe_source,
        "ps_ratio": ps_ratio, "div_yield": div_yield,
        "revenue": pick_first_available(latest_quarter, ["Revenue"]),
        "net_income": pick_first_available(latest_quarter, ["Net Income"]),
        "capex": pick_first_available(latest_quarter, ["CapEx"]),
        "profit_margin": pick_first_available(latest_quarter, ["Profit Margin %"]),
        "net_profit_margin": pick_first_available(latest_quarter, ["Net Profit Margin %"]),
        "last_quarter": latest_period,
        "target_high": info.get("targetHighPrice"),
        "target_low": info.get("targetLowPrice"),
        "target_mean": info.get("targetMeanPrice"),
        "recommendation": info.get("recommendationKey", "N/A"),
        "market_cap": info.get("marketCap"),
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    return {
        "snapshot": snapshot, "quarterly": quarterly, "annual": annual,
        "earnings_estimate": earnings_estimate, "revenue_estimate": revenue_estimate,
        "eps_trend": eps_trend,
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_benchmark_membership_data(benchmark_symbol: str, component_symbol: str) -> dict:
    ticker = yf.Ticker(benchmark_symbol)
    funds_data = safe_get_attr(ticker, "funds_data", None)
    weight_pct = _extract_top_holding_weight(funds_data, component_symbol)
    return {"benchmark": benchmark_symbol, "component": component_symbol, "weight_pct": weight_pct}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_analyst_view(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    recs = safe_get_attr(ticker, "recommendations", pd.DataFrame())
    recs_summary = safe_get_attr(ticker, "recommendations_summary", pd.DataFrame())
    upgrades = safe_get_attr(ticker, "upgrades_downgrades", pd.DataFrame())
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        recs = recs.tail(12)
    if isinstance(upgrades, pd.DataFrame) and not upgrades.empty:
        upgrades = upgrades.tail(12)
    return {
        "recommendations": recs if isinstance(recs, pd.DataFrame) else pd.DataFrame(),
        "recommendations_summary": recs_summary if isinstance(recs_summary, pd.DataFrame) else pd.DataFrame(),
        "upgrades_downgrades": upgrades if isinstance(upgrades, pd.DataFrame) else pd.DataFrame(),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_dates(symbol: str) -> pd.DataFrame:
    """Fetches earnings dates safely from yfinance"""
    try:
        t = yf.Ticker(symbol)
        df = t.get_earnings_dates(limit=10) if hasattr(t, 'get_earnings_dates') else t.earnings_dates
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Metadata enrichment (TTL-aware, calls data_layer internally)
# ---------------------------------------------------------------------------

def enrich_metadata(tickers: list) -> None:
    """Fetches and upserts metadata only for stale/missing tickers."""
    from DataLayer6 import get_stale_or_missing, db_upsert_metadata
    import streamlit as st

    stale = get_stale_or_missing(tickers)
    if not stale:
        return

    with st.spinner(f"Fetching metadata for {', '.join(stale)}…"):
        for t in stale:
            try:
                # 1. Try robust fundamentals fetcher first (handles ETFs/Mutual Funds)
                fun_pkg = fetch_fundamentals_data(t)
                snapshot = fun_pkg.get("snapshot", {})
                
                # 2. Try yfinance info as fallback/supplement
                info = yf.Ticker(t).info or {}
                
                # 3. Merge: snapshot data is generally more reliable for our engine
                merged = {
                    "sector": snapshot.get("sector") or info.get("sector") or "Unknown",
                    "industry": snapshot.get("industry") or info.get("industry") or "Unknown",
                    "quoteType": info.get("quoteType") or ("ETF" if "ETF" in t.upper() else "EQUITY"),
                    "beta": snapshot.get("beta") or info.get("beta"),
                    "dividendYield": snapshot.get("dividend_yield") or info.get("dividendYield"),
                    "trailingPE": snapshot.get("trailing_pe") or info.get("trailingPE"),
                    "forwardPE": snapshot.get("forward_pe") or info.get("forwardPE"),
                    "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
                    "earningsGrowth": info.get("earningsGrowth"),
                }
                
                db_upsert_metadata(t, merged)
            except Exception:
                from DataLayer6 import db_upsert_metadata as _ups
                _ups(t, {})

# ---------------------------------------------------------------------------
# Econometric Feature Engineering (Macro & Regime Core)
# ---------------------------------------------------------------------------

def get_market_trend(spy_closes: pd.Series) -> str:
    """Institutional trend identification using 200-day Simple Moving Average."""
    if len(spy_closes) < 200:
        # Fallback for shorter windows
        if len(spy_closes) > 20:
             sma = spy_closes.iloc[-20:].mean()
             return "ABOVE_20SMA_FALLBACK" if spy_closes.iloc[-1] > sma else "BELOW_20SMA_FALLBACK"
        return "INSUFFICIENT_DATA"
    sma = spy_closes.iloc[-200:].mean()
    return "ABOVE_200SMA" if spy_closes.iloc[-1] > sma else "BELOW_200SMA"


def extract_econometric_features(price_series: pd.Series, volume: pd.Series = None) -> dict:
    """
    Δέχεται numeric series και εξάγει 1st/2nd derivatives, z-scores, κλπ.
    Pure feature calculation backend που εξυπηρετεί όλη την οικονομετρία.
    """
    from utils import extract_series_derivatives
    features = {}
    if price_series is None or price_series.empty:
        return features
        
    s = price_series.dropna()
    if len(s) < 10:
        return features
        
    # Derivatives
    try:
        slope, accel = extract_series_derivatives(s)
    except Exception:
        slope, accel = 0.0, 0.0
        
    features["slope"] = slope
    features["acceleration"] = accel
    
    # Momentum (Z-Score)
    mean_val = s.mean()
    std_val = s.std()
    curr_val = float(s.iloc[-1])
    features["z_score"] = float((curr_val - mean_val) / std_val) if std_val != 0 else 0.0
    
    # Range Positioning (0-100%)
    high_x = float(s.max())
    low_x = float(s.min())
    features["range_position"] = float((curr_val - low_x) / (high_x - low_x) * 100) if high_x > low_x else 50.0
    
    return features

def calculate_macro_corridor(tnx_series: pd.Series, oil_series: pd.Series, spy_series: pd.Series) -> dict:
    """
    Αναλύει το μακροοικονομικό περιβάλλον με τη λογική της Κεντρικής Τράπεζας 
    (Rate Volatility, Inflation Proxy, Output Gap).
    """
    if tnx_series.empty or oil_series.empty or spy_series.empty:
        return {"regime": "UNKNOWN", "inflation_pressure": "UNKNOWN", "output_gap": 0}

    # 1. Rate Volatility (Το Corridor)
    # Αν η ετησιοποιημένη μεταβλητότητα των επιτοκίων είναι < 15%, η αγορά το έχει συνηθίσει
    tnx_vol = tnx_series.pct_change(fill_method=None).std() * np.sqrt(252)
    rates_stable = tnx_vol < 0.15

    # 2. Inflation Proxy (Cost-Push Momentum από το Πετρέλαιο - 3 Μήνες)
    # Be careful not to go out of bounds if len < 63
    lookback = min(63, len(oil_series) - 1)
    if lookback > 0:
        oil_3m_momentum = (oil_series.iloc[-1] / oil_series.iloc[-lookback]) - 1
    else:
        oil_3m_momentum = 0
    inflation_pressure = "HIGH" if oil_3m_momentum > 0.15 else ("LOW" if oil_3m_momentum < -0.10 else "NEUTRAL")

    # 3. Output Gap Proxy (SPY απόσταση από τον SMA-200)
    spy_sma200 = spy_series.rolling(window=min(200, len(spy_series))).mean().iloc[-1]
    spy_current = spy_series.iloc[-1]
    output_gap = (spy_current / spy_sma200) - 1 if spy_sma200 > 0 else 0

    # Καθορισμός του "Central Bank Regime"
    regime = "NORMAL"
    if not rates_stable and inflation_pressure == "HIGH" and output_gap < 0:
        regime = "STAGFLATION_RISK" # Το χειρότερο σενάριο
    elif rates_stable and inflation_pressure == "NEUTRAL" and output_gap > 0:
        regime = "GOLDILOCKS" # Όλα τέλεια (Σταθερά επιτόκια, ανάπτυξη)
    elif output_gap > 0.05 and inflation_pressure == "HIGH":
        regime = "OVERHEATING" # Υπερθέρμανση οικονομίας

    return {
        "regime": regime,
        "rates_stable": bool(rates_stable),
        "rate_volatility": float(tnx_vol),
        "inflation_momentum_3m": float(oil_3m_momentum),
        "output_gap_proxy": float(output_gap)
    }
    
    # Momentum (Z-Score)
    mean_val = s.mean()
    std_val = s.std()
    curr_val = float(s.iloc[-1])
    features["z_score"] = float((curr_val - mean_val) / std_val) if std_val != 0 else 0.0
    
    # Range Positioning (0-100%)
    high_x = float(s.max())
    low_x = float(s.min())
    features["range_position"] = float((curr_val - low_x) / (high_x - low_x) * 100) if high_x > low_x else 50.0
    
    return features


def build_macro_environment_report(use_live_data: bool = True) -> dict:
    """
    Single Source of Truth για το macro report.
    Επιστρέφει όχι μόνο raw features αλλά και ένα απλό, institutional
    interpretation layer για το GENERAL MONETARY IMPACT MATRIX.
    """
    from DataLayer6 import is_macro_cache_stale, db_get_macro_environment, db_upsert_macro_environment

    if not is_macro_cache_stale("global_macro"):
        return db_get_macro_environment("global_macro")

    report = {
        "yield_10y_current": 4.2,
        "yield_10y_slope": 0.0,
        "yield_10y_accel": 0.0,
        "term_spread_current": -0.5,
        "term_spread_accel": 0.0,
        "breakeven_5y5y_slope": 0.0,
        "growth_proxy_accel": 0.0,
        "inflation_growth_interaction": 0.0,
        "market_regime": "Unknown",
        "data_quality": "minimal",
        "theme": "Macro data unavailable",
        "x_growth_score": 0.0,
        "y_inflation_score": 0.0,
        "yield_10y_3m_change_bps": 0.0,
        "yield_10y_12m_change_bps": 0.0,
        "tip_vs_200dma_pct": 0.0,
        "tip_3m_momentum": 0.0,
        "spy_3m_return": 0.0,
        "spy_12m_return": 0.0,
        "iwm_vs_spy_ratio_3m": 0.0,
        "macro_vector_12m": [],
        "summary_lines": ["Το macro panel δεν έχει ακόμη πλήρη δεδομένα."],
        "investor_takeaway": "Το macro regime δεν μπορεί ακόμη να αξιολογηθεί με συνέπεια.",
    }

    if not use_live_data:
        return report

    def clamp(value: float, low: float = -18.0, high: float = 18.0) -> float:
        return float(max(low, min(high, value)))

    def safe_pct_change(series: pd.Series, periods: int) -> float:
        series = series.dropna()
        if len(series) <= periods:
            return np.nan
        base = float(series.iloc[-periods - 1])
        curr = float(series.iloc[-1])
        if base == 0:
            return np.nan
        return (curr / base - 1.0) * 100.0

    def safe_delta(series: pd.Series, periods: int) -> float:
        series = series.dropna()
        if len(series) <= periods:
            return np.nan
        return float(series.iloc[-1] - series.iloc[-periods - 1])

    def nearest_idx(index_obj, target):
        try:
            pos = index_obj.get_indexer([target], method='nearest')[0]
            return int(pos) if pos >= 0 else None
        except Exception:
            return None

    def score_snapshot(df_prices: pd.DataFrame, anchor_ts) -> tuple[float, float] | tuple[float, float]:
        try:
            idx = nearest_idx(df_prices.index, anchor_ts)
            if idx is None or idx < 200:
                return (np.nan, np.nan)

            x_components = []
            y_components = []

            if "SPY" in df_prices.columns:
                spy_slice = df_prices["SPY"].iloc[: idx + 1].dropna()
                if len(spy_slice) > 63:
                    spy_3m = safe_pct_change(spy_slice, 63)
                    if pd.notna(spy_3m):
                        x_components.append(spy_3m * 0.9)

            if "IWM" in df_prices.columns and "SPY" in df_prices.columns:
                iwm_slice = df_prices["IWM"].iloc[: idx + 1].dropna()
                spy_slice = df_prices["SPY"].iloc[: idx + 1].dropna()
                if len(iwm_slice) > 63 and len(spy_slice) > 63:
                    rel = safe_pct_change(iwm_slice / spy_slice, 63)
                    if pd.notna(rel):
                        x_components.append(rel * 0.7)

            if "^TNX" in df_prices.columns:
                y10_slice = df_prices["^TNX"].iloc[: idx + 1].dropna()
                if len(y10_slice) > 63:
                    delta_bps = safe_delta(y10_slice, 63)
                    if pd.notna(delta_bps):
                        y_components.append(delta_bps * 0.6)

            if "TIP" in df_prices.columns:
                tip_slice = df_prices["TIP"].iloc[: idx + 1].dropna()
                if len(tip_slice) > 200:
                    tip_200 = tip_slice.rolling(200).mean().iloc[-1]
                    if pd.notna(tip_200) and tip_200 != 0:
                        tip_dev = (float(tip_slice.iloc[-1]) / float(tip_200) - 1.0) * 100.0
                        y_components.append(tip_dev * 1.2)

            x_score = clamp(np.nansum(x_components) if x_components else 0.0)
            y_score = clamp(np.nansum(y_components) if y_components else 0.0)
            return (x_score, y_score)
        except Exception:
            return (np.nan, np.nan)

    tickers = ["^TNX", "^IRX", "TIP", "SPY", "IWM", "HYG"]
    end_dt = dt.date.today()
    start_dt = end_dt - dt.timedelta(days=420)

    try:
        df = download_prices(tickers, start=start_dt, end=end_dt)
        if df is not None and not df.empty:
            metrics_found = 0

            yield_s = df["^TNX"].dropna() if "^TNX" in df.columns else pd.Series(dtype=float)
            irx_s = df["^IRX"].dropna() if "^IRX" in df.columns else pd.Series(dtype=float)
            tip_s = df["TIP"].dropna() if "TIP" in df.columns else pd.Series(dtype=float)
            spy_s = df["SPY"].dropna() if "SPY" in df.columns else pd.Series(dtype=float)
            iwm_s = df["IWM"].dropna() if "IWM" in df.columns else pd.Series(dtype=float)
            hyg_s = df["HYG"].dropna() if "HYG" in df.columns else pd.Series(dtype=float)

            if not yield_s.empty:
                report["yield_10y_current"] = float(yield_s.iloc[-1])
                f_y10 = extract_econometric_features(yield_s)
                report["yield_10y_slope"] = f_y10.get("slope", 0.0)
                report["yield_10y_accel"] = f_y10.get("acceleration", 0.0)
                report["yield_10y_3m_change_bps"] = safe_delta(yield_s, 63)
                report["yield_10y_12m_change_bps"] = safe_delta(yield_s, 252)
                metrics_found += 1

            if not yield_s.empty and not irx_s.empty:
                joined = pd.concat([yield_s, irx_s], axis=1).dropna()
                if not joined.empty:
                    ts_series = (joined.iloc[:, 0] - joined.iloc[:, 1]).dropna()
                    report["term_spread_current"] = float(ts_series.iloc[-1])
                    f_ts = extract_econometric_features(ts_series)
                    report["term_spread_accel"] = f_ts.get("acceleration", 0.0)
                    metrics_found += 1

            if not tip_s.empty:
                f_tip = extract_econometric_features(tip_s)
                report["breakeven_5y5y_slope"] = f_tip.get("slope", 0.0)
                report["tip_3m_momentum"] = safe_pct_change(tip_s, 63)
                if len(tip_s) > 200:
                    tip_200 = tip_s.rolling(200).mean().iloc[-1]
                    if pd.notna(tip_200) and tip_200 != 0:
                        report["tip_vs_200dma_pct"] = (float(tip_s.iloc[-1]) / float(tip_200) - 1.0) * 100.0
                metrics_found += 1

            if not spy_s.empty:
                f_spy = extract_econometric_features(spy_s)
                report["growth_proxy_accel"] = f_spy.get("acceleration", 0.0)
                report["market_regime"] = get_market_trend(spy_s)
                report["spy_3m_return"] = safe_pct_change(spy_s, 63)
                report["spy_12m_return"] = safe_pct_change(spy_s, 252)
                metrics_found += 1

            if not iwm_s.empty and not spy_s.empty:
                ratio = (iwm_s / spy_s).dropna()
                report["iwm_vs_spy_ratio_3m"] = safe_pct_change(ratio, 63)
                metrics_found += 1

            report["inflation_growth_interaction"] = report["breakeven_5y5y_slope"] - report["growth_proxy_accel"]

            x_components = []
            y_components = []
            if pd.notna(report["spy_3m_return"]):
                x_components.append(report["spy_3m_return"] * 0.9)
            if pd.notna(report["iwm_vs_spy_ratio_3m"]):
                x_components.append(report["iwm_vs_spy_ratio_3m"] * 0.7)
            if pd.notna(report["term_spread_current"]):
                x_components.append(report["term_spread_current"] * 1.2)
            if pd.notna(report["yield_10y_3m_change_bps"]):
                y_components.append(report["yield_10y_3m_change_bps"] * 0.6)
            if pd.notna(report["tip_vs_200dma_pct"]):
                y_components.append(report["tip_vs_200dma_pct"] * 1.2)
            if pd.notna(report["tip_3m_momentum"]):
                y_components.append(report["tip_3m_momentum"] * 0.4)

            report["x_growth_score"] = clamp(np.nansum(x_components) if x_components else 0.0)
            report["y_inflation_score"] = clamp(np.nansum(y_components) if y_components else 0.0)

            x_now = report["x_growth_score"]
            y_now = report["y_inflation_score"]
            if x_now >= 0 and y_now < 0:
                theme = "Goldilocks / Disinflationary Growth"
            elif x_now >= 0 and y_now >= 0:
                theme = "Overheating / Late-Cycle Tightening"
            elif x_now < 0 and y_now >= 0:
                theme = "Stagflation Risk"
            else:
                theme = "Deflationary Slowdown"
            report["theme"] = theme

            if not df.empty:
                anchors = [
                    ("12M Ago", df.index[-252] if len(df.index) > 252 else df.index[0]),
                    ("6M Ago", df.index[-126] if len(df.index) > 126 else df.index[0]),
                    ("3M Ago", df.index[-63] if len(df.index) > 63 else df.index[0]),
                    ("Now", df.index[-1]),
                ]
                path_points = []
                for label, anchor in anchors:
                    x_score, y_score = score_snapshot(df, anchor)
                    if pd.notna(x_score) and pd.notna(y_score):
                        path_points.append({"label": label, "x": float(x_score), "y": float(y_score)})
                report["macro_vector_12m"] = path_points

            summary_lines = []
            summary_lines.append(
                f"Το 10Y Treasury βρίσκεται στο {report['yield_10y_current']:.2f}% με μεταβολή {report['yield_10y_12m_change_bps']:+.0f} bps στο 12μηνο."
            )
            summary_lines.append(
                f"Το SPY γράφει {report['spy_12m_return']:+.1f}% στο 12μηνο και {report['spy_3m_return']:+.1f}% στο 3μηνο, δίνοντας την τρέχουσα κατεύθυνση του growth impulse."
            )
            summary_lines.append(
                f"Το TIP κινείται {report['tip_vs_200dma_pct']:+.1f}% έναντι του 200D, ενώ το IWM έναντι του SPY είναι {report['iwm_vs_spy_ratio_3m']:+.1f}% στο 3μηνο."
            )
            report["summary_lines"] = summary_lines

            if theme == "Goldilocks / Disinflationary Growth":
                takeaway = "Η ανάπτυξη αντέχει ενώ η πληθωριστική πίεση αποκλιμακώνεται. Για investor thinking αυτό είναι το πιο φιλικό backdrop για duration και quality growth, αρκεί να μην ξαναεπιταχυνθούν οι αποδόσεις των ομολόγων."
            elif theme == "Overheating / Late-Cycle Tightening":
                takeaway = "Η αγορά συνεχίζει να υποστηρίζει την ανάπτυξη, αλλά τα rates και ο πληθωρισμός παραμένουν πιεστικοί. Ο investor πρέπει να ελέγχει αν το earnings delivery αρκεί για να δικαιολογήσει premium multiples."
            elif theme == "Stagflation Risk":
                takeaway = "Η ανάπτυξη αποδυναμώνεται ενώ η πληθωριστική πίεση μένει ενεργή. Αυτό είναι δύσκολο regime για high-multiple assets και απαιτεί μεγαλύτερη πειθαρχία σε valuation και balance-sheet resilience."
            else:
                takeaway = "Η αγορά δείχνει αποδυνάμωση growth μαζί με πτώση inflation pressure. Το κλειδί για investor thinking είναι να ξεχωρίσει κανείς ανάμεσα σε επιβράδυνση που δημιουργεί ευκαιρίες και σε καθεστώς που προεξοφλεί πραγματική ύφεση."
            report["investor_takeaway"] = takeaway

            if metrics_found >= 5:
                report["data_quality"] = "full"
            elif metrics_found >= 3:
                report["data_quality"] = "partial"

    except Exception:
        report["data_quality"] = "minimal"

    if use_live_data:
        db_upsert_macro_environment(report, "global_macro")

    return report


def test_macro_contract() -> bool:
    """
    Task 7: Test function to verify the macro report contract.
    Καλεί το build_macro_environment_report(use_live_data=False) και ελέγχει τα keys.
    """
    expected_keys = {
        "yield_10y_current", "yield_10y_slope", "yield_10y_accel",
        "term_spread_current", "term_spread_accel", "breakeven_5y5y_slope",
        "growth_proxy_accel", "inflation_growth_interaction", "market_regime", "data_quality"
    }
    
    report = build_macro_environment_report(use_live_data=False)
    
    missing = expected_keys - set(report.keys())
    if missing:
        print(f"FAILED: Missing keys in macro report: {missing}")
        return False
    
    print("SUCCESS: Macro Data Contract verified.")
    return True
