"""
portfolio_engine.py — Portfolio Engine: FIFO, Positions, P/L
============================================================
Responsible for:
  - FIFO lot matching per ticker
  - Computing open positions: avg cost, cost basis, unrealized/realized P/L
  - Day's gain using prev-close anchor
  - Weighted portfolio-level beta + dividend yield
  - Sell validation (cannot sell more than held)

Contract:
  Input:  transactions DataFrame + market_data dict
  Output: pd.DataFrame (one row per open position)
  Pure computation — no I/O, no Streamlit calls.
"""
from __future__ import annotations
import datetime as dt

import numpy as np
import pandas as pd

from utils import safe_float


# ---------------------------------------------------------------------------
# FIFO lot matching
# ---------------------------------------------------------------------------

def _compute_open_lots(grp: pd.DataFrame) -> tuple[list, float]:
    """
    FIFO lot matching for a single ticker's sorted transactions.
    Returns (buy_queue, realized_pl).
    buy_queue: list of [shares_remaining, price, timestamp_str]
    """
    buy_queue: list = []
    realized_pl = 0.0
    for row in grp.itertuples(index=False):
        if row.action == "BUY":
            buy_queue.append([float(row.shares), float(row.execution_price), row.timestamp])
        elif row.action == "SELL":
            remaining = float(row.shares)
            sell_px = float(row.execution_price)
            for lot in buy_queue:
                if remaining <= 0:
                    break
                consumed = min(lot[0], remaining)
                realized_pl += consumed * (sell_px - lot[1])
                lot[0] -= consumed
                remaining -= consumed
            buy_queue = [lot for lot in buy_queue if lot[0] > 1e-9]
    return buy_queue, realized_pl


# ---------------------------------------------------------------------------
# Public: compute full portfolio state
# ---------------------------------------------------------------------------

def compute_portfolio_state(txns: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    """
    Aggregates the full transaction ledger into one row per open ticker.

    Output columns:
      Ticker, Shares, Avg Cost, Current Price, Cost Basis,
      Current Value, Unrealized P/L ($), Unrealized P/L (%),
      Day's Gain ($), Day's Gain (%), Realized P/L ($)
    """
    if txns.empty:
        return pd.DataFrame()

    today_str = dt.date.today().isoformat()
    rows = []

    for ticker, grp in txns.groupby("ticker", sort=False):
        buy_queue, realized_pl = _compute_open_lots(grp)

        total_shares = sum(lot[0] for lot in buy_queue)
        if total_shares < 1e-9:
            continue

        total_cost = sum(lot[0] * lot[1] for lot in buy_queue)
        avg_cost   = total_cost / total_shares
        cost_basis = total_cost

        mkt = market_data.get(ticker, {})
        current_px = mkt.get("current", np.nan)
        prev_close = mkt.get("prev_close", np.nan)

        current_value  = total_shares * current_px if pd.notna(current_px) else np.nan
        unrealized_pl  = current_value - cost_basis if pd.notna(current_value) else np.nan
        unrealized_pct = (unrealized_pl / cost_basis * 100) if (pd.notna(unrealized_pl) and cost_basis) else np.nan

        all_bought_today = all(lot[2][:10] == today_str for lot in buy_queue)
        day_anchor = avg_cost if all_bought_today else prev_close
        days_gain = (current_px - day_anchor) * total_shares if (pd.notna(current_px) and pd.notna(day_anchor)) else np.nan
        days_gain_pct = (days_gain / (day_anchor * total_shares) * 100) if (pd.notna(days_gain) and day_anchor and day_anchor * total_shares != 0) else np.nan

        rows.append({
            "Ticker":            ticker,
            "Shares":            round(total_shares, 6),
            "Avg Cost":          round(avg_cost, 4),
            "Current Price":     round(float(current_px), 4) if pd.notna(current_px) else np.nan,
            "Cost Basis":        round(cost_basis, 2),
            "Current Value":     round(float(current_value), 2) if pd.notna(current_value) else np.nan,
            "Unrealized P/L ($)": round(float(unrealized_pl), 2) if pd.notna(unrealized_pl) else np.nan,
            "Unrealized P/L (%)": round(float(unrealized_pct), 2) if pd.notna(unrealized_pct) else np.nan,
            "Day's Gain ($)":    round(float(days_gain), 2) if pd.notna(days_gain) else np.nan,
            "Day's Gain (%)":    round(float(days_gain_pct), 2) if pd.notna(days_gain_pct) else np.nan,
            "Realized P/L ($)":  round(realized_pl, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_sell(txns: pd.DataFrame, ticker: str, sell_shares: float) -> tuple[bool, str]:
    """Returns (ok, error_message). Checks you can't sell more than you hold."""
    if txns.empty:
        return False, "Δεν υπάρχουν συναλλαγές για αυτό το ticker."
    t_txns = txns[txns["ticker"] == ticker.upper()]
    if t_txns.empty:
        return False, f"Δεν κατέχεις {ticker}."
    buy_queue, _ = _compute_open_lots(t_txns)
    held = sum(lot[0] for lot in buy_queue)
    if sell_shares > held + 1e-9:
        return False, f"Προσπαθείς να πουλήσεις {sell_shares:.4f} αλλά κατέχεις {held:.4f}."
    return True, ""


# ---------------------------------------------------------------------------
# Weighted portfolio-level metrics (beta, dividend yield)
# ---------------------------------------------------------------------------

def compute_weighted_metrics(port_df: pd.DataFrame, meta: pd.DataFrame, snapshot_data: dict = None, fundamentals_data: dict = None) -> dict:
    """
    Returns weighted metrics including Beta, Dividend Yield, and normalized
    cash-flow yields (FCF Yield and OCF Yield).
    """
    out = {
        "beta": np.nan,
        "div_yield_pct": np.nan,
        "fcf_yield_pct": np.nan,
        "ocf_yield_pct": np.nan,
    }
    if meta.empty or port_df.empty:
        return out

    meta_idx = meta.set_index("ticker")
    valid = port_df.dropna(subset=["Current Value"]).copy()
    if valid.empty:
        return out

    total_val = valid["Current Value"].sum()
    if total_val <= 0:
        return out

    valid["_w"] = valid["Current Value"] / total_val
    valid["_beta"] = valid["Ticker"].map(
        lambda t: safe_float(meta_idx.loc[t, "beta"]) if t in meta_idx.index else np.nan
    )
    valid["_div"] = valid["Ticker"].map(
        lambda t: safe_float(meta_idx.loc[t, "div_yield"]) if t in meta_idx.index else np.nan
    )

    if snapshot_data and fundamentals_data:
        valid["_fcf_yield"] = np.nan
        valid["_ocf_yield"] = np.nan

        for idx, row in valid.iterrows():
            ticker = row["Ticker"]
            snap = snapshot_data.get(ticker, {}) if isinstance(snapshot_data, dict) else {}
            fund = fundamentals_data.get(ticker, {}) if isinstance(fundamentals_data, dict) else {}
            market_cap = safe_float(snap.get("marketCap") or fund.get("snapshot", {}).get("market_cap"))
            quarterly = fund.get("quarterly", pd.DataFrame())

            if pd.isna(market_cap) or market_cap <= 0 or not isinstance(quarterly, pd.DataFrame) or quarterly.empty:
                continue

            ttm_ocf = np.nan
            if "Operating Cash Flow" in quarterly.index:
                ocf_row = pd.to_numeric(quarterly.loc["Operating Cash Flow"], errors="coerce").dropna()
                if len(ocf_row) >= 4:
                    ttm_ocf = float(ocf_row.iloc[:4].sum())

            ttm_capex = np.nan
            if "CapEx" in quarterly.index:
                capex_row = pd.to_numeric(quarterly.loc["CapEx"], errors="coerce").dropna()
                if len(capex_row) >= 4:
                    ttm_capex = float(capex_row.iloc[:4].sum())

            ttm_fcf = np.nan
            if "Free Cash Flow" in quarterly.index:
                fcf_row = pd.to_numeric(quarterly.loc["Free Cash Flow"], errors="coerce").dropna()
                if len(fcf_row) >= 4:
                    ttm_fcf = float(fcf_row.iloc[:4].sum())
            elif pd.notna(ttm_ocf) and pd.notna(ttm_capex):
                ttm_fcf = ttm_ocf - abs(ttm_capex)

            if pd.notna(ttm_ocf):
                valid.at[idx, "_ocf_yield"] = (ttm_ocf / market_cap) * 100.0
            if pd.notna(ttm_fcf):
                valid.at[idx, "_fcf_yield"] = (ttm_fcf / market_cap) * 100.0

    for col, out_key, scale in [
        ("_beta", "beta", 1.0),
        ("_div", "div_yield_pct", 100.0),
        ("_fcf_yield", "fcf_yield_pct", 1.0),
        ("_ocf_yield", "ocf_yield_pct", 1.0),
    ]:
        if col not in valid.columns:
            continue
        valid_subset = valid.dropna(subset=[col])
        if not valid_subset.empty:
            w_adj = valid_subset["_w"] / valid_subset["_w"].sum()
            out[out_key] = float((valid_subset[col] * w_adj).sum() * scale)

    return out

# ---------------------------------------------------------------------------
# MACRO EXPOSURE PROFILE (Cross-Talk feature)
# ---------------------------------------------------------------------------

def compute_macro_exposure_profile(port_df: pd.DataFrame, meta: pd.DataFrame, sector_map: dict, macro_inputs: dict = None) -> dict:
    """
    Computes a minimal but robust "macro exposure profile" for the portfolio.
    Provides rate sensitivity, growth bias, and sector-weighted inflation hedge score.
    """
    out = {
        "rate_sensitivity_proxy": 0.0,
        "growth_exposure": 0.0,
        "inflation_hedge_score": 0.0
    }
    
    if port_df.empty or meta.empty:
        return out
        
    valid = port_df.dropna(subset=["Current Value"]).copy()
    total_val = valid["Current Value"].sum()
    if total_val <= 0:
        return out
        
    valid["weight"] = valid["Current Value"] / total_val
    meta_idx = meta.set_index("ticker")
    
    # 1. Rate Sensitivity (Beta is a standard proxy for market discount-rate elasticity)
    valid["_beta"] = valid["Ticker"].map(
        lambda t: safe_float(meta_idx.loc[t, "beta"]) if t in meta_idx.index and "beta" in meta.columns else np.nan
    )
    beta_w = valid.dropna(subset=["_beta"])
    if not beta_w.empty:
        w_b = beta_w["weight"] / beta_w["weight"].sum()
        out["rate_sensitivity_proxy"] = float((beta_w["_beta"] * w_b).sum())
        
    # 2. Growth Exposure (Forward/Trailing PE acts as organic growth expectations premium)
    valid["_pe"] = valid["Ticker"].map(
        lambda t: safe_float(meta_idx.loc[t, "trailingPE"]) if t in meta_idx.index and "trailingPE" in meta.columns else np.nan
    )
    pe_w = valid.dropna(subset=["_pe"])
    if not pe_w.empty:
        w_pe = pe_w["weight"] / pe_w["weight"].sum()
        out["growth_exposure"] = float((pe_w["_pe"] * w_pe).sum())
        
    # 3. Inflation Hedge Score 
    # Hard-Asset & Cashflow Sectors + Structural Dividend Weight
    INFLATION_HEDGE_SECTORS = {"Energy", "Financials", "Materials", "Real Estate", "Basic Materials"}
    valid["div_val"] = valid["Ticker"].map(
        lambda t: safe_float(meta_idx.loc[t, "div_yield"]) if t in meta_idx.index and "div_yield" in meta.columns else 0.0
    )
    
    hedge_score = 0.0
    for row in valid.itertuples(index=False):
        t = str(row.Ticker)
        w = float(row.weight)
        div = float(row.div_val)
        sector = sector_map.get(t, "Unknown")
        
        # Base Allocation
        if sector in INFLATION_HEDGE_SECTORS:
             hedge_score += (w * 100.0) # Adds +1 to score per 1% exposure
             
        # Add pure cash-flow yield as structural defense
        hedge_score += (div * 100.0)
        
    out["inflation_hedge_score"] = float(round(min(100.0, hedge_score), 2))
    
    return out
