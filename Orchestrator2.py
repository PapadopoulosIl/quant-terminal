"""
Orchestrator2.py — Application Service Layer (Core Core)
========================================================
Refactored Data Architecture: Single Source of Truth
Implements the Enterprise Repository Pattern to decouple
business logic from the actual data retrieval mechanisms.

The orchestrator now NEVER imports `yfinance`. It only speaks
to an abstract `MarketDataRepository` interface.
"""
from __future__ import annotations
import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import numpy as np
from utils import safe_float
import Technical_engine11 as tech_engine
import riskengine8_2 as risk_math
from core_analysis import SignalStrength

# ---------------------------------------------------------------------------
# DATA ARCHITECTURE: The Repository Interfaces
# ---------------------------------------------------------------------------

class MarketDataRepository(ABC):
    """
    Το "Συμβόλαιο" / Single Source of Truth για όλα τα Market Data.
    Ο Orchestrator απαιτεί αυτά τα δεδομένα, αλλά δεν τοννοιάζει αν έρχονται
    από InfluxDB, PostgreSQL, αρχεία CSV ή Live API.
    """
    
    @abstractmethod
    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        """Επιστρέφει DataFrame με close prices (OHLCV optional)."""
        pass

    @abstractmethod
    def get_snapshot(self, tickers: list[str]) -> dict:
        """Επιστρέφει το live snapshot (current_price, prev_close, spark)."""
        pass

    @abstractmethod
    def get_returns(self, tickers: list[str]) -> pd.DataFrame:
        """Επιστρέφει DataFrame 1-year historical returns για Risk Engines."""
        pass

    @abstractmethod
    def get_fundamentals(self, ticker: str, as_of_date: dt.date = None) -> dict:
        """Επιστρέφει fundamentals snapshot (forward PE, EPS, margins)."""
        pass

    @abstractmethod
    def get_live_briefing(self, ticker: str) -> dict:
        """Επιστρέφει intraday & momentum data."""
        pass

    @abstractmethod
    def get_benchmark_weight(self, benchmark: str, symbol: str) -> dict:
        """Επιστρέφει τη στάθμιση της μετοχής στον δείκτη."""
        pass

    @abstractmethod
    def get_macro_environment(self) -> dict:
        """Επιστρέφει βασικά econometrics (yield derivatives, inflation, growth proxies)."""
        pass


# ---------------------------------------------------------------------------
# CONCRETE REPOSITORIES 
# ---------------------------------------------------------------------------

class LiveYFinanceRepository(MarketDataRepository):
    """
    Production-ready implementation χρησιμοποιώντας το υπάρχον market_layer.
    Καταναλώνει live δεδομένα μέσω yfinance streams / web scrapers.
    """
    def __init__(self):
        # Αργότερα, το marketLayer5 μπορεί να εισαχθεί δυναμικά
        try:
            import market_layer
            self.api = market_layer
        except ImportError:
            import marketLayer5
            self.api = marketLayer5

    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        return self.api.download_prices(tickers, start, end)

    def get_snapshot(self, tickers: list[str]) -> dict:
        return self.api.fetch_market_snapshot(tickers)

    def get_returns(self, tickers: list[str]) -> pd.DataFrame:
        return self.api.fetch_historical_returns(tickers)

    def get_fundamentals(self, ticker: str, as_of_date: dt.date = None) -> dict:
        return self.api.fetch_fundamentals_data(ticker)

    def get_live_briefing(self, ticker: str) -> dict:
        return self.api.fetch_live_briefing_data(ticker)

    def get_benchmark_weight(self, benchmark: str, symbol: str) -> dict:
        return self.api.fetch_benchmark_membership_data(benchmark, symbol)

    def get_macro_environment(self) -> dict:
        """
        Αντλεί Live Econometric Macro Features από το Feature Engineering layer (marketLayer5.py).
        Υλοποιεί Cache-First architecture μέσω του DataLayer για βέλτιστο performance.
        """
        try:
            from DataLayer6 import is_macro_cache_stale, db_get_macro_environment, db_upsert_macro_environment
            
            # Check Cache
            if not is_macro_cache_stale("global_macro"):
                return db_get_macro_environment("global_macro")
                
            # If Stale, Compute Live
            if hasattr(self.api, "build_macro_environment_report"):
                report = self.api.build_macro_environment_report()
                db_upsert_macro_environment(report, "global_macro")
                return report
            else:
                return {}
        except Exception:
            # Fallback direct call if DataLayer fails
            try:
                if hasattr(self.api, "build_macro_environment_report"):
                    return self.api.build_macro_environment_report()
            except Exception:
                pass
            return {}


class TimeSeriesDBRepository(MarketDataRepository):
    """
    Το Future-State Database implementation. 
    Ο Orchestrator θα περάσει σε αυτό ΜΗΔΕΝΙΚΟ REFACTOR.
    Διαβάζει αποκλειστικά από την InfluxDB/TimescaleDB.
    """
    def __init__(self, db_pool):
        self.db = db_pool
        
    def get_prices(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        # SQL: SELECT time, close FROM market_data WHERE ticker IN (...) AND time >= ...
        raise NotImplementedError("Pending DBA Team Deployment")

    def get_snapshot(self, tickers: list[str]) -> dict:
        raise NotImplementedError("Pending DBA Team Deployment")

    def get_returns(self, tickers: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Pending DBA Team Deployment")

    def get_fundamentals(self, ticker: str, as_of_date: dt.date = None) -> dict:
        raise NotImplementedError("Pending SQL Engine implementation")

    def get_live_briefing(self, ticker: str) -> dict:
        raise NotImplementedError()

    def get_benchmark_weight(self, benchmark: str, symbol: str) -> dict:
        raise NotImplementedError()

    def get_macro_environment(self) -> dict:
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# ORCHESTRATION PIPELINES (BUSINESS LOGIC)
# ---------------------------------------------------------------------------

class HedgeFundAllocationEngine:
    """
    Continuous Gradients & Portfolio Optimization (PhD Econometrics Architecture).
    - Layer 1: Kill Switches
    - Layer 2: Continuous Scoring [0-100]
    - Layer 3: Risk-Adjusted Allocation (Target vs Current)
    """

    @staticmethod
    def layer1_kill_switches(quant: dict) -> list[str]:
        kills = []
        cvar_pct = quant.get("cvar_95_pct", np.nan)
        if pd.notna(cvar_pct) and cvar_pct > 15.0:
            kills.append("SYSTEM HALT: Το Expected Shortfall > 15%. Απαιτείται άμεσο De-Exposure.")
        return kills

    @staticmethod
    def layer2_scoring_engine(asset: str, row: pd.Series, returns: pd.Series) -> float:
        """Normalized Scoring [0-100] avoiding binary IFs."""
        score = 50.0  # Base neutral
        
        # 1. Trend/Momentum Gradient (-20 to +20)
        pl_pct = row.get("Unrealized P/L (%)", np.nan)
        if pd.notna(pl_pct):
            score += max(-20.0, min(20.0, pl_pct / 2.0))
            
        # 2. Volatility Penalty Gradient
        if not returns.empty:
            vol = returns.std() * np.sqrt(252) * 100
            if pd.notna(vol):
                penalty = max(-15.0, (vol - 25.0) * -0.5) # Penalty if vol > 25%
                score += min(5.0, penalty)
        
        return max(0.0, min(100.0, score))

    @staticmethod
    def layer3_allocation_engine(port_df: pd.DataFrame, returns_df: pd.DataFrame, quant: dict) -> pd.DataFrame:
        if port_df.empty:
            return pd.DataFrame()
            
        allocations = []
        total_val = port_df["Current Value"].sum()
        if total_val <= 0:
            return pd.DataFrame()
            
        # Global Cap
        max_asset_weight = 15.0 
        if quant.get("rolling_corr_alert", False):
            max_asset_weight = 8.0 # Regime shift lowers the concentration cap
            
        for _, row in port_df.iterrows():
            val = row.get("Current Value", np.nan)
            if pd.isna(val) or val <= 0:
                continue
                
            ticker = row["Ticker"]
            current_wt = (val / total_val) * 100
            
            asset_rets = returns_df[ticker] if ticker in returns_df.columns else pd.Series(dtype=float)
            score = HedgeFundAllocationEngine.layer2_scoring_engine(ticker, row, asset_rets)
            
            # Map Score to Target Weight (Gradient)
            # Score 100 -> max_asset_weight, Score 0 -> 0%
            target_wt = (score / 100.0) * max_asset_weight
            
            drift = target_wt - current_wt
            
            action = "HOLD"
            if drift > 2.5:
                action = "BUY / ADD"
            elif drift < -2.5:
                action = "TRIM / REDUCE"
                
            allocations.append({
                "Ticker": ticker,
                "Conviction": score,
                "Current Wt (%)": current_wt,
                "Target Wt (%)": target_wt,
                "Drift (%)": drift,
                "Action": action,
                "_abs": abs(drift)
            })
            
        df = pd.DataFrame(allocations).sort_values(by="_abs", ascending=False).reset_index(drop=True)
        df["Priority"] = ["#" + str(i+1) for i in df.index]
        return df.drop(columns=["_abs"])


def run_portfolio_pipeline(
    txns: pd.DataFrame, 
    repo: MarketDataRepository, 
    analysis: dict = None,
    debug_mode: bool = False
) -> dict:
    """
    Dependency Injection του Repository: Διαχειρίζεται όλο το Application Flow
    χωρίς να κάνει το ίδιο το calculation — ορίζει ΠΟΙΟΣ κάνει ΤΙ και συντονίζει.
    """
    if txns.empty:
        return {"port_df": pd.DataFrame(), "quant": {}, "valuation": {}, "total_val": 0, "insights": []}
        
    # Domain Imports (Dynamic to avoid circular dependencies if used as script)
    try:
        from Portfolio_engine4 import compute_portfolio_state, compute_macro_exposure_profile
        from riskengine8_2 import run_risk_engine
        from Valuation_engine9 import run_valuation_engine
        from Advisory_layer3 import generate_advisory_insights
    except ImportError:
        pass # Handle imports dynamically based on runtime hooks
        
    unique_tickers = [t.upper() for t in txns["ticker"].unique()]
    
    unique_tickers_with_bench = unique_tickers.copy()
    if "SPY" not in unique_tickers_with_bench:
        unique_tickers_with_bench.append("SPY")
    if "QQQ" not in unique_tickers_with_bench:
        unique_tickers_with_bench.append("QQQ")
    
    # 1. Fetching Phase (Pure API call to the Repo Interface)
    market_data = repo.get_snapshot(unique_tickers)
    returns_df = repo.get_returns(unique_tickers)
    
    # NEW: Fetch real fundamentals from DB/API
    from marketLayer5 import enrich_metadata
    from DataLayer6 import db_get_metadata
    enrich_metadata(unique_tickers_with_bench) # Ensure DB is updated
    meta_db_df = db_get_metadata(unique_tickers_with_bench)
    # Ensure ticker column in meta_db_df is uppercase for indexing
    if not meta_db_df.empty:
        meta_db_df["ticker"] = meta_db_df["ticker"].str.upper()
    meta_idx = meta_db_df.set_index("ticker") if not meta_db_df.empty else pd.DataFrame()
    
    # 2. Portfolio Construction Phase 
    # Force txns ticker to upper for state computation consistency
    txns_upper = txns.copy()
    txns_upper["ticker"] = txns_upper["ticker"].str.upper()
    port_df = compute_portfolio_state(txns_upper, market_data)
    if port_df.empty:
        return {"port_df": pd.DataFrame(), "quant": {}, "valuation": {}, "total_val": 0, "insights": []}
    
    total_val = float(port_df["Current Value"].sum())
    
    # Build meta DataFrame and sector_map from DB metadata (Enriched)
    meta_rows = []
    sector_map = {}
    for t in unique_tickers_with_bench:
        # Default safe values to ensure coverage even if API fails
        row_data = {
            "ticker": t,
            "beta": 1.0,
            "trailing_pe": 18.0,
            "forward_pe": 16.0,
            "div_yield": 0.0,
            "price_to_sales": 2.0
        }
        
        # Override with real DB data if available
        if not meta_idx.empty and t in meta_idx.index:
            row = meta_idx.loc[t]
            sector_map[t] = str(row.get("sector", "Unknown"))
            
            # Map DB columns (which are lowercase) to row_data
            for db_col, key in [("beta", "beta"), ("trailing_pe", "trailing_pe"), 
                               ("forward_pe", "forward_pe"), ("div_yield", "div_yield"),
                               ("price_to_sales", "price_to_sales")]:
                val = row.get(db_col)
                if pd.notna(val) and val != 0:
                    row_data[key] = safe_float(val)
        else:
            sector_map[t] = "Unknown"
            
        # Add Aliases for engines that expect different casing (e.g. Portfolio_engine uses trailingPE)
        row_data["trailingPE"] = row_data["trailing_pe"]
        meta_rows.append(row_data)
        
    meta_df = pd.DataFrame(meta_rows)
    
    # 3. Engines Execution Phase (Quant & Valuation & Macro Exposure)
    # Fetch risk-free rate for valuation
    macro = repo.get_macro_environment()
    rf_rate = safe_float(macro.get("yield_10y_current", 4.3)) / 100.0
    
    snapshot_data = {}
    for t in unique_tickers:
        if t in market_data and "snapshot" in market_data[t]:
            snapshot_data[t] = market_data[t]["snapshot"]
    
    quant = run_risk_engine(port_df, returns_df, meta_df=meta_df, sector_map=sector_map, snapshot_data=snapshot_data)
    valuation = run_valuation_engine(port_df, meta_df, risk_free_rate=rf_rate) 
    
    try:
        from Portfolio_engine4 import compute_macro_exposure_profile
        macro_exposure_profile = compute_macro_exposure_profile(port_df, meta_df, sector_map)
    except ImportError:
        macro_exposure_profile = {}
    
    # 3.5 DECISION ENGINE: PhD Allocations (Continuous Gradients)
    kills = HedgeFundAllocationEngine.layer1_kill_switches(quant)
    alloc_df = HedgeFundAllocationEngine.layer3_allocation_engine(port_df, returns_df, quant)
    
    constraints = {"kill_switches": kills, "allocations": alloc_df}
    
    # 3.8 Earnings Pulse (Aspect 10 - Event Intelligence)
    try:
        from EarningsPulseEngine10 import run_earnings_pulse
        earnings_notes = run_earnings_pulse(port_df, market_data, meta_df=meta_df)
    except Exception:
        earnings_notes = []

    # 4. Unified Synthesis Phase (Layer C Coordination)
    from Advisory_layer3 import translate_metrics_to_plain_greek
    
    # Risk Engine is primary for Regime and Risk Level
    unified_risk = quant.get("unified_analysis")
    # Valuation Engine provides additional context
    unified_val = valuation.get("unified_analysis")
    
    # Consolidate into a Final Unified Result for the System
    final_unified = unified_risk if unified_risk else unified_val
    if unified_risk and unified_val:
        # Merge key drivers
        final_unified.key_drivers.extend(unified_val.key_drivers)
        # Valuation might influence signal (e.g. if Sell in valuation, downgrade signal)
        if unified_val.overall_signal == SignalStrength.SELL:
             final_unified.overall_signal = SignalStrength.SELL
    
    # Apply XAI (Explainable AI) Translation
    if final_unified:
        final_unified = translate_metrics_to_plain_greek(final_unified)
        quant["unified_analysis"] = final_unified

    # 5. Advisory Synthesis Phase
    macro_report = repo.get_macro_environment()
    try:
        from Advisory_layer3 import generate_advisory_insights, generate_dynamic_alerts
        insights = generate_advisory_insights(
            quant, valuation, total_val, 
            macro_report=macro_report, 
            macro_exposure=macro_exposure_profile,
            earnings_notes=earnings_notes,
            stress_results=quant.get("stress_results")
        )
        alerts = generate_dynamic_alerts(risk_df)
        insights.extend(alerts)
    except Exception:
        insights = []

        
    # Task 6: Logging / Debug Mode
    if debug_mode:
        print("\n--- DEBUG: MACRO REPORT ---")
        print(macro_report)
        print("\n--- DEBUG: MACRO EXPOSURE PROFILE ---")
        print(macro_exposure_profile)
        print("---------------------------\n")
    
    # Create Risk DataFrame
    risk_df_data = []
    total_val_safe = total_val if total_val > 0 else 1
    risk_decomp = quant.get("risk_decomposition", {})
    contextual_vol = quant.get("contextual_volatility", {})
    
    for _, row in port_df.iterrows():
        ticker = row["Ticker"]
        val = row.get("Current Value", 0)
        weight = (val / total_val_safe) * 100
        
        rd = risk_decomp.get(ticker, {})
        rc_pct = rd.get("RC_pct", 0) * 100
        
        cv = contextual_vol.get(ticker, {})
        ind_vol = cv.get("volatility", np.nan)
        pos_52w = cv.get("52w_position", np.nan) * 100 if pd.notna(cv.get("52w_position", np.nan)) else np.nan
        
        risk_df_data.append({
            "Ticker": ticker,
            "Weight (%)": weight,
            "Volatility (%)": ind_vol,
            "Risk Contribution (%)": rc_pct,
            "52w Position (%)": pos_52w
        })
    risk_df = pd.DataFrame(risk_df_data)

    return {
        "port_df": port_df,
        "risk_df": risk_df,
        "quant": quant,
        "valuation": valuation,
        "total_val": total_val,
        "insights": insights,
        "constraints": constraints,
        "earnings_notes": earnings_notes,
        "macro_state": macro_report,
    }


def build_analysis(
    ticker: str, 
    benchmark: str, 
    beta_window: int, 
    repo: MarketDataRepository
) -> dict:
    """
    Screener Pipeline. 
    Συνδέει Fundametals, Live Pricing και Analytics χρησιμοποιώντας 
    το Unified Data Repository Interface.
    """
    # Module Imports
    from Advisory_layer3 import build_valuation_summary, build_regime_text, build_volatility_text
    
    # --- Layer A: Data Ingestion (Raw Data) ---
    fundamentals = repo.get_fundamentals(ticker)
    benchmark_funds = repo.get_fundamentals(benchmark)
    spy_funds = repo.get_fundamentals("SPY")
    
    live_briefing = repo.get_live_briefing(ticker)
    returns_df = repo.get_returns([ticker, benchmark])
    membership = repo.get_benchmark_weight(benchmark, ticker)
    macro_report = repo.get_macro_environment()
    
    # --- Layer B: Analytical Computations (Mathematical Transforms) ---
    latest_metrics = {}
    if ticker in returns_df.columns and benchmark in returns_df.columns:
        stock_rets = returns_df[ticker].dropna().tail(beta_window)
        bench_rets = returns_df[benchmark].dropna().tail(beta_window)
        
        if len(stock_rets) > 30:
            # Risk Math (Beta, Sharpe, Drawdown)
            risk = risk_math.calculate_risk_metrics(stock_rets, bench_rets)
            latest_metrics['latest_beta_benchmark'] = risk["beta"]
            latest_metrics['latest_sharpe'] = risk["sharpe"]
            latest_metrics['latest_drawdown'] = risk["max_drawdown"]
            
            # Co-integration Math
            latest_metrics['coint_pval'] = risk_math.calculate_cointegration(stock_rets, bench_rets)
            
            # Technical Math
            try:
                cumulative = (1 + stock_rets).cumprod()
                rsi_series = tech_engine.calculate_rsi(cumulative, period=14)
                latest_metrics['latest_rsi'] = float(rsi_series.iloc[-1])
            except Exception:
                latest_metrics['latest_rsi'] = 50.0 # Neutral fallback

                
    # --- Layer B: Technical Feature Extraction ---
    daily = live_briefing.get("daily", pd.DataFrame())
    intraday = live_briefing.get("intraday", pd.DataFrame())
    
    if not daily.empty:
        # Enriched Technical Features from Layer B
        tech_features = tech_engine.extract_technical_features(daily, intraday)
        live_briefing.update(tech_features)
        
        stock_series = daily["Close"].dropna()
    else:
        stock_series = pd.Series(dtype=float)
        
    sma50  = tech_engine.calculate_sma(stock_series, 50)
    sma200 = tech_engine.calculate_sma(stock_series, 200)
    
    # Volatility Bands
    bb = tech_engine.calculate_bollinger_bands(stock_series, 20, 2)
    bb_upper, bb_lower = bb["upper"], bb["lower"]
    
    # Volatility Ribbon
    std50 = tech_engine.calculate_std(stock_series, 50)
    sma50_upper = sma50 + std50
    sma50_lower = sma50 - std50
    
    # Performance Normalization
    norm_stock = tech_engine.normalize_series(stock_series, 100.0)
    
    if benchmark in returns_df.columns:
        bench_rets = returns_df[benchmark].dropna()
        norm_bench = tech_engine.compute_cumulative_return_series(bench_rets, 100.0)
        norm_bench = norm_bench.reindex(stock_series.index).ffill()
    else:
        norm_bench = pd.Series(dtype=float)

    # --- Layer C: Business Logic & Orchestration ---
    val_cards = build_valuation_summary(ticker, benchmark, fundamentals, benchmark_funds, spy_funds, macro_report=macro_report)

    # --- Prepare metrics for Econometric Narrative ---
    z_score = 0.0
    if not stock_series.empty and not sma50.empty:
        try:
            curr_price = float(stock_series.iloc[-1])
            curr_sma = float(sma50.iloc[-1])
            # std50 is calculated above
            curr_std = float(std50.iloc[-1]) if not std50.empty else 1.0
            if curr_std > 0:
                z_score = (curr_price - curr_sma) / curr_std
        except Exception:
            pass
            
    price_vs_sma50_pct = np.nan
    current_price = np.nan
    sma50_current = np.nan
    if not stock_series.empty:
        try:
            current_price = float(stock_series.iloc[-1])
        except Exception:
            pass
    if not sma50.empty:
        try:
            sma50_current = float(sma50.iloc[-1])
        except Exception:
            pass
    if pd.notna(current_price) and pd.notna(sma50_current) and sma50_current != 0:
        price_vs_sma50_pct = (current_price / sma50_current - 1.0) * 100.0

    regime_metrics = {
        "z_score": z_score,
        "coint_pval": latest_metrics.get("coint_pval", float('nan')),
        "beta": latest_metrics.get("latest_beta_benchmark", 1.0),
        "price_vs_sma50_pct": price_vs_sma50_pct,
        "current_price": current_price,
        "sma50": sma50_current,
        "realized_vol_20": live_briefing.get("realized_vol_20", np.nan),
        "volume_ratio": live_briefing.get("volume_ratio", np.nan),
        "ret_1m": live_briefing.get("ret_1m", np.nan),
        "dist_high": live_briefing.get("dist_high", np.nan),
        "dist_low": live_briefing.get("dist_low", np.nan),
        "trend_score": live_briefing.get("trend_score", np.nan),
    }
    
    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "financials": fundamentals,
        "benchmark_financials": benchmark_funds,
        "spy_financials": spy_funds,
        "live_briefing": live_briefing,
        "latest_metrics": latest_metrics,
        "benchmark_membership": membership,
        "valuation_cards": val_cards,
        "macro_report": macro_report,
        "regime_text": build_regime_text(ticker, benchmark, regime_metrics),
        "vol_text": build_volatility_text(live_briefing.get("volume_ratio")),
        "stock": stock_series,
        "sma50": sma50,
        "sma200": sma200,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "sma50_upper": sma50_upper,
        "sma50_lower": sma50_lower,
        "normalized_stock": norm_stock,
        "normalized_benchmark": norm_bench
    }
