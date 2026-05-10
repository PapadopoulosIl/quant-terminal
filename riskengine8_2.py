"""Institutional risk analytics: VaR/CVaR, correlations, stress tests, macro-conditioned dominance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from core_analysis import (
    UnifiedAnalysisResult, 
    MarketRegime, 
    RiskLevel, 
    SignalStrength
)

# ---------------------------------------------------------------------------
# Step-3 model constants (no magic numbers scattered in code)
# ---------------------------------------------------------------------------
MODEL_QUAD_WEIGHT = 0.30
TOP1_PROB_THRESHOLD = 0.65
TOP2_PROB_THRESHOLD = 0.45
ENTROPY_TOPK_THRESHOLD = 0.55
ADAPTIVE_THRESHOLD_BASE = 0.12
ADAPTIVE_THRESHOLD_SLOPE = 0.30

# Interaction coefficients (interpretable cross-terms)
I_RATE_X_GROWTH_TILT = 0.25
I_CORR_X_LIQUIDITY = 0.25
I_THIRD_X_CORR = 0.20
I_CONC_X_LIQUIDITY = 0.20
I_RATE_X_INFLATION = 0.15

# ---------------------------------------------------------------------------
# Market Trend (SMA 200 logic)
# ---------------------------------------------------------------------------

def get_market_trend(spy_closes: pd.Series) -> str:
    """Institutional trend identification using 200-day Simple Moving Average."""
    if len(spy_closes) < 200:
        return "INSUFFICIENT_DATA"
    sma = spy_closes.iloc[-200:].mean()
    return "ABOVE_200SMA" if spy_closes.iloc[-1] > sma else "BELOW_200SMA"


# ---------------------------------------------------------------------------
# Scenario Shocks (για Stress Testing)
# ---------------------------------------------------------------------------
STRESS_SCENARIOS: dict[str, dict[str, Any]] = {
    "Financial Crisis (-40% S&P)": {
        "market_shock": -0.40,
        "tech_multiplier": 1.2,
        "description": "Σενάριο: Χρηματοπιστωτική κρίση (τύπου 2008). Systemic collapse, μαζική ρευστοποίηση.",
    },
    "Rate Shock (+200bps)": {
        "market_shock": -0.12,
        "tech_multiplier": 1.6,
        "description": "Σενάριο: Απότομη αύξηση επιτοκίων κατά 2%. Οι Growth/Tech μετοχές υποφέρουν περισσότερο λόγω προεξόφλησης μελλοντικών κερδών.",
    },
    "Continuous Growth (+10% S&P)": {
        "market_shock": +0.10,
        "tech_multiplier": 1.1,
        "description": "Σενάριο: Συνεχής ανάπτυξη με ελεγχόμενο πληθωρισμό. Bullish περιβάλλον (Goldilocks).",
    },
}

# Sectors που θεωρούνται "tech-heavy" για τον tech_multiplier
TECH_SECTORS = {"Technology", "Communication Services", "Consumer Cyclical"}


def calc_parametric_volatility(returns_slice: pd.DataFrame, w: np.ndarray) -> dict:
    """Υπολογίζει VAR(Rp) = W' * Σ * W και το κάνει Annualized"""
    if len(returns_slice) < 2:
        return {"cov_matrix": pd.DataFrame(), "vol_annual": np.nan}
        
    # 1. Υπολογισμός Covariance Matrix (Σ)
    cov_matrix = returns_slice.cov() 
    
    # 2. Εφαρμογή του W' * Σ * W
    port_variance = w.T @ cov_matrix.values @ w
    
    # 3. Annualization (Variance * 252, Volatility = sqrt(Variance * 252))
    port_volatility_annual = np.sqrt(port_variance * 252) * 100
    
    return {
        "cov_matrix": cov_matrix, 
        "vol_annual": float(port_volatility_annual)
    }

def compute_ewma_risk_profile(returns_df: pd.DataFrame, weights: np.ndarray, lambda_param: float = 0.94) -> dict:
    """
    Υπολογίζει τον πίνακα EWMA Covariance και το Annualized Volatility.
    Το alpha στο pandas είναι (1 - lambda).
    """
    if len(returns_df) < 2 or len(weights) != returns_df.shape[1]:
        return {"vol_annual": np.nan, "cov_matrix": pd.DataFrame()}

    alpha = 1.0 - lambda_param
    
    # Το ewm().cov() επιστρέφει MultiIndex DataFrame (Date, Ticker1) -> Ticker2
    ewma_cov_series = returns_df.ewm(alpha=alpha, adjust=True).cov()
    
    # Αντλούμε τον πίνακα Σ (Covariance Matrix) της ΤΕΛΕΥΤΑΙΑΣ ημέρας
    latest_date = returns_df.index[-1]
    latest_cov_matrix = ewma_cov_series.xs(latest_date, level=0)
    
    # Παραμετρικός Υπολογισμός Διακύμανσης Χαρτοφυλακίου: W' * Σ * W
    w = np.asarray(weights, dtype=float)
    port_variance_daily = w.T @ latest_cov_matrix.values @ w
    
    # Ετησιοποίηση
    port_volatility_annual = np.sqrt(port_variance_daily * 252) * 100
    
    return {
        "vol_annual": float(port_volatility_annual),
        "cov_matrix": latest_cov_matrix
    }

def validate_covariance_matrix(cov_matrix: pd.DataFrame) -> bool:
    """
    Διενεργεί τους 2 απαραίτητους οικονομετρικούς ελέγχους.
    Επιστρέφει True αν ο πίνακας είναι μαθηματικά έγκυρος.
    """
    if cov_matrix.empty:
        return False
        
    mat = cov_matrix.values
    
    # Έλεγχος 1: Συμμετρία (Symmetry). Το Cov(A,B) πρέπει να ισούται με Cov(B,A)
    # Χρησιμοποιούμε atol (absolute tolerance) για να αγνοήσουμε απειροελάχιστα rounding errors.
    is_symmetric = np.allclose(mat, mat.T, atol=1e-8)
    if not is_symmetric:
        print("WARNING: Covariance matrix is not symmetric.")
        return False
        
    # Έλεγχος 2: Θετικά Ημι-ορισμένος (Positive Semi-Definite - PSD)
    # Αν οι ιδιοτιμές (eigenvalues) είναι αρνητικές, το W'*Σ*W μπορεί να βγει αρνητικό,
    # οδηγώντας σε μιγαδικό αριθμό κατά τον υπολογισμό της μεταβλητότητας (τετραγωνική ρίζα).
    eigenvalues = np.linalg.eigvalsh(mat)
    is_psd = np.all(eigenvalues >= -1e-8)
    if not is_psd:
        print("WARNING: Covariance matrix is not Positive Semi-Definite.")
        return False
        
    return True


def run_risk_engine(
    port_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    meta_df: pd.DataFrame = pd.DataFrame(),
    sector_map: dict[str, str] | None = None,
    macro_report: dict[str, Any] | None = None,
    macro_exposure: dict[str, Any] | None = None,
    snapshot_data: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """
    Entry point. Returns VaR/CVaR/vol/correlations + raw/adjusted stress results and a
    macro-conditioned dominant risk narrative (top-1/top-2) with posterior probabilities.
    """
    out: dict[str, Any] = {
        "var_95": np.nan,
        "var_95_pct": np.nan,
        "cvar_95": np.nan,
        "cvar_95_pct": np.nan,
        "portfolio_vol_annual": np.nan,
        "correlation_warnings": [],
        "rolling_corr_alert": False,
        "rolling_corr_current": {},
        "stress_results_raw": {},
        "stress_results": {},
        "correlation_matrix": {},
        # ── Βήμα 2 ────────────────────────────────────────────────────────────
        "macro_context": {},
        # ── Βήμα 3 + 4 ────────────────────────────────────────────────────────
        # Advisory layer: insights.append(risk["dominant_risk_narrative"])
        # χωρίς extra logic.
        "regime_entropy": np.nan,
        "dominant_risk_narrative": "",
        "dominant_scenarios":      [],
        "regime_adjusted_stress":  {},
        "dynamic_risk_profile":    {},
        "ewma_volatility_annual":  np.nan,
        "ewma_covariance_matrix":  {},
        "risk_decomposition":      {},
        "diversification_ratio":   np.nan,
        "contextual_volatility":   {},
    }

    if port_df.empty or returns_df.empty:
        return out

    valid = port_df.dropna(subset=["Current Value"]).copy()
    total_val = valid["Current Value"].sum()
    if total_val <= 0:
        return out

    valid["wt"] = valid["Current Value"] / total_val
    tickers_in_port = [t for t in valid["Ticker"].tolist() if t in returns_df.columns]
    if not tickers_in_port:
        return out

    port_returns_raw = returns_df[tickers_in_port].copy()
    # Clean incomplete observations so VaR/CVaR and correlations do not collapse to NaN.
    port_returns_clean = port_returns_raw.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    weights = valid.set_index("Ticker").loc[tickers_in_port, "wt"].values
    weights = weights / weights.sum()

    lookbacks = {
        "1M (21d)": port_returns_clean.tail(21),
        "3M (63d)": port_returns_clean.tail(63),
        "1Y (252d)": port_returns_clean
    }

    dynamic_covariance_profiles = {}
    for label, ret_slice in lookbacks.items():
        res = calc_parametric_volatility(ret_slice, weights)
        cov_dict = res["cov_matrix"].round(6).to_dict() if isinstance(res["cov_matrix"], pd.DataFrame) and not res["cov_matrix"].empty else {}
        dynamic_covariance_profiles[label] = {
            "volatility": res["vol_annual"],
            "covariance_matrix": cov_dict
        }
        
    out["dynamic_risk_profile"] = dynamic_covariance_profiles

    # Υπολογισμός δυναμικού EWMA Risk Profile (αντί για στατικά timeframes)
    ewma_profile = compute_ewma_risk_profile(port_returns_clean, weights, lambda_param=0.94)
    cov_mat = ewma_profile["cov_matrix"]

    # Validation του Πίνακα και υπολογισμός Risk Decomposition (MCR / RC) & Diversification Ratio
    if validate_covariance_matrix(cov_mat):
        out["ewma_volatility_annual"] = ewma_profile["vol_annual"]
        out["ewma_covariance_matrix"] = cov_mat.round(6).to_dict()
        
        sigma_p_daily = np.sqrt(weights.T @ cov_mat.values @ weights)
        if sigma_p_daily > 0:
            mcr = (cov_mat.values @ weights) / sigma_p_daily
            rc = weights * mcr
            rc_pct = rc / sigma_p_daily
            
            risk_decomp = {}
            for i, ticker in enumerate(cov_mat.columns):
                risk_decomp[ticker] = {
                    "MCR": float(mcr[i] * np.sqrt(252) * 100),
                    "RC_pct": float(rc_pct[i])
                }
            out["risk_decomposition"] = risk_decomp
            
            # Diversification Ratio (DR)
            sigma_i_daily = np.sqrt(np.diag(cov_mat.values))
            weighted_vols = np.sum(weights * sigma_i_daily)
            dr = weighted_vols / sigma_p_daily
            out["diversification_ratio"] = float(dr)
            
            # Contextual Volatility (Z-Score Relative to Range)
            contextual_vol = {}
            if snapshot_data:
                for i, ticker in enumerate(cov_mat.columns):
                    snap = snapshot_data.get(ticker, {})
                    try:
                        high = float(snap.get("52WeekHigh") or snap.get("target_high") or np.nan)
                        low = float(snap.get("52WeekLow") or snap.get("target_low") or np.nan)
                        price = float(snap.get("previousClose") or snap.get("currentPrice") or np.nan)
                        if pd.notna(high) and pd.notna(low) and pd.notna(price) and high > low:
                            position = (price - low) / (high - low)
                            vol_i_annual = float(sigma_i_daily[i] * np.sqrt(252) * 100)
                            
                            # Kill switch: High vol + 52w-Low
                            kill_switch = bool(vol_i_annual > 40.0 and position < 0.25)
                            contextual_vol[ticker] = {
                                "52w_position": float(position),
                                "volatility": vol_i_annual,
                                "kill_switch": kill_switch
                            }
                    except (ValueError, TypeError):
                        pass
            out["contextual_volatility"] = contextual_vol
    else:
        # Fallback στον απλό ιστορικό υπολογισμό αν τα δεδομένα είναι κατεστραμμένα
        out["ewma_volatility_annual"] = np.nan
        out["ewma_covariance_matrix"] = {}

    macro_ctx = _quantify_macro_context(macro_report, macro_exposure)
    macro_ctx = dict(macro_ctx)
    if macro_exposure:
        macro_ctx["growth_tilt"] = float(macro_exposure.get("growth_tilt", 0.0) or 0.0)
        macro_ctx["rate_sensitivity_proxy"] = float(
            macro_exposure.get("rate_sensitivity_proxy", 1.0) or 1.0
        )
        macro_ctx["inflation_hedge_score"] = float(
            macro_exposure.get("inflation_hedge_score", 1.0) or 1.0
        )
        macro_ctx["liquidity_risk_score"] = float(
            macro_exposure.get("liquidity_risk_score", macro_exposure.get("liquidity_risk", 0.0)) or 0.0
        )
    macro_ctx["concentration_risk"] = _compute_portfolio_concentration_risk(weights)
    macro_ctx["spy_macro_climate_score"] = _compute_spy_macro_climate_score(returns_df)
    if "liquidity_risk_score" not in macro_ctx:
        macro_ctx["liquidity_risk_score"] = 0.0
    out["macro_context"] = macro_ctx

    if not port_returns_clean.empty:
        port_daily = port_returns_clean.values @ weights  # shape: (T,)
        port_daily = port_daily[np.isfinite(port_daily)]
    else:
        port_daily = np.array([])

    if len(port_daily) > 0:
        var_95_pct = float(np.percentile(port_daily, 5))
        out["var_95"] = abs(var_95_pct) * total_val
        out["var_95_pct"] = abs(var_95_pct) * 100

        tail_returns = port_daily[port_daily <= var_95_pct]
        if len(tail_returns) > 0:
            cvar_pct = float(np.mean(tail_returns))
            out["cvar_95"] = abs(cvar_pct) * total_val
            out["cvar_95_pct"] = abs(cvar_pct) * 100

        out["portfolio_vol_annual"] = float(np.std(port_daily) * np.sqrt(252) * 100)

    if len(tickers_in_port) > 1 and not port_returns_clean.empty:
        corr_matrix = port_returns_clean.corr()
        out["correlation_matrix"] = corr_matrix.round(3).to_dict()

        corr_vals = corr_matrix.values
        mask = np.triu(np.ones_like(corr_vals, dtype=bool), 1)
        idx = np.argwhere(mask & (corr_vals > 0.75))
        pairs = [
            (tickers_in_port[i], tickers_in_port[j], float(corr_vals[i, j]))
            for i, j in idx
        ]
        out["correlation_warnings"] = sorted(pairs, key=lambda x: x[2], reverse=True)

    if len(tickers_in_port) >= 2 and len(port_returns_clean) >= 30:
        rolling_corr_alert, rolling_summary = _compute_rolling_correlation_alert(
            port_returns_clean, tickers_in_port
        )
        out["rolling_corr_alert"] = rolling_corr_alert
        out["rolling_corr_current"] = rolling_summary

    out["stress_results_raw"] = _run_stress_tests(
        total_val, valid, tickers_in_port, sector_map or {}
    )
    max_pairs = max(1, len(tickers_in_port) * (len(tickers_in_port) - 1) // 2)
    macro_ctx["correlation_risk"] = float(np.clip(len(out["correlation_warnings"]) / max_pairs, 0.0, 1.0))
    macro_ctx["tail_risk_cvar_pct"] = float(out.get("cvar_95_pct", np.nan)) if pd.notna(out.get("cvar_95_pct")) else np.nan
    macro_ctx["vol_annual"] = float(out.get("portfolio_vol_annual", np.nan)) if pd.notna(out.get("portfolio_vol_annual")) else np.nan
    if not float(macro_ctx.get("liquidity_risk_score", 0.0) or 0.0):
        macro_ctx["liquidity_risk_score"] = float(
            np.clip(0.50 * macro_ctx.get("concentration_risk", 0.0) * macro_ctx.get("spy_macro_climate_score", 0.5), 0.0, 1.0)
        )
    out["macro_context"] = macro_ctx

    adjusted_stress = _apply_regime_adjustments(out["stress_results_raw"], macro_ctx)
    
    # Calculate portfolio diagnostics for dynamic hedging
    tech_exp = 0.0
    for ticker in tickers_in_port:
        if sector_map and sector_map.get(ticker) in TECH_SECTORS:
            tech_exp += weights[tickers_in_port.index(ticker)]
    tech_exp_pct = tech_exp * 100.0

    avg_beta = 1.0
    if not meta_df.empty and "beta" in meta_df.columns:
        # Use provided betas if available
        beta_map = meta_df.set_index("ticker")["beta"].to_dict()
        avg_beta = sum(weights[i] * beta_map.get(tickers_in_port[i], 1.0) for i in range(len(tickers_in_port)))

    # Generate dynamic hedge recommendations
    hedge_recs = _generate_dynamic_hedges(macro_ctx, tech_exp_pct, avg_beta, adjusted_stress)

    out["stress_results"] = {
        "scenarios": {
            k: {
                "pnl": v.get("pnl", 0.0), 
                "pct": v.get("pct", 0.0), 
                "description": v.get("description", "")
            }
            for k, v in adjusted_stress.items()
        },
        "portfolio_diagnostics": {
            "tech_growth_exposure_pct": tech_exp_pct,
            "avg_beta_to_spy": avg_beta
        },
        "hedge_recommendations": hedge_recs
    }

    dominant_views = _select_dominant_views(adjusted_stress, macro_ctx)
    out["dominant_scenarios"] = [v["scenario"] for v in dominant_views]
    out["regime_entropy"] = float(dominant_views[0].get("entropy_norm")) if dominant_views else np.nan
    out["dominant_risk_narrative"] = _generate_dominant_narrative(
        dominant_views=dominant_views,
        macro_ctx=macro_ctx,
        cvar_pct=out.get("cvar_95_pct", np.nan),
        vol_annual=out.get("portfolio_vol_annual", np.nan),
    )

    # --- Mapping σε Unified Model ---
    # Try to derive regime from SPY if available in returns_df
    regime_str = "NEUTRAL"
    if "SPY" in returns_df.columns:
        spy_rets = returns_df["SPY"].dropna()
        if not spy_rets.empty:
            spy_closes = (1 + spy_rets).cumprod()
            regime_str = get_market_trend(spy_closes)
            
    market_regime = MarketRegime.BULLISH if "ABOVE" in regime_str else MarketRegime.BEARISH

    # Risk Level Logic (βασισμένο σε CVaR + Stress)
    cvar = out.get("cvar_95_pct", 0)
    if cvar > 12:
        risk_level = RiskLevel.CRITICAL
    elif cvar > 8:
        risk_level = RiskLevel.HIGH
    elif cvar > 5:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    # Overall Signal (απλοποιημένο)
    if market_regime == MarketRegime.BULLISH and risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]:
        overall_signal = SignalStrength.BUY
    elif risk_level == RiskLevel.CRITICAL:
        overall_signal = SignalStrength.SELL
    else:
        overall_signal = SignalStrength.NEUTRAL

    # Δημιουργία Unified Object
    out["unified_analysis"] = UnifiedAnalysisResult(
        ticker=port_df["Ticker"].iloc[0] if not port_df.empty else "PORTFOLIO",
        market_regime=market_regime,
        risk_level=risk_level,
        overall_signal=overall_signal,
        confidence_score=round(0.75 + (0.2 * (1 - cvar/15)), 2) if cvar < 15 else 0.5,
        key_drivers=[
            f"CVaR 95%: {cvar:.1f}%",
            f"Market Regime: {regime_str}",
            f"Risk Level: {risk_level.value}"
        ],
        hedge_recommendation=hedge_recs[0] if hedge_recs else None,
        macro_context=macro_ctx.get("regime", "Unknown"),
        timestamp=pd.Timestamp.now().isoformat()
    )

    return out


# ---------------------------------------------------------------------------
# Βήμα 3: Dominant Risk View Generator
_SCENARIO_TAXONOMY: dict[str, dict] = {
    "Financial Crisis (-40% S&P)": {
        "rate_sensitive":   False,
        "growth_sensitive": False,
        "inflation_driven": False,
        "regime_affinity":  ["Recessionary"],
    },
    "Rate Shock (+200bps)": {
        "rate_sensitive":   True,
        "growth_sensitive": True,   # High-P/E growth hits harder
        "inflation_driven": True,
        "regime_affinity":  ["Stagflationary Pressure", "Reflationary Growth"],
    },
    "Continuous Growth (+10% S&P)": {
        "rate_sensitive":   False,
        "growth_sensitive": True,
        "inflation_driven": False,
        "regime_affinity":  ["Disinflationary Growth", "Reflationary Growth"],
    },
}


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax (temperature <1 decisive, >1 elastic)."""
    t = float(max(1e-6, temperature))
    z = (x / t) - np.max(x / t)
    e = np.exp(z)
    s = float(np.sum(e))
    return e / s if s > 0 else np.ones_like(x) / max(1, len(x))


def _normalized_entropy(p: np.ndarray) -> float:
    """Entropy normalized to [0,1]."""
    p = np.asarray(p, dtype=float)
    p = p / max(1e-12, float(np.sum(p)))
    p = np.clip(p, 1e-12, 1.0)
    h = -float(np.sum(p * np.log(p)))
    h_max = float(np.log(max(2, len(p))))
    return float(np.clip(h / h_max, 0.0, 1.0))


def _compute_portfolio_concentration_risk(weights: np.ndarray) -> float:
    """Concentration risk in [0,1] via normalized HHI."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    w = w / max(1e-12, float(np.sum(w)))
    n = int(w.size)
    if n <= 1:
        return 1.0
    hhi = float(np.sum(w * w))
    hhi_min = 1.0 / n
    return float(np.clip((hhi - hhi_min) / (1.0 - hhi_min), 0.0, 1.0))


def _compute_spy_macro_climate_score(returns_df: pd.DataFrame) -> float:
    """Third-force SPY climate score in [0,1] from 20d momentum + 60d realized vol."""
    if returns_df is None or returns_df.empty or "SPY" not in returns_df.columns:
        return 0.5

    spy = returns_df["SPY"].dropna()
    if spy.empty:
        return 0.5

    r20 = spy.tail(20)
    r60 = spy.tail(60)
    if r20.empty or r60.empty:
        return 0.5

    cum_20d = float((1.0 + r20).prod() - 1.0)
    vol_60d_ann = float(r60.std() * np.sqrt(252))
    momentum_term = -cum_20d / 0.05
    vol_term = (vol_60d_ann - 0.15) / 0.10
    z = float(np.clip(momentum_term + vol_term, -6.0, 6.0))
    score = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(score, 0.0, 1.0))


def _normalize_growth_amp(growth_amp: float) -> float:
    """Normalize growth_shock_amplifier (~[0.5,2.5]) into [0,1]."""
    ga = float(growth_amp)
    return float(np.clip((ga - 0.5) / 2.0, 0.0, 1.0))


def _normalize_regime_label(regime: str) -> str:
    """Harmonize regime strings across layers (extract canonical taxonomy label)."""
    r = str(regime or "Unknown")
    canonical = ("Disinflationary Growth", "Stagflationary Pressure", "Reflationary Growth", "Recessionary", "Transitional", "Unknown")
    for c in canonical:
        if c in r:
            return c
    return r


def _apply_regime_adjustments(
    stress_results: dict[str, Any],
    macro_ctx: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Regime-adjusts *all* stress scenarios via a bounded linear-algebra tilt:
      m = 1 + E @ Δ,  adj_pct = raw_pct * m (downside-only)
    """
    if not stress_results:
        return {}

    ctx_available = bool(macro_ctx.get("context_available", False))
    df = pd.DataFrame.from_dict(stress_results, orient="index")
    df["raw_pct"] = df.get("pct", 0.0).astype(float).fillna(0.0)
    df["raw_pnl"] = df.get("pnl", 0.0).astype(float).fillna(0.0)
    df["description"] = df.get("description", "").fillna("")
    if not ctx_available:
        df["pct"] = df["raw_pct"]
        df["pnl"] = df["raw_pnl"]
        df["applied"] = [{"identity": True}] * len(df)
        return df[["raw_pct", "pct", "raw_pnl", "pnl", "description", "applied"]].round(2).to_dict(orient="index")

    rate_pressure_score = float(np.clip(macro_ctx.get("rate_pressure_score", 0.5), 0.0, 1.0))
    growth_amp = float(np.clip(macro_ctx.get("growth_shock_amplifier", 1.0), 0.5, 2.5))
    inflation_adj = float(np.clip(macro_ctx.get("inflation_hedge_adj", 0.0), -1.0, 1.0))
    rate_mult = 0.75 + 0.50 * rate_pressure_score
    growth_mult = growth_amp
    inflation_mult = float(np.clip(1.0 + 0.30 * inflation_adj, 0.6, 1.4))
    delta = np.array([rate_mult - 1.0, growth_mult - 1.0, inflation_mult - 1.0], dtype=float)

    tax = pd.DataFrame.from_dict(_SCENARIO_TAXONOMY, orient="index").reindex(df.index)
    E = np.column_stack(
        [
            tax.get("rate_sensitive", False).fillna(False).astype(float).to_numpy(),
            tax.get("growth_sensitive", False).fillna(False).astype(float).to_numpy(),
            tax.get("inflation_driven", False).fillna(False).astype(float).to_numpy(),
        ]
    )
    downside = (df["raw_pct"].to_numpy() < 0.0).astype(float)
    E = E * downside[:, None]
    multipliers = np.clip(1.0 + (E @ delta), 0.40, 2.60)
    raw_pct = df["raw_pct"].to_numpy()
    raw_pnl = df["raw_pnl"].to_numpy()
    adj_pct = np.clip(np.where(downside > 0.0, raw_pct * multipliers, raw_pct), -99.0, 99.0)
    adj_pnl = np.where(np.abs(raw_pct) > 1e-9, raw_pnl * (adj_pct / raw_pct), raw_pnl)

    df["pct"] = adj_pct
    df["pnl"] = adj_pnl
    df["applied"] = [
        {
            "rate_mult": float(rate_mult),
            "growth_mult": float(growth_mult),
            "inflation_mult": float(inflation_mult),
            "multiplier_used": float(m),
        }
        for m in multipliers
    ]
    return df[["raw_pct", "pct", "raw_pnl", "pnl", "description", "applied"]].round(2).to_dict(orient="index")


def _adjust_stress_results_for_macro(  # backward-compat wrapper (deprecated)
    stress_results: dict[str, Any],
    macro_ctx: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    adjusted = _apply_regime_adjustments(stress_results, macro_ctx)
    public = {
        k: {"pnl": v.get("pnl", 0.0), "pct": v.get("pct", 0.0), "description": v.get("description", "")}
        for k, v in adjusted.items()
    }
    return adjusted, public


def _select_dominant_views(
    adjusted_stress: dict[str, dict[str, Any]],
    macro_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Advanced probabilistic dominance:
      Score = F @ W + MODEL_QUAD_WEIGHT * diag(F @ I @ F.T)
      posterior ∝ prior * softmax(loss) * softmax(score)
    """
    if not adjusted_stress:
        return []

    idx = pd.Index(adjusted_stress.keys(), name="scenario")
    scenario_names = idx.tolist()
    df = pd.DataFrame(index=idx)

    regime = _normalize_regime_label(str(macro_context.get("regime", "Unknown")))
    regime_weight = float(np.clip(macro_context.get("regime_weight", 0.5), 0.0, 1.0))
    rate_pressure_score = float(np.clip(macro_context.get("rate_pressure_score", 0.5), 0.0, 1.0))
    growth_amp_norm = _normalize_growth_amp(float(macro_context.get("growth_shock_amplifier", 1.0)))
    inflation_abs = float(np.clip(abs(float(macro_context.get("inflation_hedge_adj", 0.0))), 0.0, 1.0))
    third_force = float(np.clip(macro_context.get("spy_macro_climate_score", 0.5), 0.0, 1.0))
    corr_risk = float(np.clip(macro_context.get("correlation_risk", 0.0), 0.0, 1.0))
    conc_risk = float(np.clip(macro_context.get("concentration_risk", 0.0), 0.0, 1.0))
    liq_risk = float(np.clip(macro_context.get("liquidity_risk_score", 0.0), 0.0, 1.0))
    growth_tilt = float(np.clip(macro_context.get("growth_tilt", 0.0), -1.0, 1.0))

    tax = pd.DataFrame.from_dict(_SCENARIO_TAXONOMY, orient="index").reindex(idx)
    defs = pd.DataFrame.from_dict(STRESS_SCENARIOS, orient="index").reindex(idx)
    df["market_shock"] = defs.get("market_shock", 0.0).fillna(0.0).astype(float)
    df["tech_multiplier"] = defs.get("tech_multiplier", 1.0).fillna(1.0).astype(float)
    df["rate_sensitive"] = tax.get("rate_sensitive", False).fillna(False).astype(bool)
    df["growth_sensitive"] = tax.get("growth_sensitive", False).fillna(False).astype(bool)
    df["inflation_driven"] = tax.get("inflation_driven", False).fillna(False).astype(bool)
    df["regime_affinity"] = tax.get("regime_affinity", [[] for _ in range(len(idx))])
    df["regime_affinity"] = df["regime_affinity"].apply(lambda a: a if isinstance(a, list) else [])

    df["regime_match"] = df["regime_affinity"].apply(lambda a: 1.0 if (regime != "Unknown" and regime in (a or [])) else 0.0)
    df["rate_alignment"] = rate_pressure_score * df["rate_sensitive"].astype(float)
    df["growth_alignment"] = growth_amp_norm * df["growth_sensitive"].astype(float)
    df["inflation_alignment"] = inflation_abs * df["inflation_driven"].astype(float)

    severe = df["market_shock"].abs()
    severe = (severe / max(1e-9, float(severe.max()))).clip(0.0, 1.0)
    tech = (df["tech_multiplier"] - 1.0)
    tech = (tech / max(1e-9, float(tech.max()))).clip(0.0, 1.0)

    is_down = (df["market_shock"] < 0.0).astype(float)
    df["third_force"] = third_force * severe * is_down
    df["correlation_risk"] = corr_risk * severe
    df["concentration_risk"] = conc_risk * tech
    df["liquidity_risk"] = liq_risk * severe

    factor_cols = [
        "regime_match",
        "rate_alignment",
        "growth_alignment",
        "inflation_alignment",
        "third_force",
        "correlation_risk",
        "concentration_risk",
        "liquidity_risk",
    ]
    Fm = df[factor_cols].to_numpy(dtype=float)
    per_scenario_factors = df[factor_cols].to_dict(orient="index")

    W = np.array([regime_weight, rate_pressure_score, growth_amp_norm, inflation_abs, third_force, corr_risk, conc_risk, liq_risk], dtype=float)

    I = np.eye(len(factor_cols), dtype=float)
    I[1, 2] = I[2, 1] = I_RATE_X_GROWTH_TILT * abs(growth_tilt)
    I[5, 7] = I[7, 5] = I_CORR_X_LIQUIDITY
    I[4, 5] = I[5, 4] = I_THIRD_X_CORR
    I[6, 7] = I[7, 6] = I_CONC_X_LIQUIDITY
    I[1, 3] = I[3, 1] = I_RATE_X_INFLATION * inflation_abs

    scores = (Fm @ W) + MODEL_QUAD_WEIGHT * np.diag(Fm @ I @ Fm.T)
    p_model = _softmax(scores, temperature=float(np.clip(1.0 - 0.5 * regime_weight, 0.55, 1.0)))

    # Bayesian update.
    # Prior: regime_weight allocates mass to regime-matching scenarios.
    prior_raw = np.array([(1.0 - regime_weight) + regime_weight * per_scenario_factors[n]["regime_match"] for n in scenario_names], dtype=float)
    prior = prior_raw / max(1e-12, float(np.sum(prior_raw)))

    # Likelihood: softmax(loss_magnitude). Dominant risk = downside dominance.
    loss_mag = np.array([max(0.0, -float(adjusted_stress[n].get("pct", 0.0) or 0.0)) for n in scenario_names], dtype=float)
    # Upside scenarios get near-zero likelihood mass.
    loss_mag = np.where(loss_mag > 0.0, loss_mag, 0.0)
    likelihood = _softmax(loss_mag, temperature=1.0)
    likelihood = np.where(loss_mag > 0.0, likelihood, 1e-12)
    likelihood = likelihood / max(1e-12, float(np.sum(likelihood)))

    posterior_unnorm = prior * likelihood * p_model
    posterior = posterior_unnorm / max(1e-12, float(np.sum(posterior_unnorm)))

    entropy_raw = -float(np.sum(posterior * np.log(posterior + 1e-10)))
    entropy_norm = _normalized_entropy(posterior)
    threshold = ADAPTIVE_THRESHOLD_BASE + ADAPTIVE_THRESHOLD_SLOPE * entropy_norm
    k_entropy = 1 if entropy_norm < ENTROPY_TOPK_THRESHOLD else 2

    order = np.argsort(-posterior)
    p1 = float(posterior[int(order[0])]) if order.size else 0.0
    if p1 > TOP1_PROB_THRESHOLD:
        k = 1
        selection_mode = "top1_confident"
    elif p1 > TOP2_PROB_THRESHOLD:
        k = 2
        selection_mode = "top2_51_49_zone"
    else:
        k = 2
        selection_mode = "top2_uncertain"

    selected = order[:k].tolist()

    out: list[dict[str, Any]] = []
    for idx in selected:
        name = scenario_names[int(idx)]
        contrib = Fm[int(idx), :] * W
        top_idx = np.argsort(-contrib)[:3].tolist()
        drivers = [factor_cols[j] for j in top_idx if float(contrib[j]) > 0.0][:2]

        adj_pct = float(adjusted_stress.get(name, {}).get("pct", 0.0) or 0.0)
        raw_pct = float(adjusted_stress.get(name, {}).get("raw_pct", adj_pct) or adj_pct)

        rationale_tmpl = "Selected via posterior P≈{p:.0%} (mode={mode}); drivers={drivers}; adj={adj:.1f}%."
        rationale = rationale_tmpl.format(
            p=float(posterior[int(idx)]),
            mode=selection_mode,
            drivers=(" + ".join(drivers) if drivers else "n/a"),
            adj=adj_pct,
        )
        math_tmpl = (
            "Score = F·W + 0.3·diag(FIFᵀ); p=softmax(score). "
            "posterior ∝ prior·likelihood·p, with threshold={thr:.3f}, entropy={h:.3f} (norm={hn:.3f})."
        )
        math_just = math_tmpl.format(thr=threshold, h=entropy_raw, hn=entropy_norm)

        out.append({
            "scenario": name,
            "score": round(float(scores[int(idx)]), 6),
            "probability": round(float(posterior[int(idx)]), 4),
            "rationale": rationale,
            "mathematical_justification": math_just,
            "entropy": round(float(entropy_raw), 6),
            "entropy_norm": round(float(entropy_norm), 4),
            "threshold": round(float(threshold), 4),
            "entropy_top_k": int(k_entropy),
            "selection_k": int(k),
            "selection_mode": selection_mode,
            "top1_probability": round(float(p1), 4),
            "adjusted_pct": adj_pct,
            "raw_pct": raw_pct,
            "factors": per_scenario_factors.get(name, {}),
            "drivers": drivers,
            "prior": round(float(prior[int(idx)]), 4),
            "likelihood": round(float(likelihood[int(idx)]), 6),
        })

    return out


def _generate_dominant_narrative(
    dominant_views: list[dict[str, Any]],
    macro_ctx: dict[str, Any],
    cvar_pct: float,
    vol_annual: float,
) -> str:
    """
    Υπο-βήμα 3.4: Generation of Concise Dominant Narrative

    Structured template (όχι ad-hoc concatenation):
      - Regime context (macro_context["regime"])
      - Dominant scenario + adjusted magnitude + probability
      - Μαθηματική αιτιολόγηση (rate_pressure, regime_weight, etc.)
      - Quantitative anchors (CVaR, vol)

    Μήκος: 1-2 προτάσεις max. Ποτέ generic.
    """
    regime = _normalize_regime_label(str(macro_ctx.get("regime", "Unknown")))
    ctx_available = bool(macro_ctx.get("context_available", False))
    entropy_norm = float(dominant_views[0].get("entropy_norm")) if dominant_views else np.nan

    if not dominant_views:
        tmpl = (
            "Ιστορική ανάλυση: Annualised portfolio volatility {vol:.1f}%{cvar}. "
            "Δεν προκύπτει dominant scenario από τα stress δεδομένα."
        )
        cvar_part = ""
        if pd.notna(cvar_pct):
            cvar_part = ", CVaR (worst 5%) -{cvar:.1f}%".format(cvar=float(cvar_pct))
        if pd.notna(vol_annual):
            return tmpl.format(vol=float(vol_annual), cvar=cvar_part).strip()
        return "Ανεπαρκή δεδομένα για dominant risk view."

    regime_clause = "Βάσει ιστορικής προσομοίωσης"
    if ctx_available and regime != "Unknown":
        regime_clause = "Στο τρέχον **{regime}** regime".format(regime=regime)

    # Macro math rationale (exposed scalars)
    regime_weight = float(macro_ctx.get("regime_weight", 0.5))
    rate_pressure_score = float(macro_ctx.get("rate_pressure_score", 0.5))
    rate_pressure = float(macro_ctx.get("rate_pressure", 0.0))
    growth_amp = float(macro_ctx.get("growth_shock_amplifier", 1.0))
    inflation_adj = float(macro_ctx.get("inflation_hedge_adj", 0.0))
    third_force = float(macro_ctx.get("spy_macro_climate_score", 0.5))
    corr_risk = float(macro_ctx.get("correlation_risk", 0.0))
    conc_risk = float(macro_ctx.get("concentration_risk", 0.0))

    anchor_clause = ""
    if pd.notna(vol_annual) and pd.notna(cvar_pct):
        anchor_clause = " Anchors: vol {vol:.1f}%, CVaR -{cvar:.1f}%.".format(
            vol=float(vol_annual), cvar=float(cvar_pct)
        )
    elif pd.notna(vol_annual):
        anchor_clause = " Anchor: vol {vol:.1f}%.".format(vol=float(vol_annual))
    elif pd.notna(cvar_pct):
        anchor_clause = " Anchor: CVaR -{cvar:.1f}%.".format(cvar=float(cvar_pct))

    primary = dominant_views[0]
    p_name = str(primary.get("scenario", "Unknown"))
    p_prob = float(primary.get("probability", 0.0))
    p_pct = float(primary.get("adjusted_pct", 0.0))
    p_raw = float(primary.get("raw_pct", p_pct))
    mode = str(primary.get("selection_mode", ""))
    threshold = float(primary.get("threshold", np.nan))

    driver_clause = ""
    drivers = primary.get("drivers") or []
    if drivers:
        driver_clause = " Drivers: {d1}{d2}.".format(
            d1=str(drivers[0]),
            d2=(" + {d}".format(d=str(drivers[1])) if len(drivers) > 1 else ""),
        )

    adjust_clause = ""
    if abs(p_pct - p_raw) > 0.5:
        direction = "ενισχύθηκε" if (p_pct < p_raw) else "αποδυναμώθηκε"
        adjust_clause = " Macro-adjusted: {dir} από {raw:.1f}% → {adj:.1f}%.".format(
            dir=direction, raw=float(p_raw), adj=float(p_pct)
        )

    # Mathematical justification clause (user-friendly but explicit scalars).
    math_clause = (
        " Math: regime_weight={rw:.2f}, entropy={en:.2f}, threshold={thr:.2f}; "
        "rate_pressure_score={rps:.2f} (rate_pressure={rp:.3f}), growth_amp={ga:.2f}, inflation_adj={ia:.2f}; "
        "third_force={tf:.2f}, corr_risk={cr:.2f}, conc_risk={cc:.2f}."
    ).format(
        rw=regime_weight,
        en=(entropy_norm if pd.notna(entropy_norm) else float("nan")),
        thr=(threshold if pd.notna(threshold) else float("nan")),
        rps=rate_pressure_score,
        rp=rate_pressure,
        ga=growth_amp,
        ia=inflation_adj,
        tf=third_force,
        cr=corr_risk,
        cc=conc_risk,
    )

    s1_tmpl = "{regime}, dominant risk: **{scenario}** (P≈{p:.0%}) → projected impact {pct:.1f}%.{math}{drivers}{adj}{anchors}"
    s1 = s1_tmpl.format(
        regime=regime_clause,
        scenario=p_name,
        p=p_prob,
        pct=p_pct,
        math=" " + math_clause,
        drivers=(" " + driver_clause if driver_clause else ""),
        adj=(" " + adjust_clause if adjust_clause else ""),
        anchors=anchor_clause,
    ).strip()

    if len(dominant_views) >= 2:
        secondary = dominant_views[1]
        s_name = str(secondary.get("scenario", "Unknown"))
        s_prob = float(secondary.get("probability", 0.0))
        s_pct = float(secondary.get("adjusted_pct", 0.0))
        zone_clause = ""
        if mode == "top2_51_49_zone":
            zone_clause = " (51/49 zone)"
        s2_tmpl = "Secondary{zone}: **{scenario}** (P≈{p:.0%}) → {pct:.1f}%."
        s2 = s2_tmpl.format(zone=zone_clause, scenario=s_name, p=s_prob, pct=s_pct).strip()
        # When entropy is high, add an explicit uncertainty clause (still actionable).
        if pd.notna(entropy_norm) and entropy_norm > 0.60:
            u_tmpl = "Uncertainty elevated (entropy={en:.2f}) — maintain hedges for both dominant tails."
            u = u_tmpl.format(en=float(entropy_norm))
            return "{s1} {s2} {u}".format(s1=s1, s2=s2, u=u).strip()
        return "{s1} {s2}".format(s1=s1, s2=s2).strip()

    return s1.strip()


def _generate_dominant_risk_view(
    stress_results: dict[str, Any],
    macro_ctx: dict[str, Any],
    cvar_pct: float,
    vol_annual: float,
    corr_warnings: list,
) -> dict[str, Any]:
    """Orchestrates: adjust stress → probabilistic dominance → concise narrative."""
    adjusted_detail = _apply_regime_adjustments(stress_results, macro_ctx)
    adjusted_public = {
        k: {"pnl": v.get("pnl", 0.0), "pct": v.get("pct", 0.0), "description": v.get("description", "")}
        for k, v in adjusted_detail.items()
    }
    dominant_views = _select_dominant_views(adjusted_detail, macro_ctx)
    top_scenarios = [v["scenario"] for v in dominant_views]
    narrative = _generate_dominant_narrative(
        dominant_views=dominant_views,
        macro_ctx=macro_ctx,
        cvar_pct=cvar_pct,
        vol_annual=vol_annual,
    )

    return {
        "narrative":       narrative,
        "top_scenarios":   top_scenarios,
        "adjusted_stress": adjusted_detail,
        "stress_results":  adjusted_public,
        "dominant_views":  dominant_views,
        "regime_entropy":  float(dominant_views[0].get("entropy_norm")) if dominant_views else np.nan,
    }


def _quantify_macro_context(
    macro_report: dict[str, Any] | None,
    macro_exposure: dict[str, Any] | None,
) -> dict[str, Any]:
    """Quantifies macro_report + macro_exposure into bounded features/weights (Step 3.1)."""
    def _to_float(x: Any) -> float | None:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if np.isnan(v):
            return None
        return v

    # ── Default / neutral context (backward-compatible baseline) ────────────
    ctx: dict[str, Any] = {
        "regime":                 "Unknown",
        "regime_weight":          0.5,   # Neutral — δεν ξέρουμε, δεν αλλάζουμε τίποτα
        "rate_pressure":          0.0,   # Core feature (signed)
        "rate_pressure_score":    0.5,   # Neutral
        "inflation_hedge_adjustment": 0.0,  # Core feature (signed)
        "inflation_hedge_adj":    0.0,   # No adjustment
        "growth_sensitivity":     0.0,   # Core feature (signed)
        "growth_shock_amplifier": 1.0,   # No amplification
        "context_available":      False,
    }

    # ── Αν δεν υπάρχουν macro data → επιστρέφουμε neutral ───────────────────
    if macro_report is None and macro_exposure is None:
        return ctx

    ctx["context_available"] = True

    # ── 1. Regime label & weight ───────────────────────────────────────────
    # Mathematician-financial rationale:
    # Το regime_weight λειτουργεί ως "prior strength" (πόσο αποφασιστικό είναι το καθεστώς).
    regime = "Unknown"
    if macro_report:
        regime = str(macro_report.get("macro_regime") or "Unknown")
    ctx["regime"] = regime

    strength = None
    if macro_report:
        strength = _to_float(macro_report.get("regime_strength"))

    # Fallback mapping όταν δεν υπάρχει explicit strength.
    _REGIME_WEIGHTS: dict[str, float] = {
        "Disinflationary Growth":   0.80,
        "Reflationary Growth":      0.75,
        "Stagflationary Pressure":  0.85,  # Σπάνιο αλλά πολύ ισχυρό signal
        "Recessionary":             0.80,
        "Transitional":             0.55,
        "Unknown":                  0.50,
    }
    if strength is None:
        strength = _REGIME_WEIGHTS.get(regime, 0.50)
    ctx["regime_weight"] = float(np.clip(strength, 0.0, 1.0))

    # ── 2. Rate pressure feature + score ────────────────────────────────────
    # Βήμα 3.1:
    #   rate_pressure = yield_10y_slope * term_spread_accel (signed)
    # Οικονομική λογική: flattening acceleration υπό normal slope → tightening pressure.
    rate_pressure_feature = 0.0
    rate_pressure_score = 0.5  # neutral default
    if macro_report:
        term_spread_accel = _to_float(macro_report.get("term_spread_accel"))
        yield_10y_slope = _to_float(macro_report.get("yield_10y_slope"))

        if term_spread_accel is not None and yield_10y_slope is not None:
            rate_pressure_feature = float(yield_10y_slope * term_spread_accel)
            # Map signed → [0, 1] for multiplier use. Negative = more pressure (invert).
            scaled = float(np.tanh(rate_pressure_feature / 0.25))  # [-1, +1]
            rate_pressure_score = float(np.clip(0.5 - 0.5 * scaled, 0.0, 1.0))
        elif term_spread_accel is not None:
            # Fallback (backward-compatible): term_spread_accel μόνο του.
            tsa_clipped = float(np.clip(term_spread_accel, -0.5, 0.5))
            rate_pressure_score = float(np.clip(0.5 - tsa_clipped, 0.0, 1.0))
    ctx["rate_pressure"] = round(rate_pressure_feature, 6)
    ctx["rate_pressure_score"] = round(rate_pressure_score, 4)

    # inflation_hedge_adjustment = inflation_growth_interaction * inflation_hedge_score
    inflation_feature = 0.0
    inflation_adj = 0.0
    igi = None
    if macro_report:
        igi = _to_float(macro_report.get("inflation_growth_interaction"))
    inflation_hedge_score = 1.0
    if macro_exposure:
        inflation_hedge_score = _to_float(macro_exposure.get("inflation_hedge_score")) or 1.0
    if igi is not None:
        inflation_feature = float(igi * inflation_hedge_score)
        inflation_adj = float(np.tanh(inflation_feature))  # clamp to [-1, +1]
    ctx["inflation_hedge_adjustment"] = round(inflation_feature, 6)
    ctx["inflation_hedge_adj"] = round(inflation_adj, 4)

    # growth_shock_amplifier: scales growth-sensitive scenario impacts
    amplifier = 1.0
    if macro_exposure:
        growth_tilt = macro_exposure.get("growth_tilt", 0.0) or 0.0
        rate_sens   = macro_exposure.get("rate_sensitivity_proxy", 1.0) or 1.0
        amplifier = float(np.clip(1.0 + float(growth_tilt), 0.5, 2.0))
        if rate_sens > 1.2:
            amplifier = float(np.clip(amplifier * (rate_sens / 1.2), 0.5, 2.5))
    ctx["growth_shock_amplifier"] = round(amplifier, 4)

    # growth_sensitivity = rate_sensitivity_proxy * growth_proxy_accel
    growth_sensitivity_feature = 0.0
    growth_proxy_accel = None
    if macro_report:
        growth_proxy_accel = _to_float(macro_report.get("growth_proxy_accel"))
    rate_sens_feature = 1.0
    if macro_exposure:
        rate_sens_feature = _to_float(macro_exposure.get("rate_sensitivity_proxy")) or 1.0
    if growth_proxy_accel is not None:
        growth_sensitivity_feature = float(rate_sens_feature * growth_proxy_accel)
    ctx["growth_sensitivity"] = round(growth_sensitivity_feature, 6)

    return ctx


def _compute_rolling_correlation_alert(
    port_returns: pd.DataFrame,
    tickers: list[str],
) -> tuple[bool, dict]:
    """Rolling 30d correlation shift alert + current pair summaries."""
    if len(port_returns) < 60 or len(tickers) < 2:
        return False, {}

    df = port_returns[tickers].copy()
    if len(df) < 60:
        return False, {}

    # Vectorized rolling correlation for all pairs at once:
    # returns DataFrame with MultiIndex index (date, ticker_1) and columns ticker_2.
    rolling_corr = df.rolling(30, min_periods=30).corr()

    # Long Series with MultiIndex (date, ticker_1, ticker_2)
    corr_series = rolling_corr.stack()
    t1 = corr_series.index.get_level_values(1)
    t2 = corr_series.index.get_level_values(2)
    corr_upper = corr_series[t1 < t2].dropna()
    if corr_upper.empty:
        return False, {}

    daily_mean_corr = corr_upper.groupby(level=0).mean().dropna()
    if len(daily_mean_corr) < 30:
        return False, {}

    recent_mean = float(daily_mean_corr.iloc[-30:].mean())
    historical_mean = (
        float(daily_mean_corr.iloc[:-30].mean()) if len(daily_mean_corr) > 30 else recent_mean
    )
    alert = (recent_mean - historical_mean) > 0.15 and recent_mean > 0.70

    # Summary per pair: last rolling corr + historical average across the sample.
    pair_mean = corr_upper.groupby(level=[1, 2]).mean()
    last_date = daily_mean_corr.index[-1]
    last_mat = rolling_corr.xs(last_date, level=0).reindex(index=tickers, columns=tickers)
    vals = last_mat.values
    n = len(tickers)
    iu = np.triu_indices(n, k=1)
    t1_arr = np.asarray(tickers, dtype=object)[iu[0]]
    t2_arr = np.asarray(tickers, dtype=object)[iu[1]]
    curr_arr = vals[iu]

    summary: dict[str, dict[str, float]] = {}
    for a, b, curr in zip(t1_arr, t2_arr, curr_arr, strict=False):
        if pd.isna(curr):
            continue
        hist = pair_mean.get((a, b))
        if hist is None or pd.isna(hist):
            continue
        summary[f"{a}-{b}"] = {
            "current_30d": round(float(curr), 3),
            "historical_avg": round(float(hist), 3),
        }

    return alert, summary


def _generate_dynamic_hedges(macro_ctx, tech_exp, avg_beta, stress_results) -> list[str]:
    """Druckenmiller-style dynamic hedging logic based on regime and portfolio profile."""
    recs = []
    regime = macro_ctx.get("regime", "Unknown")
    
    # 1. Asymmetric Tail Protection
    worst_scenario = min(stress_results.items(), key=lambda x: x[1]["pct"]) if stress_results else (None, None)
    if worst_scenario[0]:
        name, data = worst_scenario
        if data["pct"] < -15:
            recs.append(f"🛡️ **Tail Risk Hedge:** Προστασία έναντι {name}. Σκεφτείτε Put Options on SPY/QQQ.")

    # 2. Regime-Aware Hedging
    if "Stagflation" in regime:
        recs.append("🛡️ **Inflation Hedge:** Overweight Gold & Energy (XLE). Μείωση duration στα ομόλογα.")
    elif "Growth Scare" in regime or avg_beta > 1.3:
        recs.append("🛡️ **Capital Preservation:** Αύξηση Cash/Money Market (BIL) και Long-Term Treasuries (TLT).")
    
    # 3. Sector-Specific
    if tech_exp > 45:
        recs.append("🛡️ **Tech Concentration:** Υψηλή έκθεση σε growth. Αντιστάθμιση μέσω Short QQQ ή rotation σε Value (VTV).")
    
    if not recs:
        recs.append("🛡️ **Standard Hedge:** Διατήρηση επιπέδων ρευστότητας και επιλεκτική τοποθέτηση σε uncorrelated assets.")
        
    return recs


def _run_stress_tests(
    total_val: float,
    valid_port: pd.DataFrame,
    tickers: list[str],
    sector_map: dict[str, str],
) -> dict[str, Any]:
    """Applies predefined scenario shocks (tech multiplier by sector)."""
    results = {}
    weights = valid_port.set_index("Ticker")["wt"].to_dict()

    for scenario_name, scenario in STRESS_SCENARIOS.items():
        market_shock = scenario["market_shock"]
        tech_mult = scenario["tech_multiplier"]
        weighted_shock = 0.0
        for ticker in tickers:
            w = weights.get(ticker, 0.0)
            sector = sector_map.get(ticker, "Unknown")
            multiplier = tech_mult if sector in TECH_SECTORS else 1.0
            weighted_shock += w * market_shock * multiplier

        pnl_dollar = weighted_shock * total_val
        results[scenario_name] = {
            "pnl": round(pnl_dollar, 2),
            "pct": round(weighted_shock * 100, 2),
            "description": scenario["description"]
        }

    return results

def simulate_dynamic_macro_shock(
    port_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    benchmark: str = "SPY",
    shock_pct: float = -10.0,
) -> dict[str, Any]:
    """Beta×correlation macro shock transmission (benchmark-dampened)."""
    results = {
        "scenario_shock": shock_pct,
        "benchmark": benchmark,
        "total_start_value": 0.0,
        "total_simulated_value": 0.0,
        "total_expected_pl": 0.0,
        "asset_impacts": {},
    }
    if port_df.empty or returns_df.empty or benchmark not in returns_df.columns:
        return results
    valid = port_df.dropna(subset=["Current Value"]).copy()
    if valid.empty:
        return results
    start_value = valid["Current Value"].sum()
    results["total_start_value"] = float(start_value)
    benchmark_returns = returns_df[benchmark]
    simulated_total = 0.0
    for _, row in valid.iterrows():
        t = row["Ticker"]
        val = row["Current Value"]
        if t in returns_df.columns:
            asset_rets = returns_df[t]
            var = benchmark_returns.var()
            cov = asset_rets.cov(benchmark_returns)
            beta = cov / var if var != 0 else 1.0
            corr = asset_rets.corr(benchmark_returns)
            if pd.isna(corr):
                corr = 0.0
            implied_asset_shock_pct = shock_pct * beta * abs(corr)  # Damping logic
            simulated_asset_val = val * (1 + implied_asset_shock_pct / 100.0)
        else:
            beta = np.nan
            corr = np.nan
            implied_asset_shock_pct = 0.0
            simulated_asset_val = val
        simulated_total += simulated_asset_val
        results["asset_impacts"][t] = {
            "beta": float(beta) if pd.notna(beta) else 0.0,
            "correlation": float(corr) if pd.notna(corr) else 0.0,
            "current_value": float(val),
            "simulated_value": float(simulated_asset_val),
            "implied_shock_pct": float(implied_asset_shock_pct),
            "pl_impact": float(simulated_asset_val - val),
        }
    results["total_simulated_value"] = float(simulated_total)
    results["total_expected_pl"] = float(simulated_total - start_value)
    return results


"""
Risk_metrics_engine.py — Layer B: Risk Domain Mathematical Transforms
======================================================================
Contains pure mathematical functions for risk-related metrics (Sharpe, Beta, Drawdown).
Stateless and independent of I/O.
"""

import pandas as pd
import numpy as np

def calculate_risk_metrics(stock_rets: pd.Series, bench_rets: pd.Series, rf_rate: float = 0.04) -> dict:
    """
    Layer B: Risk & Return Metrics (Sharpe, Drawdown, Beta).
    Pure math transformation.
    """
    out = {
        "beta": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan
    }
    if stock_rets.empty:
        return out
        
    # Beta
    if not bench_rets.empty and len(stock_rets) == len(bench_rets):
        cov = stock_rets.cov(bench_rets)
        var = bench_rets.var()
        out["beta"] = float(cov / var) if var != 0 else np.nan
        
    # Sharpe
    ret_mean = stock_rets.mean() * 252
    ret_vol = stock_rets.std() * np.sqrt(252)
    out["sharpe"] = float((ret_mean - rf_rate) / ret_vol) if ret_vol != 0 else np.nan
    
    # Drawdown
    cumulative = (1 + stock_rets).cumprod()
    peaks = cumulative.cummax()
    drawdown = (cumulative - peaks) / peaks
    out["max_drawdown"] = float(drawdown.min() * 100)
    
    return out

def calculate_cointegration(stock_rets: pd.Series, bench_rets: pd.Series) -> float:
    """
    Layer B: Co-integration test (ADF on residuals).
    Pure statistical math.
    """
    import statsmodels.api as sm
    try:
        y = (1 + stock_rets).cumprod().values
        X = sm.add_constant((1 + bench_rets).cumprod().values)
        res = sm.tsa.stattools.adfuller(sm.OLS(y, X).fit().resid)
        return float(res[1]) # p-value
    except Exception:
        return np.nan
