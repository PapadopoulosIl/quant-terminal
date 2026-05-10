"""
valuation_engine.py — Institutional Valuation Analytics Pipeline
=================================================================
Αυτόνομο "μαύρο κουτί" για αποτιμήσεις χαρτοφυλακίου.
Δέχεται port_df + meta_df και επιστρέφει institutional-grade valuation metrics.

Metrics που υπολογίζονται:
  - Harmonic Mean Forward P/E (Αρμονικός Μέσος — ο μαθηματικά σωστός τρόπος)  ← ΝΕΟ
  - Arithmetic Mean P/E (για σύγκριση / εκπαίδευση)
  - Equity Risk Premium (ERP): Earnings Yield vs Risk-Free Rate  ← ΝΕΟ
  - Weighted P/S Ratio
  - Benchmark comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from core_analysis import UnifiedAnalysisResult, MarketRegime, RiskLevel, SignalStrength

# ---------------------------------------------------------------------------
# Σταθερές
# ---------------------------------------------------------------------------
# Τρέχον 10-ετές US Treasury yield (risk-free rate).
# Σε production, θα φέρναμε αυτό live από FRED API ή yfinance (^TNX).
# Εδώ ορίζεται ως fallback constant — το UI το override-αρει αν φέρει live data.
DEFAULT_RISK_FREE_RATE = 0.043  # 4.3% (approximate 10Y UST)

# Threshold για ERP alert: αν το risk premium < αυτό το threshold,
# οι μετοχές δεν πληρώνουν αρκετό premium πάνω από τα risk-free bonds.
ERP_MIN_THRESHOLD = 0.015  # 1.5% minimum acceptable equity risk premium

# P/E Clip bounds — αφαιρεί outliers (αρνητικά P/E, εξωφρενικά P/E)
PE_CLIP_LOW = 1.0
PE_CLIP_HIGH = 150.0


def run_valuation_engine(
    port_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> dict[str, Any]:
    """
    Κεντρικό entry point του Valuation Engine.

    Parameters
    ----------
    port_df          : DataFrame με ["Ticker", "Current Value"]
    meta_df          : DataFrame με ["ticker", "forward_pe", "trailing_pe",
                       "price_to_sales", ...] από asset_metadata
    risk_free_rate   : Τρέχον 10Y risk-free rate (decimal, π.χ. 0.043 = 4.3%)

    Returns
    -------
    dict με keys:
      harmonic_forward_pe,       ← Αρμονικός Μέσος (mathematically correct)
      arithmetic_forward_pe,     ← Αριθμητικός Μέσος (για σύγκριση)
      pe_bias_pct,               ← % που ο arithmetic υπερεκτιμά το P/E
      spy_forward_pe,
      benchmark_name,
      earnings_yield_pct,        ← 1/P/E * 100
      risk_free_rate_pct,        ← risk_free_rate * 100
      equity_risk_premium_pct,   ← Earnings Yield - Risk Free Rate
      erp_alert,                 ← True αν το ERP είναι επικίνδυνα χαμηλό
      weighted_ps_ratio,
      coverage_pct,              ← % portfolio value με διαθέσιμο P/E
    """
    out: dict[str, Any] = {
        "harmonic_forward_pe": np.nan,
        "arithmetic_forward_pe": np.nan,
        "pe_bias_pct": np.nan,
        "spy_forward_pe": np.nan,
        "benchmark_name": "SPY",
        "earnings_yield_pct": np.nan,
        "risk_free_rate_pct": risk_free_rate * 100,
        "equity_risk_premium_pct": np.nan,
        "erp_alert": False,
        "weighted_ps_ratio": np.nan,
        "coverage_pct": 0.0,
        # ── ΝΕΟ: Portfolio-level Weighted P/E ────────────────────────────────
        "portfolio_weighted_forward_pe": np.nan,
        "portfolio_weighted_trailing_pe": np.nan,
        "portfolio_weights": {},   # {ticker: weight_pct}
    }

    if port_df.empty or meta_df.empty:
        return out

    meta_idx = meta_df.set_index("ticker")
    valid = port_df.dropna(subset=["Current Value"]).copy()
    total_val = valid["Current Value"].sum()
    if total_val <= 0:
        return out

    valid = valid.copy()
    valid["wt"] = valid["Current Value"] / total_val

    # ── Attach forward P/E from metadata ────────────────────────────────────
    def _get_pe(ticker: str) -> float:
        if ticker not in meta_idx.index:
            return np.nan
        raw = meta_idx.loc[ticker, "forward_pe"] if "forward_pe" in meta_idx.columns else np.nan
        return _safe_float(raw)

    valid["fwd_pe"] = valid["Ticker"].map(_get_pe)
    valid["fwd_pe"] = pd.to_numeric(valid["fwd_pe"], errors="coerce").clip(PE_CLIP_LOW, PE_CLIP_HIGH)

    # ── Trailing P/E (αν υπάρχει στο meta_df) ────────────────────────────────
    def _get_trailing_pe(ticker: str) -> float:
        if ticker not in meta_idx.index:
            return np.nan
        raw = meta_idx.loc[ticker, "trailing_pe"] if "trailing_pe" in meta_idx.columns else np.nan
        return _safe_float(raw)

    valid["trailing_pe"] = valid["Ticker"].map(_get_trailing_pe)
    valid["trailing_pe"] = pd.to_numeric(valid["trailing_pe"], errors="coerce").clip(PE_CLIP_LOW, PE_CLIP_HIGH)

    pe_valid = valid.dropna(subset=["fwd_pe"]).copy()

    # Coverage: ποιο % της αξίας του portfolio έχει P/E data
    coverage = pe_valid["wt"].sum() if not pe_valid.empty else 0.0
    out["coverage_pct"] = round(coverage * 100, 1)

    if pe_valid.empty:
        return out

    # Re-normalise weights for the covered subset
    w = pe_valid["wt"] / pe_valid["wt"].sum()
    pe_vals = pe_valid["fwd_pe"].values

    # ── 1. Arithmetic Mean P/E (παλιός/λάθος τρόπος — για σύγκριση) ─────────
    arithmetic_pe = float(np.sum(w.values * pe_vals))
    out["arithmetic_forward_pe"] = round(arithmetic_pe, 2)

    # ── 2. Harmonic Mean P/E (ο σωστός τρόπος για ratios) ───────────────────
    # Formula: 1 / Σ(w_i / PE_i)
    # Αντίστοιχο με: Σ(Value_i) / Σ(Value_i / PE_i) = Total Value / Total Earnings
    harmonic_pe = float(1.0 / np.sum(w.values / pe_vals))
    out["harmonic_forward_pe"] = round(harmonic_pe, 2)

    # Bias: πόσο ο arithmetic υπερεκτιμά το P/E (π.χ. φαίνεται 30% ακριβότερο)
    if harmonic_pe > 0:
        bias = (arithmetic_pe - harmonic_pe) / harmonic_pe * 100
        out["pe_bias_pct"] = round(bias, 1)
        out["portfolio_weighted_forward_pe"] = out["harmonic_forward_pe"]

    # Trailing P/E (Weighted Harmonic)
    trailing_valid = valid.dropna(subset=["trailing_pe"]).copy()
    if not trailing_valid.empty:
        w_trail = trailing_valid["wt"] / trailing_valid["wt"].sum()
        portfolio_harmonic_trail = float(1.0 / np.sum(w_trail.values / trailing_valid["trailing_pe"].values))
        out["portfolio_weighted_trailing_pe"] = round(portfolio_harmonic_trail, 2)

    # Αποθήκευση ποσοστών βάρους (για UI / reporting)
    out["portfolio_weights"] = {
        row["Ticker"]: round(row["wt"] * 100, 2)
        for _, row in valid.iterrows()
    }

    # ── 3. Earnings Yield & Equity Risk Premium ──────────────────────────────
    # Χρησιμοποιούμε τον Harmonic P/E (σωστός)
    if harmonic_pe > 0:
        earnings_yield = 1.0 / harmonic_pe
        erp = earnings_yield - risk_free_rate
        out["earnings_yield_pct"] = round(earnings_yield * 100, 2)
        out["equity_risk_premium_pct"] = round(erp * 100, 2)
        # Alert αν το risk premium είναι πολύ χαμηλό (δεν αξίζει να παίρνεις μετοχικό ρίσκο)
        out["erp_alert"] = erp < ERP_MIN_THRESHOLD

    # ── 4. Benchmark P/E (SPY / QQQ) ────────────────────────────────────────
    for bench in ["SPY", "QQQ"]:
        if bench in meta_idx.index:
            bench_pe = _safe_float(
                meta_idx.loc[bench, "forward_pe"] if "forward_pe" in meta_idx.columns else np.nan
            )
            if pd.notna(bench_pe) and bench_pe > 0:
                out["spy_forward_pe"] = round(bench_pe, 2)
                out["benchmark_name"] = bench
                break

    # ── 5. Weighted P/S Ratio ────────────────────────────────────────────────
    def _get_ps(ticker: str) -> float:
        if ticker not in meta_idx.index:
            return np.nan
        raw = meta_idx.loc[ticker, "price_to_sales"] if "price_to_sales" in meta_idx.columns else np.nan
        return _safe_float(raw)

    valid["ps"] = valid["Ticker"].map(_get_ps)
    ps_valid = valid.dropna(subset=["ps"])
    if not ps_valid.empty:
        w_ps = ps_valid["wt"] / ps_valid["wt"].sum()
        out["weighted_ps_ratio"] = round(float(np.sum(w_ps.values * ps_valid["ps"].values)), 2)

    # --- Mapping σε Unified Model ---
    erp = out.get("equity_risk_premium_pct", 0) / 100.0
    erp_alert = out.get("erp_alert", False)
    
    risk_level = RiskLevel.MEDIUM if erp_alert else RiskLevel.LOW
    
    # Simple Signal logic for Valuation
    if erp > (ERP_MIN_THRESHOLD * 2):
        overall_signal = SignalStrength.BUY
    elif erp < ERP_MIN_THRESHOLD:
        overall_signal = SignalStrength.SELL
    else:
        overall_signal = SignalStrength.NEUTRAL

    out["unified_analysis"] = UnifiedAnalysisResult(
        ticker="PORTFOLIO",
        market_regime=MarketRegime.NEUTRAL,
        risk_level=risk_level,
        overall_signal=overall_signal,
        confidence_score=round(out.get("coverage_pct", 0) / 100.0, 2),
        key_drivers=[
            f"ERP: {out.get('equity_risk_premium_pct'):.2f}%",
            f"Harmonic P/E: {out.get('harmonic_forward_pe'):.1f}x",
            f"Coverage: {out.get('coverage_pct'):.1f}%"
        ],
        timestamp=pd.Timestamp.now().isoformat()
    )

    return out


def _safe_float(v: Any) -> float:
    """Coerce to float; return NaN if impossible."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan

def calculate_dynamic_pricing_power(historical_financials: pd.DataFrame) -> int:
    """
    Υπολογίζει το Pricing Power (0-5) βάσει ιστορικής σταθερότητας και τάσης του Gross Margin.
    Το DataFrame πρέπει να περιέχει ετήσια ή τριμηνιαία 'gross_margin'.
    """
    if historical_financials is None or historical_financials.empty or 'gross_margin' not in historical_financials.columns:
        return 2 # Default/Average score

    margins = historical_financials['gross_margin'].dropna()
    if len(margins) < 3:
        return 2

    current_margin = margins.iloc[-1]
    mean_margin = margins.mean()
    std_margin = margins.std()

    score = 0
    
    # 1. Βάση: Έχει γενικά καλό περιθώριο; (> 40%)
    if current_margin > 0.40:
        score += 1

    # 2. Σταθερότητα: Συντελεστής Μεταβλητότητας (CV). Το margin δεν καταρρέει στις κρίσεις.
    cv = std_margin / mean_margin if mean_margin > 0 else 1.0
    if cv < 0.10: # Πολύ σταθερό ιστορικά
        score += 2

    # 3. Momentum: Το τρέχον margin είναι καλύτερο από τον ιστορικό μέσο όρο;
    if current_margin > mean_margin:
        score += 1

    # 4. Z-Score: Μήπως βρίσκεται σε ιστορικό υψηλό περιθωρίων;
    z_score = (current_margin - mean_margin) / std_margin if std_margin > 0 else 0
    if z_score > 1.0:
        score += 1

    return min(score, 5) # Επιστρέφει 0 (Αδύναμη) έως 5 (Απόλυτο Μονοπώλιο)


def _extract_statement_series(statement: pd.DataFrame, candidates: list[str]) -> pd.Series:
    if statement is None or not isinstance(statement, pd.DataFrame) or statement.empty:
        return pd.Series(dtype=float)
    for label in candidates:
        if label in statement.index:
            return pd.to_numeric(statement.loc[label], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _infer_treasury_cap(macro_report: dict | None) -> float:
    if not macro_report:
        return 0.025
    candidates = [
        macro_report.get("yield_10y_current"),
        macro_report.get("ten_year_yield"),
        macro_report.get("10y_treasury"),
    ]
    for val in candidates:
        rate = _safe_float(val)
        if pd.notna(rate):
            if rate > 1:
                rate /= 100.0
            return float(np.clip(rate, 0.0, 0.025))
    return 0.025


def _infer_macro_wacc_penalty(macro_report: dict | None) -> float:
    if not macro_report:
        return 0.0
    slope = _safe_float(
        macro_report.get("yield_10y_slope", macro_report.get("rate_of_change_10y", np.nan))
    )
    regime = str(
        macro_report.get("macro_regime", macro_report.get("regime", ""))
    ).lower()
    penalty = 0.0
    if pd.notna(slope) and slope > 0:
        penalty += 0.005 + min(0.01, abs(slope) / 100.0)
    if "stagflation" in regime:
        penalty += 0.0075
    elif "overheating" in regime or "inflation" in regime:
        penalty += 0.005
    return float(np.clip(penalty, 0.0, 0.015))


def calculate_disruptive_capex_efficiency(financials_df, wacc: float) -> dict[str, Any]:
    """
    Estimate cash-based ROIIC and derive non-linear FCF/EPS growth paths.
    Accepts either a statement DataFrame or a fundamentals package with
    `quarterly` / `annual` tables.
    """
    out = {
        "cash_roiic": np.nan,
        "projected_fcf_growth_rates": [0.02] * 10,
        "projected_eps_growth_rates": [0.02] * 10,
        "growth_capex_ttm": np.nan,
        "capex_spike": False,
        "capex_intensity_zscore": 0.0,
        "data_frequency": "unknown",
    }

    if isinstance(financials_df, dict):
        quarterly = financials_df.get("quarterly", pd.DataFrame())
        annual = financials_df.get("annual", pd.DataFrame())
    else:
        quarterly = financials_df if isinstance(financials_df, pd.DataFrame) else pd.DataFrame()
        annual = financials_df if isinstance(financials_df, pd.DataFrame) else pd.DataFrame()

    quarterly = quarterly if isinstance(quarterly, pd.DataFrame) else pd.DataFrame()
    annual = annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()

    # Use annual statements for ROIIC when available, because the current ingestion keeps only 4 periods.
    roiic_source = annual if not annual.empty else quarterly
    freq = "annual" if not annual.empty else "quarterly"
    out["data_frequency"] = freq

    ocf = _extract_statement_series(roiic_source, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _extract_statement_series(roiic_source, ["CapEx", "Capital Expenditure"])
    dna = _extract_statement_series(roiic_source, [
        "Depreciation & Amortization",
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
        "Depreciation",
    ])
    revenue = _extract_statement_series(quarterly if not quarterly.empty else roiic_source, ["Revenue", "Total Revenue", "Operating Revenue"])

    aligned = pd.concat(
        [ocf.rename("ocf"), capex.rename("capex"), dna.rename("dna")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) >= 2:
        aligned["growth_capex"] = (aligned["capex"].abs() - aligned["dna"].abs()).clip(lower=0.0)
        lag_periods = 3 if freq == "annual" else min(12, max(2, len(aligned) - 1))
        lag_periods = min(lag_periods, len(aligned) - 1)
        delta_ocf = float(aligned["ocf"].iloc[0] - aligned["ocf"].iloc[lag_periods])
        invested_growth_capex = float(aligned["growth_capex"].iloc[1:lag_periods + 1].sum())
        out["growth_capex_ttm"] = float(aligned["growth_capex"].iloc[: min(4, len(aligned))].sum())
        if invested_growth_capex > 0:
            out["cash_roiic"] = float(delta_ocf / invested_growth_capex)

    if len(revenue) >= 3 and len(capex) >= 3:
        capex_ratio = capex.abs().iloc[: len(revenue)] / revenue.replace(0, np.nan).iloc[: len(capex)]
        capex_ratio = capex_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(capex_ratio) >= 3:
            hist = capex_ratio.iloc[1:]
            hist_std = float(hist.std()) if len(hist) > 1 else 0.0
            if hist_std > 0:
                z = float((capex_ratio.iloc[0] - hist.mean()) / hist_std)
            else:
                z = 0.0
            out["capex_intensity_zscore"] = z
            out["capex_spike"] = bool(z > 2.0)

    terminal_rate = min(0.025, max(0.01, wacc * 0.30))
    roiic = out["cash_roiic"]
    excess_return = 0.0 if pd.isna(roiic) else roiic - wacc
    spike_boost = max(0.0, out["capex_intensity_zscore"] - 1.5) * 0.01

    if pd.isna(roiic):
        upper_fcf = 0.06
        upper_eps = 0.05
    else:
        quality_score = 1.0 / (1.0 + np.exp(-10.0 * excess_return))
        upper_fcf = float(np.clip(terminal_rate + 0.02 + 0.16 * quality_score + spike_boost, terminal_rate, 0.24))
        upper_eps = float(np.clip(terminal_rate + 0.01 + 0.10 * quality_score, terminal_rate, 0.18))

    fcf_rates = []
    eps_rates = []
    midpoint = 4.0 if excess_return >= 0 else 2.5
    steepness = 0.9 if out["capex_spike"] else 0.7
    for year in range(1, 11):
        logistic = terminal_rate + (upper_fcf - terminal_rate) / (1.0 + np.exp(-steepness * (year - midpoint)))
        eps_logistic = terminal_rate + (upper_eps - terminal_rate) / (1.0 + np.exp(-0.8 * (year - (midpoint + 0.5))))

        if out["capex_spike"]:
            if year <= 2:
                fcf_rate = logistic - (0.03 if year == 1 else 0.015)
                eps_rate = eps_logistic - (0.05 if year == 1 else 0.03)
            elif year <= 6:
                fcf_rate = logistic + 0.015
                eps_rate = eps_logistic
            else:
                fcf_rate = logistic
                eps_rate = eps_logistic
        else:
            fcf_rate = logistic
            eps_rate = eps_logistic

        if pd.notna(roiic) and roiic < wacc:
            fade = 0.85 ** (year - 1)
            fcf_rate = terminal_rate + (fcf_rate - terminal_rate) * fade
            eps_rate = terminal_rate + (eps_rate - terminal_rate) * fade

        fcf_rates.append(float(np.clip(fcf_rate, -0.08, 0.30)))
        eps_rates.append(float(np.clip(eps_rate, -0.10, 0.24)))

    out["projected_fcf_growth_rates"] = fcf_rates
    out["projected_eps_growth_rates"] = eps_rates
    return out


def calculate_grounded_dcf(
    current_fcf: float,
    implied_growth_rates: list[float],
    wacc: float,
    shares_out: float,
    macro_report: dict | None = None,
) -> dict[str, Any]:
    """
    Grounded 2-stage DCF with duration-risk adjustment and rate sensitivity.
    """
    out = {
        "intrinsic_value_per_share": np.nan,
        "pv_of_fcf": np.nan,
        "pv_of_terminal_value": np.nan,
        "terminal_value_dependency_pct": np.nan,
        "effective_wacc": np.nan,
        "terminal_growth_rate": np.nan,
        "sensitivity_matrix": pd.DataFrame(),
    }

    shares_out = _safe_float(shares_out)
    current_fcf = _safe_float(current_fcf)
    if pd.isna(shares_out) or shares_out <= 0 or pd.isna(current_fcf):
        return out

    macro_penalty = _infer_macro_wacc_penalty(macro_report)
    effective_wacc = float(max(0.05, wacc + macro_penalty))
    terminal_growth = float(min(0.025, _infer_treasury_cap(macro_report)))
    growth_rates = list(implied_growth_rates[:5]) if implied_growth_rates else [0.02] * 5
    while len(growth_rates) < 5:
        growth_rates.append(growth_rates[-1] if growth_rates else 0.02)

    def _run_scenario(discount_rate: float) -> tuple[float, float, float]:
        fcf = current_fcf
        pv_fcf = 0.0
        year5_fcf = current_fcf
        for year, g in enumerate(growth_rates, start=1):
            fcf = fcf * (1.0 + float(g))
            year5_fcf = fcf
            pv_fcf += fcf / ((1.0 + discount_rate) ** year)

        terminal_value = 0.0
        if year5_fcf > 0 and discount_rate > terminal_growth + 0.0025:
            terminal_value = (year5_fcf * (1.0 + terminal_growth)) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1.0 + discount_rate) ** 5)
        total_pv = pv_fcf + pv_terminal
        intrinsic = total_pv / shares_out if shares_out > 0 else np.nan
        return float(intrinsic), float(pv_fcf), float(pv_terminal)

    base_intrinsic, pv_fcf, pv_terminal = _run_scenario(effective_wacc)
    total_pv = pv_fcf + pv_terminal

    scenarios = [
        ("Rate Cut (-100bps)", max(0.04, effective_wacc - 0.01)),
        ("Base Case", effective_wacc),
        ("Rate Shock (+100bps)", effective_wacc + 0.01),
    ]
    sensitivity = []
    for label, rate in scenarios:
        intrinsic, pv_stage1, pv_tv = _run_scenario(rate)
        sensitivity.append({
            "Scenario": label,
            "WACC": rate,
            "Intrinsic Value": intrinsic,
            "PV FCF": pv_stage1,
            "PV Terminal": pv_tv,
        })

    out["intrinsic_value_per_share"] = base_intrinsic
    out["pv_of_fcf"] = pv_fcf
    out["pv_of_terminal_value"] = pv_terminal
    out["terminal_value_dependency_pct"] = float((pv_terminal / total_pv) * 100.0) if total_pv > 0 else np.nan
    out["effective_wacc"] = effective_wacc
    out["terminal_growth_rate"] = terminal_growth
    out["sensitivity_matrix"] = pd.DataFrame(sensitivity)
    return out


def analyze_capex_monetization_profile(fundamentals: dict[str, Any], macro_report: dict | None = None) -> dict[str, Any]:
    """
    High-level wrapper: converts current fundamentals into a grounded capex/DCF view.
    """
    snapshot = fundamentals.get("snapshot", {}) if isinstance(fundamentals, dict) else {}
    quarterly = fundamentals.get("quarterly", pd.DataFrame()) if isinstance(fundamentals, dict) else pd.DataFrame()
    annual = fundamentals.get("annual", pd.DataFrame()) if isinstance(fundamentals, dict) else pd.DataFrame()

    base_wacc = max(0.07, DEFAULT_RISK_FREE_RATE + 0.055)
    efficiency = calculate_disruptive_capex_efficiency(
        {"quarterly": quarterly, "annual": annual},
        wacc=base_wacc,
    )

    current_fcf = np.nan
    current_ocf = np.nan
    if isinstance(quarterly, pd.DataFrame) and not quarterly.empty:
        fcf_row = _extract_statement_series(quarterly, ["Free Cash Flow"])
        ocf_row = _extract_statement_series(quarterly, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex_row = _extract_statement_series(quarterly, ["CapEx", "Capital Expenditure"])
        if len(fcf_row) >= 4:
            current_fcf = float(fcf_row.iloc[:4].sum())
        if len(ocf_row) >= 4:
            current_ocf = float(ocf_row.iloc[:4].sum())
        if pd.isna(current_fcf) and len(ocf_row) >= 4 and len(capex_row) >= 4:
            current_fcf = float(ocf_row.iloc[:4].sum() - capex_row.iloc[:4].abs().sum())

    use_ocf_penalty = False
    if pd.isna(current_fcf) or current_fcf <= 0:
        current_fcf = current_ocf
        use_ocf_penalty = pd.notna(current_ocf)

    current_price = _safe_float(snapshot.get("current_price"))
    market_cap = _safe_float(snapshot.get("market_cap"))
    shares_out = _safe_float(snapshot.get("shares_outstanding"))
    if pd.isna(shares_out) and pd.notna(market_cap) and pd.notna(current_price) and current_price > 0:
        shares_out = market_cap / current_price

    dcf_wacc = base_wacc + (0.01 if use_ocf_penalty else 0.0)
    dcf = calculate_grounded_dcf(
        current_fcf=current_fcf,
        implied_growth_rates=efficiency.get("projected_fcf_growth_rates", []),
        wacc=dcf_wacc,
        shares_out=shares_out,
        macro_report=macro_report,
    )
    dcf["current_fcf_basis"] = current_fcf
    dcf["used_ocf_as_base"] = use_ocf_penalty
    return {"efficiency": efficiency, "dcf": dcf}

def generate_valuation_insights(
    val: dict[str, Any],
    total_value: float,
) -> list[str]:
    """
    Παράγει advisory insights βάσει Harmonic P/E και ERP.
    Επιστρέφει list[str] με markdown-formatted insights.
    """
    insights = []

    harmonic_pe = val.get("harmonic_forward_pe", np.nan)
    arithmetic_pe = val.get("arithmetic_forward_pe", np.nan)
    spy_pe = val.get("spy_forward_pe", np.nan)
    bench_name = val.get("benchmark_name", "SPY")
    erp = val.get("equity_risk_premium_pct", np.nan)
    rf_rate = val.get("risk_free_rate_pct", np.nan)
    ey = val.get("earnings_yield_pct", np.nan)
    erp_alert = val.get("erp_alert", False)
    bias = val.get("pe_bias_pct", np.nan)
    coverage = val.get("coverage_pct", 0.0)

    # ── A. Harmonic vs Arithmetic bias disclosure ────────────────────────────
    if pd.notna(harmonic_pe) and pd.notna(arithmetic_pe) and pd.notna(bias) and bias > 5:
        insights.append(
            f"**⚖️ Harmonic P/E Correction:** Ο σωστός Αρμονικός Forward P/E είναι **{harmonic_pe:.1f}x** "
            f"(έναντι {arithmetic_pe:.1f}x αριθμητικού). Ο απλός μέσος υπερεκτιμά το κόστος κατά **{bias:.1f}%** "
            f"— το χαρτοφυλάκιο είναι φθηνότερο από ό,τι δείχνουν τα raw averages."
        )

    # ── Portfolio-Level Valuation Insight ────────────────────────────────────
    portfolio_fwd = val.get("portfolio_weighted_forward_pe", np.nan)
    if pd.notna(portfolio_fwd):
        insights.append(
            f"**📈 Portfolio-Level Valuation:** Το συνολικό χαρτοφυλάκιο έχει "
            f"**Forward P/E {portfolio_fwd:.1f}x** (σταθμισμένος μέσος)."
        )

    # ── B. Valuation vs Benchmark ────────────────────────────────────────────
    if pd.notna(harmonic_pe) and pd.notna(spy_pe) and spy_pe > 0:
        diff = harmonic_pe - spy_pe
        if diff > spy_pe * 0.20:
            insights.append(
                f"**📊 Valuation Premium:** Το χαρτοφυλάκιο διαπραγματεύεται στο {harmonic_pe:.1f}x Forward P/E, "
                f"δηλ. **+{diff:.1f}x premium** έναντι {bench_name} ({spy_pe:.1f}x). "
                f"Συνιστάται tactical rebalancing ή αύξηση cash buffer."
            )
        elif diff < -spy_pe * 0.15:
            insights.append(
                f"**✅ Value Opportunity:** Forward P/E στο {harmonic_pe:.1f}x, "
                f"**discount {abs(diff):.1f}x** έναντι {bench_name} ({spy_pe:.1f}x). "
                f"Το χαρτοφυλάκιο φαίνεται relative undervalued."
            )
        else:
            insights.append(
                f"**🔵 Fair Valued:** Forward P/E {harmonic_pe:.1f}x, "
                f"ευθυγραμμισμένο με {bench_name} ({spy_pe:.1f}x). Spread: {diff:+.1f}x."
            )

    # ── C. Equity Risk Premium Alert ─────────────────────────────────────────
    if pd.notna(erp) and pd.notna(rf_rate) and pd.notna(ey):
        if erp_alert:
            insights.append(
                f"**🚨 Equity Risk Premium Alert:** Earnings Yield = **{ey:.2f}%** vs Risk-Free Rate "
                f"**{rf_rate:.2f}%** → ERP = **{erp:.2f}%**. "
                f"Εξαιρετικά χαμηλό risk premium — τα ομόλογα ανταγωνίζονται τις μετοχές. "
                f"Σκεφτείτε μείωση equity exposure ή rotation σε value sectors."
            )
        else:
            insights.append(
                f"**💚 ERP Healthy:** Earnings Yield {ey:.2f}% vs RF {rf_rate:.2f}% "
                f"→ Equity Risk Premium = **{erp:.2f}%**. Επαρκής αποζημίωση για το μετοχικό ρίσκο."
            )

    # ── D. Coverage warning ──────────────────────────────────────────────────
    if coverage < 70:
        insights.append(
            f"**⚠️ Incomplete Data:** P/E data διαθέσιμο για {coverage:.0f}% της αξίας χαρτοφυλακίου. "
            f"Τα multiples ενδέχεται να μην αντικατοπτρίζουν το σύνολο των θέσεων."
        )

    return insights
