"""
advisory_layer.py — Advanced Hedge-Fund Advisory Layer (Druckenmiller Ontology)
=================================================================
Responsible for:
  - Econometric, regime-aware, probabilistic insights
  - Non-binary thinking (mathematical nuance instead of hard thresholds)
  - Integration of Risk, Valuation, Earnings Pulse & Macro
  - Druckenmiller-style language: regime mastery, asymmetry, capital preservation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from core_analysis import UnifiedAnalysisResult, MarketRegime, RiskLevel, SignalStrength

# Assuming these helpers exist in a shared utils or similar, if not we define them or use placeholders
# Based on previous context, we had format_human_value and format_percent
try:
    from utils import format_human_value, format_percent
except ImportError:
    def format_human_value(v): return f"${v:,.0f}"
    def format_percent(v): return f"{v:.2f}%"

from Valuation_engine9 import generate_valuation_insights

# =============================================================================
# Druckenmiller-Inspired Econometric Thresholds (mathematical, not arbitrary)
# =============================================================================
ECONOMETRIC_THRESHOLDS = {
    "corr_high_risk_zscore": 1.8,      # Current corr > historical mean + 1.8σ
    "corr_moderate_risk_zscore": 1.2,
    "beta_extreme": 1.45,
    "tech_concentration_warning": 45.0,
    "concentration_high": 8,           # positions
}

def generate_advisory_insights(
    quant: dict,
    valuation: dict,
    total_val: float,
    macro_report: dict = None,
    macro_exposure: dict = None,
    earnings_notes: list = None,
    stress_results: dict = None,
    snapshot_data: dict = None
) -> list:
    """
    ΝΕΑ ΕΚΔΟΣΗ: Χρησιμοποιεί το UnifiedAnalysisResult από το Risk Engine.
    Δεν δημιουργεί δικό της regime/risk logic. Μόνο εμπλουτισμός με humanized text.
    """

    unified: UnifiedAnalysisResult = quant.get("unified_analysis")
    if not unified:
        # Fallback (για transition περίοδο)
        unified = UnifiedAnalysisResult(
            ticker="PORTFOLIO",
            market_regime=MarketRegime.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            overall_signal=SignalStrength.NEUTRAL,
            confidence_score=0.6,
            key_drivers=["Fallback mode - Risk engine not updated yet"]
        )

    # === Humanized Narrative (μόνο εδώ επιτρέπεται) ===
    regime_text = {
        MarketRegime.BULLISH: "Η αγορά δείχνει σταθερή ανοδική τάση.",
        MarketRegime.BEARISH: "Η αγορά βρίσκεται σε πτωτική φάση με αυξημένη προσοχή.",
        MarketRegime.NEUTRAL: "Η αγορά κινείται σε ουδέτερη ζώνη.",
        MarketRegime.HIGH_VOLATILITY: "Υψηλή μεταβλητότητα – απαιτείται προσοχή."
    }.get(unified.market_regime, "Αβέβαιη κατάσταση αγοράς.")

    risk_text = {
        RiskLevel.LOW: "Το συνολικό ρίσκο είναι χαμηλό.",
        RiskLevel.MEDIUM: "Το ρίσκο είναι σε μέτρια επίπεδα.",
        RiskLevel.HIGH: "Αυξημένο ρίσκο – συνιστάται προσοχή.",
        RiskLevel.CRITICAL: "Κρίσιμο επίπεδο ρίσκου – απαιτείται άμεση δράση."
    }.get(unified.risk_level, "")

    # Κύριο μήνυμα (humanized)
    main_message = f"**{regime_text}** {risk_text} Το συνολικό σήμα είναι **{unified.overall_signal.value}** με confidence **{unified.confidence_score:.0%}**."

    insights = [main_message]

    # Προσθήκη key drivers (από το unified)
    for driver in unified.key_drivers:
        insights.append(f"• {driver}")

    # Earnings notes (αν υπάρχουν) – μόνο εμπλουτισμός
    if earnings_notes:
        insights.append("📊 **Earnings Intelligence**")
        for note in earnings_notes[:3]:
            # Handle both list of dicts or list of notes
            if isinstance(note, dict):
                insights.append(f"• **{note.get('ticker')}**: {note.get('note')}")
            else:
                insights.append(f"• {note}")

    # Hedge recommendation (αν υπάρχει)
    if unified.hedge_recommendation:
        insights.append(f"🛡️ **Προτεινόμενη αντιστάθμιση**: {unified.hedge_recommendation}")

    # --- Νέα Οντολογία: Alpha Generation Matrix ---
    if snapshot_data and isinstance(snapshot_data, dict):
        index_data = {"spy_trailing_pe": valuation.get("spy_forward_pe", 20)} # Placeholder / proxy for index data
        insights.append("🧠 **Alpha Generation Matrix (Fundamental Analysis)**")
        for ticker, snap in snapshot_data.items():
            if not snap or ticker == "SPY" or ticker == "QQQ":
                continue
            
            # Extract features for Alpha Signal
            high = float(snap.get("52WeekHigh") or snap.get("target_high") or np.nan)
            low = float(snap.get("52WeekLow") or snap.get("target_low") or np.nan)
            price = float(snap.get("previousClose") or snap.get("currentPrice") or np.nan)
            
            if pd.notna(high) and pd.notna(low) and pd.notna(price) and high > low:
                price_to_52w_high = (price - low) / (high - low)
            else:
                price_to_52w_high = 0.5
                
            forward_pe = float(snap.get("forwardPE", 20))
            avg_pe_5y = float(snap.get("trailingPE", 20)) # fallback proxy to trailing PE if 5y is missing
            
            # Check if earnings are available in snapshot or financials
            eps_forward = float(snap.get("epsForward", 0))
            eps_trailing = float(snap.get("epsTrailingTwelveMonths", 0))
            if eps_trailing > 0:
                eps_growth = (eps_forward - eps_trailing) / eps_trailing
            else:
                eps_growth = 0.05 # Neutral default
                
            ticker_data = {
                "price_percentile_52w": price_to_52w_high,
                "forward_pe": forward_pe,
                "trailing_pe_5y_avg": avg_pe_5y,
                "forward_eps_growth": eps_growth
            }
            
            signal = generate_alpha_signal(ticker_data, index_data)
            if not signal.startswith("NEUTRAL"):
                insights.append(f"• **{ticker}**: {signal}")

    return insights

def generate_alpha_signal(ticker_data: dict, index_data: dict) -> str:
    # Ανάκτηση των δεδομένων
    price_to_52w_high = ticker_data.get("price_percentile_52w", 0.5) # 0.0 to 1.0
    forward_pe = ticker_data.get("forward_pe", 20)
    avg_pe_5y = ticker_data.get("trailing_pe_5y_avg", 20) # Το 5Y Average
    eps_growth = ticker_data.get("forward_eps_growth", 0) # Ποσοστό
    
    spy_pe = index_data.get("spy_trailing_pe", 20)
    
    # 1. Υπολογισμός Valuation Premium/Discount
    pe_discount = (avg_pe_5y - forward_pe) / avg_pe_5y if avg_pe_5y > 0 else 0
    
    # --- LOGIC REGIMES ---
    
    # Regime 1: Deep Value / Alpha Inception
    if price_to_52w_high < 0.30 and pe_discount > 0.15 and eps_growth > 0.10:
        return "ALPHA INCEPTION (Strong Buy): Παρά την πτώση της τιμής, η μετοχή τελεί υπό ακραία υποτίμηση σε σχέση με τον ιστορικό μέσο όρο (P/E discount). Τα forward earnings αυξάνονται. Ιστορική ευκαιρία αγοράς."
        
    # Regime 2: Fundamental Momentum
    elif price_to_52w_high > 0.80 and forward_pe <= (avg_pe_5y * 1.1) and eps_growth > 0.15:
        return "FUNDAMENTAL MOMENTUM (Hold/Add): Η μετοχή βρίσκεται σε υψηλά 52 εβδομάδων, αλλά η άνοδος υποστηρίζεται απόλυτα από τα μελλοντικά κέρδη. Το P/E παραμένει εντός ιστορικών ορίων."
        
    # Regime 3: Multiple Exhaustion (Φούσκα)
    elif price_to_52w_high > 0.80 and forward_pe > (avg_pe_5y * 1.4) and spy_pe > 22:
        return "MULTIPLE EXHAUSTION (Trim/Derisk): Η τιμή έχει αποσυνδεθεί από τα θεμελιώδη. Διαπραγματεύεται ακριβά πάνω από το ιστορικό της P/E, ενώ και η αγορά είναι ακριβή. Εξέτασε άμεση μείωση ρίσκου."
        
    # Default / Neutral
    else:
        return "NEUTRAL: Μεικτή εικόνα μεταξύ αποτίμησης και δυναμικής κερδών. Διατήρηση τρέχουσας στρατηγικής."

def generate_dynamic_alerts(risk_df: pd.DataFrame) -> list:
    """
    Δημιουργεί στοχευμένα (precise) alerts βάσει Marginal Contribution to Risk (MCR)
    και Market Context (52w Position).
    """
    alerts = []
    if risk_df is None or risk_df.empty:
        return alerts
        
    for _, row in risk_df.iterrows():
        ticker = row.get("Ticker", "Unknown")
        weight = row.get("Weight (%)", 0)
        rc_pct = row.get("Risk Contribution (%)", 0)
        pos_52w = row.get("52w Position (%)", np.nan)
        
        # Αν η μετοχή καταναλώνει υπερβολικό ρίσκο αναλογικά με το κεφάλαιό της
        if rc_pct > weight * 1.5 and weight > 0:
            if pd.notna(pos_52w):
                if pos_52w > 80:
                    alerts.append(f"🟢 **Healthy Momentum Expansion:** Η **{ticker}** ευθύνεται για το {rc_pct:.1f}% του συνολικού ρίσκου (βάρος {weight:.1f}%), αλλά βρίσκεται κοντά σε 52-week High. Το volatility τροφοδοτείται από ισχυρό ανοδικό breakout.")
                elif pos_52w < 25:
                    alerts.append(f"🔴 **Toxic Volatility / Capitulation:** Η **{ticker}** απορροφά δυσανάλογα μεγάλο ρίσκο ({rc_pct:.1f}% έναντι {weight:.1f}% βάρους) ενώ καταρρέει προς το 52-week Low. Εξετάστε άμεσα τη μείωση θέσης.")
                else:
                    alerts.append(f"⚠️ **Disproportionate Risk:** Η **{ticker}** συνεισφέρει το {rc_pct:.1f}% της μεταβλητότητας με μόλις {weight:.1f}% βάρος. Οριακή αποδοτικότητα ρίσκου.")
            else:
                alerts.append(f"⚠️ **Disproportionate Risk:** Η **{ticker}** συνεισφέρει το {rc_pct:.1f}% της μεταβλητότητας με μόλις {weight:.1f}% βάρος.")
                
    return alerts

def build_risk_advisory(quant: dict) -> list[str]:
    """Nuanced, mathematically-driven risk advisory (no black/white)."""
    insights = []

    # Nuanced Correlation Risk (z-score based)
    high_corr = quant.get("correlation_warnings", [])
    if high_corr:
        # Αντί για σκληρό threshold 0.8, χρησιμοποιούμε context
        top_pairs = [f"{t1}-{t2} ({corr:.2f})" for t1, t2, corr in high_corr[:2]]
        insights.append(
            f"**🔗 Correlation Risk**: {', '.join(top_pairs)}. "
            "Η συσχέτιση είναι υψηλή — σε bear regime μπορεί να οδηγήσει σε synchronized drawdown."
        )

    # Beta & Concentration (regime-aware)
    # (μπορείς να επεκτείνεις με περισσότερα metrics από quant)

    return insights


def generate_stress_insights(stress_results: dict, total_value: float) -> list[str]:
    """Hedge-fund style stress insights with hedge recommendations."""
    insights = []
    
    # Check if stress_results is the new structured dict or the old one
    scenarios = stress_results.get("scenarios", {})
    if not scenarios and isinstance(stress_results, dict):
        # Fallback for old structure if necessary, though we aim for the new one
        scenarios = {k: v for k, v in stress_results.items() if isinstance(v, dict) and "pct" in v}

    diagnostics = stress_results.get("portfolio_diagnostics", {})
    recs = stress_results.get("hedge_recommendations", [])

    tech_exp = diagnostics.get("tech_growth_exposure_pct", 0)
    avg_beta = diagnostics.get("avg_beta_to_spy", 1.0)

    if tech_exp > 45:
        insights.append(f"**Tech-Heavy Exposure** ({tech_exp:.0f}%): Υψηλή ευαισθησία σε growth rotation.")
    if avg_beta > 1.4:
        insights.append(f"**High Beta** (avg {avg_beta:.2f}): Το portfolio κινείται επιθετικά — προσοχή σε bear regimes.")

    # Hedge Recommendations (Druckenmiller style)
    if recs:
        insights.append("🛡️ **Strategic Hedge Recommendations**")
        for rec in recs[:3]:
            insights.append(rec)

    return insights


# =============================================================================
# Υπάρχουσες συναρτήσεις (κρατάμε + μικρές βελτιώσεις)
# =============================================================================
def build_macro_economic_narrative(macro_report: dict) -> list[str]:
    """
    Παράγει modular list από επαγγελματικά paragraphs (όχι στεγνά μαθηματικά).
    Druckenmiller-style: Regime mastery and probabilistic outlook.
    """
    if not macro_report:
        return ["Το Macro Data Layer είναι ανενεργό. Δεν υφίστανται ισχυρά macro-regime signals (Neutral State)."]
        
    paragraphs = []
    
    # 1. Market Trend Signal (SMA 200)
    market_regime = macro_report.get("market_regime", "Unknown")
    if market_regime != "Unknown":
        regime_label = "BULLISH (Above 200-SMA)" if "ABOVE" in market_regime else "BEARISH (Below 200-SMA)"
        paragraphs.append(f"📉 **Market Trend:** Ο δείκτης SPY βρίσκεται σε **{regime_label}** καθεστώς.")

    reg_name = macro_report.get("macro_regime", "Unknown")
    reg_str  = macro_report.get("regime_strength", 0.5)
    
    para1 = f"📍 **Macro Regime: {reg_name}** (ισχύς {reg_str:.2f}/1.00)"
    paragraphs.append(para1)
    
    return paragraphs

def build_macro_regime_overlay(macro_exposure: dict, macro_report: dict) -> list[str]:
    """
    Αφαιρεί τη μαθηματική γλώσσα από τον τελικό χρήστη. 
    Παράγει απλές, κατανοητές, ανθρώπινες δομές για τους Financial Advisors.
    """
    if not macro_exposure or not macro_report:
        return []
        
    if macro_report.get("data_quality", "full") == "minimal":
        return ["**Macro Overlay**: Macro data unavailable or incomplete. Assuming a neutral regime for portfolio mapping."]
        
    rate_proxy = macro_exposure.get("rate_sensitivity_proxy", 0.0)
    hedge_score = macro_exposure.get("inflation_hedge_score", 0.0)
    
    # ── Bullet 1: Το Alignment του Portfolio ────────────────────────────────
    if rate_proxy > 1.15:
        rate_text = "υψηλότερη από τον μέσο όρο ευαισθησία στις μεταβολές των επιτοκίων (growth-tilt)"
    elif rate_proxy < 0.85:
        rate_text = "επιφυλακτική/συντηρητική ευαισθησία στα επιτόκια"
    else:
        rate_text = "ισορροπημένη ευαισθησία στις μεταβολές επιτοκίων"
        
    if hedge_score > 30.0:
        hedge_text = f"έχει σημαντική οργανική προστασία έναντι πληθωρισμού ({hedge_score:.0f}%)"
    else:
        hedge_text = f"διατηρεί χαμηλή προστασία έναντι πληθωρισμού ({hedge_score:.0f}%)"
        
    bullet1 = f"🎯 **Portfolio-Macro Alignment:** Το χαρτοφυλάκιο {hedge_text} και χαρακτηρίζεται από {rate_text}."
    
    return [bullet1]


# ---------------------------------------------------------------------------
# Compatibility Wrappers for app1.py
# ---------------------------------------------------------------------------

def build_live_briefing_text(analysis: dict) -> str:
    """
    Compatibility wrapper for app1.py. 
    Synthesizes the new Druckenmiller-style insights into a single briefing string.
    """
    insights = generate_advisory_insights(
        quant=analysis.get("quant", {}),
        valuation=analysis.get("valuation", {}),
        total_val=analysis.get("total_val", 0.0),
        macro_report=analysis.get("macro_report"),
        macro_exposure=analysis.get("macro_exposure"),
        earnings_notes=analysis.get("earnings_notes", []),
        stress_results=analysis.get("quant", {}).get("stress_results")
    )
    return "\n\n".join(insights) if insights else "Δεν υπάρχουν διαθέσιμα insights για την τρέχουσα κατάσταση."


def build_benchmark_sensitivity_text(analysis: dict) -> str:
    """
    Compatibility wrapper for app1.py.
    Provides a concise note on benchmark sensitivity/beta.
    """
    quant = analysis.get("quant", {})
    beta = quant.get("beta", 1.0)
    bench = analysis.get("valuation", {}).get("benchmark_name", "SPY")
    
    if pd.isna(beta):
        return "Στοιχεία ευαισθησίας benchmark μη διαθέσιμα."
        
    if beta > 1.3:
        return f"Υψηλή ευαισθησία στο {bench} (Beta: {beta:.2f}). Το χαρτοφυλάκιο αναμένεται να υπεραντιδράσει σε κινήσεις της αγοράς."
    elif beta < 0.7:
        return f"Αμυντική τοποθέτηση έναντι του {bench} (Beta: {beta:.2f}). Περιορισμένη έκθεση σε συστημικό κίνδυνο."
    else:
        return f"Ισορροπημένη συσχέτιση με το {bench} (Beta: {beta:.2f})."


# ---------------------------------------------------------------------------
# Screener / Single-Ticker Analysis Helpers (for Orchestrator2.build_analysis)
# ---------------------------------------------------------------------------

def build_valuation_summary(ticker, benchmark, fundamentals, benchmark_funds, spy_funds, macro_report=None) -> list:
    """
    Builds a simplified valuation rail with only the highest-signal metrics.
    """
    s_snap = fundamentals.get("snapshot", {})
    b_snap = benchmark_funds.get("snapshot", {})
    spy_snap = spy_funds.get("snapshot", {})

    def get_val(snap, key):
        return snap.get(key, np.nan)

    def fmt_multiple(v):
        return f"{v:.1f}x" if pd.notna(v) else "N/A"

    def fmt_spread(v1, v2):
        if pd.isna(v1) or pd.isna(v2):
            return "N/A"
        return f"{v1 - v2:+.1f}x"

    cards = []

    s_tpe = get_val(s_snap, "trailing_pe")
    b_tpe = get_val(b_snap, "trailing_pe")
    spy_tpe = get_val(spy_snap, "trailing_pe")
    cards.append({
        "title": "Trailing P/E",
        "value": fmt_multiple(s_tpe),
        "subtitle": f"vs {benchmark}: {fmt_spread(s_tpe, b_tpe)}",
        "delta": f"SPY: {fmt_multiple(spy_tpe)}",
        "metric_key": "trailing_pe",
        "symbol": ticker,
    })

    s_fpe = get_val(s_snap, "forward_pe")
    b_fpe = get_val(b_snap, "forward_pe")
    spy_fpe = get_val(spy_snap, "forward_pe")
    cards.append({
        "title": "Forward P/E",
        "value": fmt_multiple(s_fpe),
        "subtitle": f"vs {benchmark}: {fmt_spread(s_fpe, b_fpe)}",
        "delta": f"SPY: {fmt_multiple(spy_fpe)}",
        "metric_key": "forward_pe",
        "symbol": ticker,
    })

    s_ps = get_val(s_snap, "ps_ratio")
    b_ps = get_val(b_snap, "ps_ratio")
    spy_ps = get_val(spy_snap, "ps_ratio")
    cards.append({
        "title": "Price/Sales",
        "value": fmt_multiple(s_ps),
        "subtitle": f"vs {benchmark}: {fmt_spread(s_ps, b_ps)}",
        "delta": f"SPY: {fmt_multiple(spy_ps)}",
        "metric_key": "ps_ratio",
        "symbol": ticker,
    })

    s_margin = get_val(s_snap, "profit_margin")
    cards.append({
        "title": "Profit Margin",
        "value": f"{s_margin:.2f}%" if pd.notna(s_margin) else "N/A",
        "subtitle": f"Revenue: {format_human_value(get_val(s_snap, 'revenue'))}",
        "delta": f"Net Income: {format_human_value(get_val(s_snap, 'net_income'))}",
        "metric_key": "profit_margin",
        "symbol": ticker,
    })

    return cards


def build_regime_ontology(ticker: str, benchmark: str, metrics: dict) -> dict[str, str]:
    """
    Maps raw market statistics to a structured analyst-facing ontology in Greek.
    The goal is to translate sigma-language into portfolio context and risk meaning.
    """
    z_score = float(metrics.get("z_score", np.nan)) if pd.notna(metrics.get("z_score", np.nan)) else np.nan
    coint_pval = float(metrics.get("coint_pval", np.nan)) if pd.notna(metrics.get("coint_pval", np.nan)) else np.nan
    beta = float(metrics.get("beta", np.nan)) if pd.notna(metrics.get("beta", np.nan)) else np.nan
    price_vs_sma50_pct = float(metrics.get("price_vs_sma50_pct", np.nan)) if pd.notna(metrics.get("price_vs_sma50_pct", np.nan)) else np.nan
    realized_vol_20 = float(metrics.get("realized_vol_20", np.nan)) if pd.notna(metrics.get("realized_vol_20", np.nan)) else np.nan
    volume_ratio = float(metrics.get("volume_ratio", np.nan)) if pd.notna(metrics.get("volume_ratio", np.nan)) else np.nan
    ret_1m = float(metrics.get("ret_1m", np.nan)) if pd.notna(metrics.get("ret_1m", np.nan)) else np.nan
    dist_high = float(metrics.get("dist_high", np.nan)) if pd.notna(metrics.get("dist_high", np.nan)) else np.nan
    dist_low = float(metrics.get("dist_low", np.nan)) if pd.notna(metrics.get("dist_low", np.nan)) else np.nan
    trend_score = float(metrics.get("trend_score", np.nan)) if pd.notna(metrics.get("trend_score", np.nan)) else np.nan

    if pd.isna(z_score):
        regime_label = "Ασαφές καθεστώς"
        deviation_text = "Δεν υπάρχουν επαρκή στοιχεία για να μετρηθεί με αξιοπιστία η απόσταση της τιμής από τον μέσο όρο."
    elif z_score >= 1.5:
        regime_label = "Επιθετική ανοδική επέκταση"
        deviation_text = (
            f"Η τιμή βρίσκεται {price_vs_sma50_pct:+.1f}% πάνω από τον 50ήμερο μέσο όρο, δηλαδή {z_score:.1f} τυπικές αποκλίσεις πάνω από το συνηθισμένο εύρος. "
            "Αυτό σημαίνει ότι η αγορά έχει τρέξει αισθητά ταχύτερα από τον πρόσφατο ρυθμό ισορροπίας και αυξάνει ο κίνδυνος βραχυπρόθεσμης αποσυμπίεσης."
        )
    elif z_score >= 0.5:
        regime_label = "Υγιής ανοδική κλίση"
        deviation_text = (
            f"Η τιμή βρίσκεται {price_vs_sma50_pct:+.1f}% πάνω από τον 50ήμερο μέσο όρο, δηλαδή {z_score:.1f} τυπικές αποκλίσεις πάνω από το κανονικό εύρος. "
            "Για analyst χρήση αυτό διαβάζεται ως ανοδική τάση χωρίς ακόμη ακραία υπερέκταση."
        )
    elif z_score <= -1.5:
        regime_label = "Βίαιη καθοδική απομάκρυνση"
        deviation_text = (
            f"Η τιμή βρίσκεται {price_vs_sma50_pct:+.1f}% κάτω από τον 50ήμερο μέσο όρο, δηλαδή {abs(z_score):.1f} τυπικές αποκλίσεις χαμηλότερα από το σύνηθες εύρος. "
            "Η απομάκρυνση είναι ακραία και απαιτεί διάκριση μεταξύ panic move και πραγματικής αλλαγής θεμελιώδους καθεστώτος."
        )
    elif z_score <= -0.5:
        regime_label = "Ελεγχόμενη καθοδική πίεση"
        deviation_text = (
            f"Η τιμή βρίσκεται {price_vs_sma50_pct:+.1f}% κάτω από τον 50ήμερο μέσο όρο, δηλαδή {abs(z_score):.1f} τυπικές αποκλίσεις χαμηλότερα από το κανονικό εύρος. "
            "Η τάση είναι αρνητική, αλλά όχι ακόμη αποδιοργανωμένη."
        )
    else:
        regime_label = "Ουδέτερη ισορροπία"
        deviation_text = (
            f"Η τιμή απέχει {price_vs_sma50_pct:+.1f}% από τον 50ήμερο μέσο όρο και κινείται μόλις {z_score:+.1f} τυπικές αποκλίσεις από το κέντρο της πρόσφατης κατανομής. "
            "Άρα η αγορά βρίσκεται κοντά στη ζώνη ισορροπίας και δεν δίνει ισχυρό directional edge από μόνη της."
        )

    if pd.isna(coint_pval):
        benchmark_text = f"Δεν υπάρχουν επαρκή στατιστικά στοιχεία για ασφαλές συμπέρασμα ως προς τη σχέση με το {benchmark}."
        signal_strength = "Απροσδιόριστη"
    elif coint_pval <= 0.05:
        benchmark_text = (
            f"Η συνολοκλήρωση με το {benchmark} παραμένει ισχυρή (p-value {coint_pval:.3f}). "
            "Πρακτικά αυτό σημαίνει ότι η σχετική απόκλιση δεν πρέπει να αντιμετωπίζεται μόνο ως νέα τάση, αλλά και ως πιθανή ευκαιρία επαναφοράς προς τη σχετική ισορροπία."
        )
        signal_strength = "Υψηλή στατιστική πειθαρχία"
    elif coint_pval <= 0.12:
        benchmark_text = (
            f"Η σχέση με το {benchmark} είναι υπαρκτή αλλά όχι απόλυτα σταθερή (p-value {coint_pval:.3f}). "
            "Ο analyst πρέπει να αντιμετωπίσει το benchmark περισσότερο ως σημείο αναφοράς παρά ως αυστηρό μηχανισμό mean reversion."
        )
        signal_strength = "Μέτρια στατιστική πειθαρχία"
    else:
        benchmark_text = (
            f"Η στατιστική σύνδεση με το {benchmark} είναι αδύναμη (p-value {coint_pval:.3f}). "
            "Άρα η συμπεριφορά του asset φαίνεται πιο idiosyncratic και κάθε valuation ή risk view χρειάζεται μεγαλύτερο βάρος στα δικά του micro drivers."
        )
        signal_strength = "Χαμηλή στατιστική πειθαρχία"

    if pd.isna(beta):
        beta_text = "Το Beta δεν είναι διαθέσιμο, άρα το systemic transmission risk δεν μπορεί να ποσοτικοποιηθεί με συνέπεια."
    elif beta >= 1.35:
        beta_text = (
            f"Το Beta στο {beta:.2f} δηλώνει επιθετική ευαισθησία στον κύκλο του {benchmark}. "
            "Σε πρακτικούς όρους, μια κίνηση της αγοράς μπορεί να πολλαπλασιαστεί στο συγκεκριμένο asset και να ενισχύσει τόσο το upside όσο και το drawdown risk."
        )
    elif beta >= 0.85:
        beta_text = (
            f"Το Beta στο {beta:.2f} δείχνει αρκετά κανονική συστημική έκθεση. "
            "Η συμπεριφορά του asset παραμένει market-linked χωρίς να είναι υπερβολικά μοχλευμένη στον κύκλο."
        )
    else:
        beta_text = (
            f"Το Beta στο {beta:.2f} υποδηλώνει αμυντικότερη συμπεριφορά από το {benchmark}. "
            "Η θέση μπορεί να λειτουργήσει πιο σταθεροποιητικά μέσα στο portfolio, αλλά με χαμηλότερη συμμετοχή σε momentum rallies."
        )

    if pd.isna(realized_vol_20):
        vol_text = "Η βραχυπρόθεσμη πραγματοποιημένη μεταβλητότητα δεν είναι διαθέσιμη."
    elif realized_vol_20 >= 45:
        vol_text = f"Η 20ήμερη πραγματοποιημένη μεταβλητότητα βρίσκεται στο {realized_vol_20:.1f}%, επίπεδο υψηλής αστάθειας που απαιτεί μικρότερη ανοχή θέσης και αυστηρότερο sizing."
    elif realized_vol_20 >= 28:
        vol_text = f"Η 20ήμερη πραγματοποιημένη μεταβλητότητα είναι {realized_vol_20:.1f}%, δηλαδή αυξημένη αλλά ακόμη διαχειρίσιμη για growth / high-beta profile."
    else:
        vol_text = f"Η 20ήμερη πραγματοποιημένη μεταβλητότητα στο {realized_vol_20:.1f}% παραμένει σε σχετικά ελεγχόμενο εύρος."

    if pd.isna(volume_ratio):
        flow_text = "Δεν υπάρχουν επαρκή στοιχεία όγκου για ασφαλές συμπέρασμα σχετικά με τη συμμετοχή της αγοράς."
    elif volume_ratio > 1.5:
        flow_text = f"Ο σχετικός όγκος στο {volume_ratio:.2f}x δείχνει ενεργή συμμετοχή κεφαλαίων και επιβεβαιώνει ότι η κίνηση δεν είναι απλώς thin-market drift."
    elif volume_ratio < 0.7:
        flow_text = f"Ο σχετικός όγκος στο {volume_ratio:.2f}x είναι υποτονικός, άρα η κίνηση χρειάζεται επιβεβαίωση πριν θεωρηθεί θεσμικά πειστική."
    else:
        flow_text = f"Ο σχετικός όγκος στο {volume_ratio:.2f}x είναι ουδέτερος, χωρίς ένδειξη ακραίας συσσώρευσης ή διανομής."

    trend_parts = []
    if pd.notna(ret_1m):
        trend_parts.append(f"μηνιαία απόδοση {ret_1m:+.1f}%")
    if pd.notna(dist_high):
        trend_parts.append(f"{abs(dist_high):.1f}% από το 52w high" if dist_high < 0 else f"{dist_high:+.1f}% πάνω από το 52w high")
    if pd.notna(dist_low):
        trend_parts.append(f"{dist_low:+.1f}% από το 52w low")
    if pd.notna(trend_score):
        trend_parts.append(f"trend score {int(trend_score)}/3")
    trend_context = ", ".join(trend_parts) if trend_parts else "περιορισμένα στοιχεία trend context"

    analyst_takeaway = (
        f"Συμπέρασμα analyst: το καθεστώς ταξινομείται ως '{regime_label}' με {signal_strength.lower()}. "
        f"Η βασική ερώτηση δεν είναι μόνο αν η τάση είναι θετική ή αρνητική, αλλά αν η τρέχουσα απόσταση από τον μέσο όρο υποστηρίζεται από επαρκή μεταβλητότητα, συμμετοχή όγκου και στατιστική σχέση με το {benchmark}."
    )

    return {
        "regime_label": regime_label,
        "deviation_text": deviation_text,
        "benchmark_text": benchmark_text,
        "beta_text": beta_text,
        "vol_text": vol_text,
        "flow_text": flow_text,
        "trend_context": trend_context,
        "signal_strength": signal_strength,
        "analyst_takeaway": analyst_takeaway,
    }


def build_regime_text(ticker: str, benchmark: str, metrics: dict) -> str:
    ontology = build_regime_ontology(ticker, benchmark, metrics)
    lines = [
        f"Καθεστώς: {ontology['regime_label']}",
        f"Απόκλιση από μέσο όρο: {ontology['deviation_text']}",
        f"Σχέση με benchmark: {ontology['benchmark_text']}",
        f"Systemic risk / Beta: {ontology['beta_text']}",
        f"Volatility context: {ontology['vol_text']}",
        f"Flow confirmation: {ontology['flow_text']}",
        f"Trend context: {ontology['trend_context']}",
        f"Ερμηνεία: {ontology['analyst_takeaway']}",
    ]
    return "\n\n".join(lines)


def build_volatility_text(volume_ratio) -> str:
    """
    Summarizes volume/volatility dynamics.
    """
    if not volume_ratio or pd.isna(volume_ratio):
        return "Στοιχεία όγκου μη διαθέσιμα."
        
    if volume_ratio > 1.5:
        return "Υψηλή συναλλακτική δραστηριότητα (Relative Volume > 1.5x). Αυξημένο ενδιαφέρον από θεσμικούς."
    elif volume_ratio < 0.5:
        return "Χαμηλός όγκος συναλλαγών. Πιθανή έλλειψη κατεύθυνσης (Consolidation)."
    else:
        return "Κανονικά επίπεδα συναλλακτικής δραστηριότητας."


"""
user_explanation_engine.py — Layer B: Explainable Output Generator
==================================================================
Translates institutional-grade metrics into retail-friendly Greek narratives.
Stateless and focused on 'Human-in-the-loop' communication.
"""


def generate_user_friendly_output(result: UnifiedAnalysisResult) -> dict:
    """
    Μετατρέπει το τεχνικό αποτέλεσμα σε γλώσσα που καταλαβαίνει ο απλός χρήστης.
    """
    return {
        "titre": "Συμπέρασμα",
        "summary": result.plain_summary,
        "why": result.why_this_signal,
        "risk_in_simple_words": result.user_friendly_risk,
        "what_to_do": result.recommended_action,
        "confidence": result.confidence_level,
        "main_reasons": result.key_drivers,
    }

def translate_metrics_to_plain_greek(result: UnifiedAnalysisResult) -> UnifiedAnalysisResult:
    """
    Εμπλουτίζει το UnifiedAnalysisResult με επεξηγηματικά πεδία.
    """
    # 1. Summary & Why
    if result.overall_signal == SignalStrength.BUY:
        result.plain_summary = "Η αγορά προσφέρει μια καλή ευκαιρία αγοράς με ελεγχόμενο ρίσκο."
        result.why_this_signal = "Οι δείκτες δείχνουν ανοδική τάση και οι αποτιμήσεις είναι ελκυστικές."
    elif result.overall_signal == SignalStrength.SELL:
        result.plain_summary = "Η αγορά φαίνεται υπερτιμημένη και επικίνδυνη αυτή τη στιγμή."
        result.why_this_signal = "Υπάρχει συνδυασμός υψηλής τιμής και αυξημένης πιθανότητας πτώσης."
    else:
        result.plain_summary = "Η αγορά βρίσκεται σε φάση αναμονής χωρίς ξεκάθαρη κατεύθυνση."
        result.why_this_signal = "Δεν υπάρχουν ισχυρά σήματα που να δικαιολογούν επιθετικές κινήσεις."

    # 2. User Friendly Risk
    if result.risk_level == RiskLevel.CRITICAL:
        result.user_friendly_risk = "Πολύ υψηλός κίνδυνος: Μεγάλες πιθανότητες για απότομη πτώση σύντομα."
    elif result.risk_level == RiskLevel.HIGH:
        result.user_friendly_risk = "Αυξημένο ρίσκο: Υπάρχει περίπου 1 στις 7 πιθανότητες για σημαντικές απώλειες."
    elif result.risk_level == RiskLevel.MEDIUM:
        result.user_friendly_risk = "Μέτριο ρίσκο: Η αγορά έχει τις συνηθισμένες διακυμάνσεις."
    else:
        result.user_friendly_risk = "Χαμηλό ρίσκο: Οι συνθήκες είναι σταθερές και ασφαλείς."

    # 3. Recommended Action
    if result.overall_signal == SignalStrength.SELL:
        result.recommended_action = "Μείωσε τη θέση σου κατά 30-40% ή περίμενε καλύτερη τιμή εισόδου."
    elif result.overall_signal == SignalStrength.BUY:
        result.recommended_action = "Εξέτασε το ενδεχόμενο αύξησης της θέσης σου σταδιακά."
    else:
        result.recommended_action = "Διατήρησε την τρέχουσα θέση σου χωρίς βιαστικές κινήσεις."

    # 4. Confidence Level
    conf_pct = int(result.confidence_score * 100)
    if result.confidence_score > 0.8:
        result.confidence_level = f"Υψηλή εμπιστοσύνη ({conf_pct}%)"
    elif result.confidence_score > 0.6:
        result.confidence_level = f"Μέτρια εμπιστοσύνη ({conf_pct}%)"
    else:
        result.confidence_level = f"Χαμηλή εμπιστοσύνη ({conf_pct}%)"

    return result

def generate_central_banker_advisory(macro_state: dict, pricing_power_score: int, ticker: str) -> str:
    regime = macro_state.get("regime", "NORMAL")
    output_gap = macro_state.get("output_gap_proxy", 0)
    inflation = macro_state.get("inflation_momentum_3m", 0)
    
    # Κτίσιμο του κειμένου με σύνθεση
    advisory_text = f"📊 **Μακροοικονομική Σύνθεση (Central Bank View):**\n"
    
    # 1. Ανάλυση της Μακρο-κατάστασης
    if regime == "GOLDILOCKS":
        advisory_text += "Η οικονομία βρίσκεται σε φάση 'Goldilocks'. Το χάσμα παραγωγής είναι θετικό, τα επιτόκια έχουν σταθεροποιηθεί και δεν υπάρχουν ισχυρές πληθωριστικές πιέσεις. "
    elif regime == "STAGFLATION_RISK":
        advisory_text += "Εντοπίζονται κίνδυνοι στασιμοπληθωρισμού. Το ενεργειακό κόστος αυξάνεται (Cost-Push) ενώ η ευρύτερη αγορά (Output Gap) δείχνει σημάδια κόπωσης. "
    elif regime == "OVERHEATING":
        advisory_text += "Σημάδια Υπερθέρμανσης (Overheating). Η ισχυρή οικονομική δραστηριότητα συνοδεύεται από αναζωπύρωση των τιμών ενέργειας/εμπορευμάτων, αυξάνοντας την πιθανότητα παρέμβασης της Fed. "
    else:
        advisory_text += "Το μακροοικονομικό περιβάλλον είναι μεικτό, με τα επιτόκια και τον πληθωρισμό εντός φυσιολογικών ιστορικών ορίων (Steady Corridor). "

    advisory_text += "\n\n🎯 **Στρατηγική Τοποθέτηση:**\n"
    
    # 2. Η Ρεαλιστική Συμβουλή (Micro + Macro Integration)
    if pricing_power_score >= 4:
        advisory_text += f"Η **{ticker}** διαθέτει **Εξαιρετική Τιμολογιακή Δύναμη (Pricing Power: {pricing_power_score}/5)**. Ιστορικά, έχει αποδείξει ότι διατηρεί τα περιθώρια κέρδους της ανεξάρτητα από το κόστος εισροών. "
        if regime == "STAGFLATION_RISK":
            advisory_text += "Λειτουργεί ως αμυντικό Hedge απέναντι στον πληθωρισμό. Διακράτηση θέσης (Hold). Αποφύγετε ρευστοποιήσεις πανικού."
        else:
            advisory_text += "Στο τρέχον περιβάλλον, τα ισχυρά θεμελιώδη της επιτρέπουν οργανική ανάπτυξη. Σταδιακή Συσσώρευση (Selective Accumulation) σε πτωτικές συνεδριάσεις."
            
    elif pricing_power_score <= 2:
        advisory_text += f"Η **{ticker}** έχει **Αδύναμη Ανθεκτικότητα Περιθωρίων (Pricing Power: {pricing_power_score}/5)**. Είναι ευάλωτη σε μακροοικονομικά σοκ. "
        if regime in ["STAGFLATION_RISK", "OVERHEATING"]:
            advisory_text += "Τα αυξημένα κόστη ενδέχεται να συμπιέσουν άμεσα την κερδοφορία της, καθώς αδυνατεί να αυξήσει τις τιμές στους καταναλωτές. Προτείνεται στρατηγική μείωση έκθεσης (Macro-Prudential Trim)."
        else:
            advisory_text += "Ευνοείται από την τρέχουσα ηρεμία της αγοράς, αλλά απαιτεί αυστηρό risk management (Trailing Stop) αν τα επιτόκια γίνουν ξανά ασταθή."
            
    else:
        advisory_text += f"Η **{ticker}** έχει μεσαία ευαισθησία στο μακροοικονομικό περιβάλλον (Pricing Power: {pricing_power_score}/5). Διατήρηση ουδέτερης στάσης (Neutral Hold) με παρακολούθηση των επόμενων earnings."

    return advisory_text
