"""
app.py — UI Layer (Streamlit)
=============================
The ONLY file that imports Streamlit. Zero business logic.
All computation is delegated to the layer modules.

Entry point for Antigravity / local runner:
  streamlit run app.py

File tree (all in same directory):
  app.py              ← this file
  Orchestrator2.py     ← pipeline brain
  riskengine8_2.py      ← quant risk service
  Valuation_engine9.py ← valuation service
  Advisory_layer3.py   ← insights synthesis
  Portfolio_engine4.py ← FIFO + P/L
  marketLayer5.py     ← yfinance ingestion
  DataLayer6.py       ← SQLite + metadata
  utils.py            ← pure helpers
"""
from __future__ import annotations
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
try:
    pd.options.mode.use_inf_as_na = True
except Exception:
    pass

# ── Layer imports ─────────────────────────────────────────────────────────────
from utils import (
    format_human_value, format_percent, wrap_text,
    format_display_table,
)
from DataLayer6 import (
    init_portfolio_db, db_add_user, db_get_transactions,
    db_log_transaction, db_delete_transaction,
)
from marketLayer5 import (
    fetch_range_reference_data,
)
from Portfolio_engine4 import (
    compute_portfolio_state, compute_weighted_metrics, validate_sell,
)
from Orchestrator2 import build_analysis, run_portfolio_pipeline, LiveYFinanceRepository
from Advisory_layer3 import build_live_briefing_text, build_benchmark_sensitivity_text

from ui_components import THEME, ALTAIR_CONFIG, apply_chart_theme, render_range_bar, render_earnings_dots, build_rev_earning_chart, build_line_area_chart, build_dual_line_chart

# ---------------------------------------------------------------------------
# Analyst helpers (kept in UI layer — pure display logic)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_styles():
    st.markdown(f"""
        <style>
        :root {{
            --bg: #0F1117;
            --panel: #161A24;
            --border: #22262F;
            --text: #FFFFFF;
            --muted: #8A8F9A;
            --accent: #00E5C4;
        }}
        .stApp {{ background: var(--bg); color: var(--text); font-family: "Inter", "SF Pro Display", sans-serif; }}
        [data-testid="stSidebar"] {{ background: #0b0d12; border-right: 1px solid var(--border); }}
        .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1580px; }}
        
        /* Modern KPI Cards */
        div[data-testid="stMetric"], .kpi-card, .glass-card {{
            background: var(--panel) !important; 
            border: 1px solid var(--border) !important; 
            border-radius: 8px !important; 
            box-shadow: none !important;
            padding: 16px !important;
        }}
        
        .section-label {{ 
            color: var(--muted); 
            text-transform: uppercase; 
            letter-spacing: 0.1em;
            font-size: 0.72rem; 
            font-weight: 600; 
            margin-bottom: 12px;
            border-bottom: 1px solid var(--border); 
            padding-bottom: 6px; 
        }}
            
        .kpi-title {{ color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }}
        .kpi-value {{ color: var(--text); font-size: 1.25rem; font-weight: 700; margin-top: 4px; }}
        .kpi-sub {{ color: var(--muted); font-size: 0.75rem; margin-top: 4px; }}
        .kpi-delta {{ font-size: 0.75rem; font-weight: 600; margin-top: 2px; }}
        
        .pl-pos {{ color: #26A69A !important; }}
        .pl-neg {{ color: #EF5350 !important; }}
        
        .commentary-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
            padding: 1rem; color: #D1D4DC; line-height: 1.6; white-space: pre-wrap; font-size: 0.85rem; }}
            
        .pulse-stat-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }}
        .pulse-stat-label {{ font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
        .pulse-stat-value {{ font-size: 1.05rem; font-weight: 700; color: var(--text); }}
        
        /* Minimalist Buttons */
        .stButton button {{
            background: transparent !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-size: 0.8rem !important;
            padding: 0.4rem 1rem !important;
            transition: all 0.2s ease;
            width: 100%;
        }}
        .stButton button:hover {{
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            background: rgba(0, 229, 196, 0.05) !important;
        }}
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid var(--border); }}
        .stTabs [data-baseweb="tab"] {{ color: var(--muted); padding: 0.5rem 1rem; font-size: 0.85rem; }}
        .stTabs [aria-selected="true"] {{ color: var(--text); border-bottom: 2px solid var(--accent) !important; }}
        
        /* Inputs */
        div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
            background: #0b0d12 !important; color: var(--text) !important;
            border: 1px solid var(--border) !important; border-radius: 6px !important; }}
        
        /* Charts Background */
        .vega-bind {{ color: var(--text) !important; }}
        </style>""", unsafe_allow_html=True)








# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Render: Hero + KPI rail
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown("""
    <div style="padding: 2.2rem 0 1.8rem; border-bottom: 1px solid #22262F; margin-bottom: 1.5rem;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:42px; height:42px; background:#00E5C4; border-radius:8px; display:flex; align-items:center; justify-content:center;">
                <span style="color:#0F1117; font-weight:700; font-size:1.35rem;">Q</span>
            </div>
            <div>
                <h1 style="margin:0; color:#FFFFFF; font-size:2.35rem; font-weight:600; letter-spacing:-0.02em;">Quant Terminal</h1>
                <p style="margin:0.15rem 0 0; color:#8A8F9A; font-size:0.95rem;">Institutional Portfolio Intelligence</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_rail(cards):
    st.markdown('<div class="section-label">Valuation Rail</div>', unsafe_allow_html=True)
    for card in cards:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">{card["title"]}</div><div class="kpi-value">{card["value"]}</div><div class="kpi-sub">{card["subtitle"]}</div><div class="kpi-delta">{card["delta"]}</div></div>', unsafe_allow_html=True)


def render_benchmark_multiples(benchmark_symbol, benchmark_fundamentals):
    snapshot = benchmark_fundamentals["snapshot"]
    st.markdown('<div class="section-label">Benchmark Multiples</div>', unsafe_allow_html=True)
    for label, value in [
        ("Trailing P/E", f"{format_human_value(snapshot['trailing_pe'])}x"),
        ("Forward P/E",  f"{format_human_value(snapshot['forward_pe'])}x"),
        ("P/S Ratio",    f"{format_human_value(snapshot['ps_ratio'])}x"),
    ]:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">{benchmark_symbol} {label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">Updated: {snapshot["updated"]}</div></div>', unsafe_allow_html=True)


def render_secondary_metrics(analysis):
    metrics = analysis["latest_metrics"]
    cols = st.columns(4, gap="medium")
    items = [
        ("RSI 14", f"{metrics['latest_rsi']:.2f}"),
        ("Sharpe 60d", f"{metrics['latest_sharpe']:.2f}"),
        (f"Beta {analysis['benchmark']}", f"{metrics['latest_beta_benchmark']:.2f}"),
        ("Drawdown", f"{metrics['latest_drawdown']:.2f}%"),
    ]
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="glass-card"><div class="kpi-title">{label}</div><div class="kpi-value" style="font-size:1.12rem;">{value}</div></div>', unsafe_allow_html=True)


def render_range_bar(label, low_value, high_value, current_value, color="#FFFFFF"):
    if any(pd.isna(v) for v in [low_value, high_value, current_value]) or high_value <= low_value:
        st.markdown(f'<div class="glass-card"><div class="kpi-title">{label}</div><div class="small-note">Range data not available.</div></div>', unsafe_allow_html=True)
        return
    position = max(0.0, min(100.0, ((current_value - low_value) / (high_value - low_value)) * 100))
    st.markdown(f"""<div class="glass-card"><div class="kpi-title">{label}</div>
        <div style="display:flex;justify-content:space-between;color:#8ea0b8;font-size:0.86rem;margin:0.25rem 0 0.55rem;">
            <span>{low_value:.2f}</span><span>{high_value:.2f}</span></div>
        <div style="position:relative;height:10px;border-radius:999px;background:#18222e;overflow:visible;">
            <div style="position:absolute;inset:0;border-radius:999px;background:linear-gradient(90deg,#1f2a36,#253445);"></div>
            <div style="position:absolute;left:calc({position}% - 7px);top:-4px;width:14px;height:14px;border-radius:999px;background:{color};box-shadow:0 0 0 3px {color}14;"></div>
        </div><div style="margin-top:0.55rem;color:#dfe7f2;font-size:0.9rem;">Last: {current_value:.2f}</div></div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Tab content
# ---------------------------------------------------------------------------

def render_overview_tab(analysis):
    # Construct DF from the full series but tail(252) for the visual window
    # This ensures SMAs (calculated on 2y) are fully populated in the view.
    overview_df = pd.DataFrame({
        "Date": analysis["stock"].index, 
        "Price": analysis["stock"].values, 
        "SMA 50": analysis["sma50"].values, 
        "SMA 200": analysis["sma200"].values,
        "BB_Upper": analysis.get("bb_upper", pd.Series()).values,
        "BB_Lower": analysis.get("bb_lower", pd.Series()).values,
        "SMA50_Upper": analysis.get("sma50_upper", pd.Series()).values,
        "SMA50_Lower": analysis.get("sma50_lower", pd.Series()).values
    }).tail(252)
    
    with st.container(border=True):
        st.markdown('<div class="section-label">Institutional Technical Architecture</div>', unsafe_allow_html=True)
        
        # ── Advanced Yahoo/TradingView Style Layout ──
        price_min = float(overview_df["Price"].min())
        price_max = float(overview_df["Price"].max())
        padding = (price_max - price_min) * 0.2
        
        # 1. Bollinger Bands (Volatility Shading)
        bb_area = alt.Chart(overview_df).mark_area(
            color='rgba(41, 98, 255, 0.05)', # Light TradingView Blue
            interpolate='monotone',
            opacity=0.3
        ).encode(
            x="Date:T",
            y="BB_Lower:Q",
            y2="BB_Upper:Q"
        )
        
        # 2. SMA 50 Volatility Ribbon
        sma50_ribbon = alt.Chart(overview_df).mark_area(
            color='rgba(21, 181, 122, 0.08)', # Institutional Green
            interpolate='monotone'
        ).encode(
            x="Date:T",
            y="SMA50_Lower:Q",
            y2="SMA50_Upper:Q"
        )
        
        # 3. Base Price Area (Premium Gradient)
        price_area = alt.Chart(overview_df).mark_area(
            line={'color': '#FFFFFF', 'strokeWidth': 2.2},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='rgba(255, 255, 255, 0.12)', offset=0),
                       alt.GradientStop(color='rgba(255, 255, 255, 0)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            ),
            interpolate='monotone',
            clip=True
        ).encode(
            x=alt.X("Date:T", axis=alt.Axis(labelColor="#8A8F9A", title=None, grid=False, tickCount=10)),
            y=alt.Y("Price:Q", axis=alt.Axis(labelColor="#8A8F9A", title=None, grid=True, gridColor="rgba(255,255,255,0.05)", tickCount=6),
                    scale=alt.Scale(domain=[price_min - padding, price_max + padding], zero=False)),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Price:Q", format="$.2f", title="Price")]
        )
        
        # 4. Moving Averages
        sma50_line = alt.Chart(overview_df).mark_line(color="#15B57A", strokeWidth=1.5, opacity=0.9).encode(x="Date:T", y="SMA 50:Q")
        sma200_line = alt.Chart(overview_df).mark_line(color="#F59E0B", strokeWidth=1.5, opacity=0.9).encode(x="Date:T", y="SMA 200:Q")
        
        # 5. Dynamic Markers
        latest_price = float(overview_df["Price"].dropna().iloc[-1])
        latest_date = overview_df["Date"].dropna().iloc[-1]
        marker = alt.Chart(pd.DataFrame({"Date": [latest_date], "Price": [latest_price]})).mark_point(color="#FFFFFF", size=80, filled=True).encode(x="Date:T", y="Price:Q")
        
        # Merge All Layers
        final_chart = (bb_area + sma50_ribbon + price_area + sma50_line + sma200_line + marker).properties(height=520)
        st.altair_chart(apply_chart_theme(final_chart), use_container_width=True)
        
        # Professional Technical Legend
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:25px; margin-top:-5px; font-size:0.7rem; color:#8A8F9A; text-transform:uppercase; letter-spacing:0.05em;">
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; background:rgba(41, 98, 255, 0.2); border-radius:2px;"></div> Volatility (BB)</div>
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:12px; height:2px; background:#15B57A;"></div> 50 DMA</div>
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:12px; height:2px; background:#F59E0B;"></div> 200 DMA</div>
            <div style="display:flex; align-items:center; gap:6px;"><div style="width:6px; height:6px; background:#FFFFFF; border-radius:50%;"></div> Current: $""" + f"{latest_price:,.2f}" + """</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-label" style="margin-top:20px;">Regime & Narrative</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:var(--muted);margin-bottom:10px;">Ontology-based note: η απόκλιση μεταφράζεται σε ποσοστιαία απόσταση από τον 50ήμερο μέσο όρο, benchmark discipline, systemic beta risk και quality of participation.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="commentary-card">{wrap_text(analysis["regime_text"])}</div>', unsafe_allow_html=True)


def build_market_pulse_commentary(ticker, range_key, pulse_data: pd.DataFrame, probability_metrics: dict | None = None):
    close_series = pulse_data["Close"].dropna() if isinstance(pulse_data, pd.DataFrame) and "Close" in pulse_data else pd.Series(dtype=float)
    if close_series.empty or len(close_series) < 2:
        return f"Insufficient data for {ticker} pulse analysis."

    change_pct = (float(close_series.iloc[-1]) / float(close_series.iloc[0]) - 1) * 100
    returns = close_series.pct_change().dropna()

    time_span_days = (close_series.index[-1] - close_series.index[0]).days or 1
    obs_per_day = len(close_series) / time_span_days
    ann_factor = np.sqrt(252 * max(1, obs_per_day))
    vol = returns.std() * ann_factor * 100 if not returns.empty else 0

    latest_row = pulse_data.dropna(how="all").iloc[-1] if isinstance(pulse_data, pd.DataFrame) and not pulse_data.empty else pd.Series(dtype=float)
    open_px = float(latest_row.get("Open")) if pd.notna(latest_row.get("Open")) else np.nan
    high_px = float(latest_row.get("High")) if pd.notna(latest_row.get("High")) else np.nan
    low_px = float(latest_row.get("Low")) if pd.notna(latest_row.get("Low")) else np.nan
    close_px = float(latest_row.get("Close")) if pd.notna(latest_row.get("Close")) else float(close_series.iloc[-1])
    volume = float(latest_row.get("Volume")) if pd.notna(latest_row.get("Volume")) else np.nan

    body_pct = ((close_px / open_px) - 1.0) * 100.0 if pd.notna(open_px) and open_px != 0 else np.nan
    intraday_range_pct = ((high_px / low_px) - 1.0) * 100.0 if pd.notna(high_px) and pd.notna(low_px) and low_px != 0 else np.nan

    volume_ratio = np.nan
    if "Volume" in pulse_data and pulse_data["Volume"].dropna().shape[0] >= 5:
        vol_series = pulse_data["Volume"].dropna().astype(float)
        vol_avg = float(vol_series.tail(min(10, len(vol_series))).mean())
        if vol_avg > 0 and pd.notna(volume):
            volume_ratio = volume / vol_avg

    if pd.notna(body_pct):
        if body_pct > 0.35:
            candle_text = f"Το σημερινό κερί είναι ανοδικό: άνοιξε στα {open_px:,.2f} και κινείται προς κλείσιμο στα {close_px:,.2f}, δηλαδή περίπου {body_pct:+.2f}% πάνω από το άνοιγμα."
        elif body_pct < -0.35:
            candle_text = f"Το σημερινό κερί είναι καθοδικό: άνοιξε στα {open_px:,.2f} και βρίσκεται στα {close_px:,.2f}, δηλαδή περίπου {body_pct:+.2f}% χαμηλότερα από το άνοιγμα."
        else:
            candle_text = f"Το σημερινό κερί είναι σχετικά ουδέτερο: άνοιξε στα {open_px:,.2f} και η τρέχουσα τιμή στα {close_px:,.2f} απέχει μόλις {body_pct:+.2f}% από το άνοιγμα."
    else:
        candle_text = f"Η τρέχουσα τιμή είναι στα {close_px:,.2f}, αλλά δεν υπάρχουν πλήρη στοιχεία για ασφαλή ανάγνωση του σημερινού κεριού."

    range_text = (
        f" Το εύρος της κίνησης μέχρι τώρα είναι από {low_px:,.2f} έως {high_px:,.2f}, δηλαδή περίπου {intraday_range_pct:.2f}% εύρος διακύμανσης."
        if pd.notna(intraday_range_pct) else ""
    )

    if pd.notna(volume_ratio):
        if volume_ratio >= 1.2:
            volume_text = f" Ο όγκος είναι αυξημένος, περίπου {volume_ratio:.2f}x πάνω από τον πρόσφατο μέσο όρο, άρα η κίνηση έχει καλύτερη συμμετοχή."
        elif volume_ratio <= 0.8:
            volume_text = f" Ο όγκος είναι χαμηλός, περίπου {volume_ratio:.2f}x του πρόσφατου μέσου όρου, άρα η κίνηση θέλει προσοχή γιατί δεν έχει ακόμη ισχυρή επιβεβαίωση."
        else:
            volume_text = f" Ο όγκος είναι κοντά στο συνηθισμένο, περίπου {volume_ratio:.2f}x του πρόσφατου μέσου όρου."
    else:
        volume_text = ""

    probability_text = ""
    if probability_metrics:
        sigma_abs = abs(probability_metrics["sigma_score"])
        tail_prob = probability_metrics["tail_probability_pct"]
        if sigma_abs < 0.75:
            state_text = "πολύ κοντά στο φυσιολογικό εύρος"
        elif sigma_abs < 1.5:
            state_text = "λίγο έξω από το κέντρο της κανονικής κίνησης, αλλά ακόμη σε λογικά όρια"
        elif sigma_abs < 2.5:
            state_text = "αρκετά τεντωμένη σε σχέση με τη συνηθισμένη συμπεριφορά"
        else:
            state_text = "σε σπάνιο και ακραίο σημείο για το πρόσφατο pulse"
        probability_text = (
            f" Στατιστικά, η τιμή είναι {probability_metrics['deviation_pct']:+.2f}% μακριά από τον πρόσφατο μέσο όρο και βρίσκεται {state_text}. "
            f"Με βάση Student-t προσέγγιση, μια τόσο μεγάλη ή μεγαλύτερη απόκλιση εμφανίζεται περίπου στο {tail_prob:.1f}% παρόμοιων περιπτώσεων. "
            f"Άρα όταν αυτό το ποσοστό είναι χαμηλό, η κίνηση είναι πιο ασυνήθιστη· όταν είναι υψηλό, η αγορά παραμένει πιο κοντά στη συνηθισμένη συμπεριφορά της."
        )

    regime_text = ""
    if change_pct > 0.5 and vol <= 35:
        regime_text = f" Συνολικά, το pulse δείχνει ανοδική κίνηση {change_pct:+.2f}% με ελεγχόμενη μεταβλητότητα {vol:.1f}%, κάτι που ταιριάζει περισσότερο με orderly accumulation."
    elif change_pct > 0.5 and vol > 35:
        regime_text = f" Συνολικά, η άνοδος {change_pct:+.2f}% συνοδεύεται από υψηλή μεταβλητότητα {vol:.1f}%, άρα η κίνηση είναι πιο επιθετική και λιγότερο καθαρή."
    elif change_pct < -0.5 and vol > 35:
        regime_text = f" Συνολικά, η πτώση {change_pct:.2f}% με μεταβλητότητα {vol:.1f}% δείχνει πίεση που μπορεί να εξελιχθεί σε panic move αν συνεχιστεί."
    elif change_pct < -0.5:
        regime_text = f" Συνολικά, η πτώση {change_pct:.2f}% παραμένει ελεγχόμενη με μεταβλητότητα {vol:.1f}%."
    else:
        regime_text = f" Συνολικά, η αγορά κινείται σχετικά ουδέτερα με μεταβολή {change_pct:+.2f}% και μεταβλητότητα {vol:.1f}%."

    return candle_text + range_text + volume_text + probability_text + regime_text


def compute_market_pulse_probability_metrics(close_series: pd.Series, window: int) -> dict:
    if close_series.empty or len(close_series) < max(5, window):
        return {}

    try:
        from scipy.stats import t as student_t
    except Exception:
        return {}

    sample = close_series.tail(window).astype(float)
    mean_price = float(sample.mean())
    std_price = float(sample.std(ddof=1))
    latest_price = float(sample.iloc[-1])
    n_obs = len(sample)
    df = max(n_obs - 1, 1)

    if std_price <= 0:
        deviation_pct = 0.0
        sigma_score = 0.0
        tail_prob = 1.0
    else:
        deviation_pct = ((latest_price / mean_price) - 1.0) * 100.0 if mean_price else np.nan
        sigma_score = (latest_price - mean_price) / std_price
        tail_prob = float(student_t.sf(abs(sigma_score), df) * 2.0)

    likelihood_pct = max(0.0, min(100.0, (1.0 - tail_prob) * 100.0))
    return {
        "mean_price": mean_price,
        "std_price": std_price,
        "sigma_score": sigma_score,
        "tail_probability": tail_prob,
        "tail_probability_pct": tail_prob * 100.0,
        "likelihood_pct": likelihood_pct,
        "deviation_pct": deviation_pct,
        "df": df,
        "n_obs": n_obs,
    }


def render_market_pulse_tab(ticker, analysis):
    range_key = st.radio("Range", ["1D", "1W", "1M", "1Y"], horizontal=True, key="pulse_range")
    try:
        from marketLayer5 import fetch_market_pulse_data
        pulse_data, config = fetch_market_pulse_data(ticker, range_key)
    except ValueError as e:
        st.error(str(e)); return

    close = pulse_data["Close"].dropna() if "Close" in pulse_data else pd.Series(dtype=float)
    if close.empty:
        st.warning("Δεν υπάρχουν διαθέσιμα δεδομένα."); return

    pulse_df = pd.DataFrame({"Date": close.index, "Close": close.values}).copy()
    window_map = {"1D": 12, "1W": 10, "1M": 12, "1Y": 20}
    trend_window = window_map.get(range_key, 12)
    slow_window = max(trend_window * 2, trend_window + 3)

    pulse_df["Mean"] = pulse_df["Close"].rolling(trend_window, min_periods=max(4, trend_window // 2)).mean()
    pulse_df["EMA Fast"] = pulse_df["Close"].ewm(span=trend_window, adjust=False).mean()
    pulse_df["EMA Slow"] = pulse_df["Close"].ewm(span=slow_window, adjust=False).mean()
    rolling_std = pulse_df["Close"].rolling(trend_window, min_periods=max(3, trend_window // 2)).std()
    pulse_df["Std Upper"] = pulse_df["Mean"] + rolling_std.fillna(0)
    pulse_df["Std Lower"] = pulse_df["Mean"] - rolling_std.fillna(0)

    price_floor = float(pulse_df[["Close", "Std Lower"]].min().min())
    price_ceiling = float(pulse_df[["Close", "Std Upper"]].max().max())
    price_span = price_ceiling - price_floor
    padding = price_span * 0.12 if price_span > 0 else max(price_ceiling * 0.01, 1.0)
    latest_price = float(close.iloc[-1])
    latest_point_df = pd.DataFrame({"Date": [close.index[-1]], "Close": [latest_price]})
    start_price = float(close.iloc[0]) if len(close) > 0 else latest_price
    reference_line_df = pd.DataFrame({"Reference": [start_price]})
    probability_metrics = compute_market_pulse_probability_metrics(close, trend_window)

    reference_daily = fetch_range_reference_data(ticker)
    latest_session = reference_daily.dropna(how="all").iloc[-1] if not reference_daily.empty else pd.Series(dtype=float)
    close_52w = reference_daily["Close"].dropna() if (not reference_daily.empty and "Close" in reference_daily) else pd.Series(dtype=float)
    change_pct = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100 if len(close) > 1 else np.nan
    line_color = "#26A69A" if change_pct >= 0 else "#EF5350"

    base = alt.Chart(pulse_df).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(grid=False, labelColor="#8A8F9A"))
    )

    std_band = base.mark_area(color="rgba(41, 98, 255, 0.10)", interpolate='monotone').encode(
        y=alt.Y("Std Lower:Q", scale=alt.Scale(domain=[price_floor - padding, price_ceiling + padding], zero=False), title=None,
                axis=alt.Axis(labelColor="#8A8F9A", grid=False)),
        y2="Std Upper:Q"
    )
    mean_line = base.mark_line(color="#72F1FF", strokeWidth=2.2, opacity=0.95, interpolate='monotone').encode(y="Mean:Q")
    fast_line = base.mark_line(color=line_color, strokeWidth=3, interpolate='monotone').encode(y="Close:Q")
    ema_fast = base.mark_line(color="#15B57A", strokeWidth=2, opacity=0.95, interpolate='monotone').encode(y="EMA Fast:Q")
    ema_slow = base.mark_line(color="#F59E0B", strokeWidth=2, opacity=0.9, interpolate='monotone').encode(y="EMA Slow:Q")
    ref_rule = alt.Chart(reference_line_df).mark_rule(color="#3A4757", strokeDash=[4, 4]).encode(y="Reference:Q")
    last_marker = alt.Chart(latest_point_df).mark_point(color="#FFFFFF", filled=True, size=90, stroke=line_color, strokeWidth=2).encode(x="Date:T", y="Close:Q")

    pulse_chart = (std_band + mean_line + fast_line + ema_fast + ema_slow + ref_rule + last_marker).properties(height=340)

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-label">Pulse Chart</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="display:flex; justify-content:center; gap:22px; margin-top:-4px; font-size:0.7rem; color:#8A8F9A; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; background:rgba(41, 98, 255, 0.18); border-radius:2px;"></div> Std Deviation Band</div>
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:12px; height:2px; background:#72F1FF;"></div> Statistical Mean</div>
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:12px; height:2px; background:#15B57A;"></div> Fast Trend</div>
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:12px; height:2px; background:#F59E0B;"></div> Slow Trend</div>
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:12px; height:2px; background:#3A4757;"></div> Range Anchor</div>
            </div>
            """, unsafe_allow_html=True)
            st.altair_chart(apply_chart_theme(pulse_chart, height=340), use_container_width=True)
        stats_cols = st.columns(6, gap="small")
        metrics_data = [
            ("Low", f"{price_floor:,.2f}"), 
            ("High", f"{price_ceiling:,.2f}"), 
            ("Last", f"{latest_price:,.2f}"), 
            ("Move", format_percent(change_pct)),
            ("Mean Δ", f"{probability_metrics.get('deviation_pct', np.nan):+.2f}%" if probability_metrics else "-"),
            ("Tail Prob", f"{probability_metrics.get('tail_probability_pct', np.nan):.1f}%" if probability_metrics else "-"),
        ]
        for i, (label, val) in enumerate(metrics_data):
            stats_cols[i].markdown(f"""
                <div class="pulse-stat-card">
                    <div class="pulse-stat-label">{label}</div>
                    <div class="pulse-stat-value">{val}</div>
                </div>
            """, unsafe_allow_html=True)
        if probability_metrics:
            sigma_abs = abs(probability_metrics["sigma_score"])
            sigma_direction = "πάνω" if probability_metrics["sigma_score"] > 0 else "κάτω"
            if sigma_abs < 0.75:
                plain_state = "πολύ κοντά στον συνηθισμένο μέσο όρο"
            elif sigma_abs < 1.5:
                plain_state = "λίγο απομακρυσμένη από το φυσιολογικό εύρος, αλλά όχι σε ανησυχητικό βαθμό"
            elif sigma_abs < 2.5:
                plain_state = "αρκετά τεντωμένη σε σχέση με το πρόσφατο συνηθισμένο εύρος"
            else:
                plain_state = "σε ασυνήθιστα ακραίο σημείο για το πρόσφατο pulse"

            t_note = (
                f"Με απλά λόγια, η τιμή είναι τώρα {probability_metrics['deviation_pct']:+.2f}% {sigma_direction if sigma_abs > 0 else ''} "
                f"από τον πρόσφατο μέσο όρο και βρίσκεται {plain_state}. "
                f"Το μοντέλο Student-t εκτιμά ότι μια τόσο μεγάλη ή μεγαλύτερη απόκλιση εμφανίζεται περίπου στο "
                f"{probability_metrics['tail_probability_pct']:.1f}% των αντίστοιχων περιπτώσεων. "
                f"Άρα δεν μιλάμε για κάτι εντελώς σπάνιο, αλλά ούτε και για τελείως ουδέτερη θέση όταν η απόκλιση μεγαλώνει."
            )
            st.markdown(f'<div class="commentary-card" style="margin-top:14px;">{t_note}</div>', unsafe_allow_html=True)
        rc = st.columns(2, gap="medium")
        with rc[0]: render_range_bar("Day's Range", latest_session.get("Low", np.nan), latest_session.get("High", np.nan), latest_price)
        with rc[1]: render_range_bar("52W Range", float(close_52w.min()) if not close_52w.empty else np.nan, float(close_52w.max()) if not close_52w.empty else np.nan, latest_price)
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-label">Coverage & Analyst Targets</div>', unsafe_allow_html=True)
            
            snap = analysis["financials"]["snapshot"]
            t_high = snap.get("target_high")
            t_low  = snap.get("target_low")
            t_mean = snap.get("target_mean")
            rec    = str(snap.get("recommendation", "N/A")).upper()
            
            cols = st.columns(2)
            cols[0].markdown(f'<div style="font-size:0.75rem;color:var(--muted);">Consensus</div><div style="font-size:1.1rem;font-weight:700;color:#2962FF;">{rec}</div>', unsafe_allow_html=True)
            if t_mean and latest_price:
                upside = (t_mean / latest_price - 1) * 100
                color = "#15B57A" if upside > 0 else "#EF5350"
                cols[1].markdown(f'<div style="font-size:0.75rem;color:var(--muted);">Implied Upside</div><div style="font-size:1.1rem;font-weight:700;color:{color};">{upside:+.1f}%</div>', unsafe_allow_html=True)
            
            st.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)
            if t_high: st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;"><span>Target High</span><span style="font-weight:600;">${t_high:.2f}</span></div>', unsafe_allow_html=True)
            if t_mean: st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;"><span>Target Mean</span><span style="font-weight:600;">${t_mean:.2f}</span></div>', unsafe_allow_html=True)
            if t_low:  st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;"><span>Target Low</span><span style="font-weight:600;">${t_low:.2f}</span></div>', unsafe_allow_html=True)
            
            st.markdown('<hr style="margin:15px 0; border:0; border-top:1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
            commentary = build_market_pulse_commentary(ticker, range_key, pulse_data, probability_metrics)
            st.markdown(f'<div style="font-size:0.85rem; color:#D1D4DC; line-height:1.5;">{commentary}</div>', unsafe_allow_html=True)


def render_financials_tab(analysis):
    financials = analysis["financials"]
    snapshot = financials["snapshot"]
    st.markdown('<div class="section-label">Financial Snapshot</div>', unsafe_allow_html=True)
    cards = st.columns(8, gap="medium")
    items = [
        ("Last Quarter", snapshot["last_quarter"]),
        ("Trailing EPS", format_human_value(snapshot["trailing_eps"])),
        ("Forward EPS",  format_human_value(snapshot["forward_eps"])),
        ("Trailing P/E", f"{format_human_value(snapshot['trailing_pe'])}x"),
        ("Forward P/E",  f"{format_human_value(snapshot['forward_pe'])}x"),
        ("P/S",          f"{format_human_value(snapshot['ps_ratio'])}x"),
        ("Profit Margin", format_percent(snapshot["profit_margin"])),
        ("Net Margin",   format_percent(snapshot["net_profit_margin"])),
    ]
    for col, (label, value) in zip(cards, items):
        col.markdown(f'<div class="glass-card"><div class="kpi-title">{label}</div><div class="kpi-value" style="font-size:1.15rem;">{value}</div></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Quarterly", "Annual", "Forward Earnings", "Forward Revenue", "EPS Revisions"])
    with tabs[0]: st.dataframe(format_display_table(financials["quarterly"]), width="stretch", height=360)
    with tabs[1]: 
        ticker = analysis.get("ticker")
        if ticker:
            cols_charts = st.columns([1, 1], gap="large")
            with cols_charts[0]:
                build_rev_earning_chart(financials["annual"])
            with cols_charts[1]:
                render_earnings_dots(ticker)
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        st.dataframe(format_display_table(financials["annual"]), width="stretch", height=360)
    with tabs[2]: st.dataframe(format_display_table(financials["earnings_estimate"]), width="stretch", height=320)
    with tabs[3]: st.dataframe(format_display_table(financials["revenue_estimate"]), width="stretch", height=320)
    with tabs[4]: st.dataframe(format_display_table(financials["eps_trend"]), width="stretch", height=320)


def render_live_briefing_tab(analysis):
    data = analysis["live_briefing"]
    benchmark_note = build_benchmark_sensitivity_text(analysis)
    cols = st.columns(6, gap="medium")
    for col, (lbl, val) in zip(cols, [
        ("1D", format_percent(data["ret_1d"])), ("1W", format_percent(data["ret_1w"])),
        ("1M", format_percent(data["ret_1m"])), ("YTD", format_percent(data["ytd_ret"])),
        ("Vol 20d", f"{data['realized_vol_20']:.1f}%" if pd.notna(data["realized_vol_20"]) else "-"),
        ("Vol Ratio", f"{data['volume_ratio']:.2f}x" if pd.notna(data["volume_ratio"]) else "-"),
    ]):
        col.markdown(f'<div class="glass-card"><div class="kpi-title">{lbl}</div><div class="kpi-value" style="font-size:1.15rem;">{val}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        close = data["daily"]["Close"].dropna()
        price_df = pd.DataFrame({"Date": close.index, "Close": close.values}).reset_index(drop=True)
        with st.container(border=True):
            st.markdown('<div class="section-label">Relative Strength</div>', unsafe_allow_html=True)
            baseline_df = pd.DataFrame({"Baseline": [100]})
            rel_strength = build_line_area_chart(price_df, "Date", "Close", color=THEME["accent"], fill=THEME["accent"])
            rel_strength = rel_strength + alt.Chart(baseline_df).mark_rule(color=THEME["muted"], strokeDash=[4, 2]).encode(y="Baseline:Q")
            st.altair_chart(apply_chart_theme(rel_strength, height=360), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-label">Research Brief</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="commentary-card">{wrap_text(build_live_briefing_text(data))}</div>', unsafe_allow_html=True)


def render_cash_flow_analysis(ticker_symbol, analysis_data):
    with st.container(border=True):
        st.markdown('<div class="section-label">Cash Flow Dynamics</div>', unsafe_allow_html=True)

        period_choice = st.radio(
            "Οπτική Γραφήματος:",
            options=["Τριμηνιαία (Quarterly)", "Ετήσια (Annual)"],
            horizontal=True,
            key=f"cf_toggle_{ticker_symbol}"
        )

        data_key = "quarterly" if "Quarterly" in period_choice else "annual"
        financials = analysis_data.get(data_key, pd.DataFrame()) if isinstance(analysis_data, dict) else pd.DataFrame()
        if not isinstance(financials, pd.DataFrame) or financials.empty:
            st.info(f"Δεν υπάρχουν διαθέσιμα στοιχεία για {ticker_symbol} ({data_key}).")
            return

        required_rows = ["Operating Cash Flow", "CapEx"]
        missing_core = [row for row in required_rows if row not in financials.index]
        if missing_core:
            st.warning(f"Λείπουν βασικά cash flow στοιχεία για {ticker_symbol}: {', '.join(missing_core)}.")
            return

        period_cols = list(financials.columns[:4])
        if not period_cols:
            st.info(f"Δεν υπάρχουν διαθέσιμες περίοδοι για {ticker_symbol}.")
            return

        ocf_series = pd.to_numeric(financials.loc["Operating Cash Flow", period_cols], errors="coerce")
        capex_series = pd.to_numeric(financials.loc["CapEx", period_cols], errors="coerce")
        capex_series = capex_series.apply(lambda x: -abs(x) if pd.notna(x) else np.nan)

        if "Free Cash Flow" in financials.index:
            fcf_series = pd.to_numeric(financials.loc["Free Cash Flow", period_cols], errors="coerce")
        else:
            fcf_series = ocf_series + capex_series

        plot_df = pd.DataFrame({
            "Period": period_cols,
            "Operating Cash Flow": ocf_series.values,
            "CapEx": capex_series.values,
            "Free Cash Flow": fcf_series.values,
        })

        plot_df = plot_df.dropna(subset=["Operating Cash Flow", "CapEx"], how="all")
        if plot_df.empty:
            st.info(f"Τα cash flow δεδομένα για {ticker_symbol} δεν είναι αρκετά καθαρά.")
            return

        plot_df = plot_df.iloc[::-1].reset_index(drop=True)
        period_order = plot_df["Period"].tolist()

        bar_df = plot_df.melt(
            id_vars="Period",
            value_vars=["Operating Cash Flow", "CapEx"],
            var_name="Metric",
            value_name="Value",
        )

        bar_colors = alt.Scale(
            domain=["Operating Cash Flow", "CapEx"],
            range=["#26A69A", "#EF5350"],
        )

        bars = alt.Chart(bar_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Period:N", sort=period_order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Metric:N", sort=["Operating Cash Flow", "CapEx"]),
            y=alt.Y("Value:Q", title="Cash Flow", axis=alt.Axis(format=",.2s")),
            color=alt.Color("Metric:N", scale=bar_colors, legend=alt.Legend(title=None, orient="top")),
            tooltip=[
                alt.Tooltip("Period:N", title="Περίοδος"),
                alt.Tooltip("Metric:N", title="Μετρική"),
                alt.Tooltip("Value:Q", title="Αξία ($)", format=",.2f"),
            ],
        )

        fcf_line = alt.Chart(plot_df).mark_line(color="#42A5F5", strokeWidth=2.5, point=alt.OverlayMarkDef(
            color="#42A5F5",
            fill="#0F1117",
            stroke="#FFFFFF",
            strokeWidth=1.5,
            size=70,
        )).encode(
            x=alt.X("Period:N", sort=period_order, title=None),
            y=alt.Y("Free Cash Flow:Q", title="Cash Flow"),
            tooltip=[
                alt.Tooltip("Period:N", title="Περίοδος"),
                alt.Tooltip("Free Cash Flow:Q", title="Free Cash Flow ($)", format=",.2f"),
            ],
        )

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="rgba(255,255,255,0.15)",
            strokeDash=[4, 3]
        ).encode(y="y:Q")

        chart = alt.layer(bars, zero_line, fcf_line).resolve_scale(
            y="shared"
        ).properties(height=320)

        st.altair_chart(apply_chart_theme(chart, height=320), use_container_width=True)

        latest = plot_df.iloc[-1]
        k1, k2, k3 = st.columns(3, gap="medium")
        k1.markdown(f'<div class="glass-card"><div class="kpi-title">Latest OCF</div><div class="kpi-value" style="font-size:1rem;color:#26A69A;">{format_human_value(latest["Operating Cash Flow"])}</div><div class="kpi-sub">{latest["Period"]}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="glass-card"><div class="kpi-title">Latest CapEx</div><div class="kpi-value" style="font-size:1rem;color:#EF5350;">{format_human_value(latest["CapEx"])}</div><div class="kpi-sub">{latest["Period"]}</div></div>', unsafe_allow_html=True)
        fcf_color = "#26A69A" if pd.notna(latest["Free Cash Flow"]) and latest["Free Cash Flow"] >= 0 else "#EF5350"
        k3.markdown(f'<div class="glass-card"><div class="kpi-title">Latest Free Cash Flow</div><div class="kpi-value" style="font-size:1rem;color:{fcf_color};">{format_human_value(latest["Free Cash Flow"])}</div><div class="kpi-sub">{latest["Period"]}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Portfolio tab
# ---------------------------------------------------------------------------

def render_portfolio_tab():
    if "logged_in_user" not in st.session_state:
        col_form, _ = st.columns([1, 2], gap="large")
        with col_form:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔐 Είσοδος")
            username = st.text_input("Username:", key="login_input")
            if st.button("Σύνδεση", use_container_width=True):
                u = username.strip()
                if u:
                    db_add_user(u); st.session_state.logged_in_user = u; st.rerun()
                else:
                    st.warning("Εισάγετε έγκυρο username.")
            st.markdown("</div>", unsafe_allow_html=True)
        return

    user = st.session_state.logged_in_user
    hdr_l, hdr_r = st.columns([5, 1], gap="medium")
    hdr_l.markdown(f'<span style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;">Portfolio Wealth Terminal</span>&nbsp;&nbsp;<span style="font-weight:700;">{user}</span>', unsafe_allow_html=True)
    if hdr_r.button("Sign out", use_container_width=True):
        del st.session_state.logged_in_user; st.rerun()

    with st.expander("＋ New Transaction", expanded=False):
        with st.form("tx_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4, gap="medium")
            tx_ticker = c1.text_input("Ticker").strip().upper()
            tx_action = c2.selectbox("Side", ["BUY", "SELL"])
            tx_shares = c3.number_input("Qty", min_value=0.0001, step=1.0, format="%.4f")
            tx_price  = c4.number_input("Execution Price ($)", min_value=0.0001, step=0.01, format="%.4f")
            submitted = st.form_submit_button("Submit")
        if submitted:
            if not tx_ticker or tx_shares <= 0 or tx_price <= 0:
                st.error("All fields required.")
            elif tx_action == "SELL":
                ok, msg = validate_sell(db_get_transactions(user), tx_ticker, tx_shares)
                if not ok: st.error(f"⚠️ {msg}")
                else:
                    db_log_transaction(user, tx_ticker, tx_action, tx_shares, tx_price)
                    st.success(f"SELL {tx_shares:.4f} {tx_ticker} @ ${tx_price:.2f} booked."); st.rerun()
            else:
                db_log_transaction(user, tx_ticker, tx_action, tx_shares, tx_price)
                st.success(f"BUY {tx_shares:.4f} {tx_ticker} @ ${tx_price:.2f} booked."); st.rerun()

    txns = db_get_transactions(user)
    if txns.empty:
        st.info("No transactions yet."); return

    from marketLayer5 import fetch_market_snapshot, enrich_metadata, fetch_fundamentals_data
    from DataLayer6 import db_get_metadata
    unique_tickers = txns["ticker"].unique().tolist()
    enrich_metadata(unique_tickers)
    meta_df     = db_get_metadata(unique_tickers)
    market_data = fetch_market_snapshot(unique_tickers)
    port_df     = compute_portfolio_state(txns, market_data)
    if port_df.empty:
        st.info("No open positions."); return

    meta_idx = meta_df.set_index("ticker") if not meta_df.empty else pd.DataFrame()
    port_df["Sector"] = port_df["Ticker"].map(lambda t: str(meta_idx.loc[t, "sector"]) if (not meta_idx.empty and t in meta_idx.index) else "Unknown")
    port_df["7D Trend"] = port_df["Ticker"].map(lambda t: market_data.get(t, {}).get("spark", []))
    
    # ── ΝΕΟ: 52-Week Range Logic ──
    def _calc_range_pos(t):
        d = market_data.get(t, {})
        low, high, curr = d.get("low_52w"), d.get("high_52w"), d.get("current")
        if all(pd.notna(v) for v in [low, high, curr]) and high > low:
            return (curr - low) / (high - low)
        return np.nan
        
    port_df["52W Range"] = port_df["Ticker"].map(_calc_range_pos)
    
    total_val = port_df["Current Value"].dropna().sum()
    port_df["Wt %"] = (port_df["Current Value"] / total_val * 100).round(2) if total_val else np.nan

    total_cost    = port_df["Cost Basis"].sum()
    total_mkt     = port_df["Current Value"].sum()
    total_unreal  = port_df["Unrealized P/L ($)"].sum()
    total_unreal_pct = (total_unreal / total_cost * 100) if total_cost else 0.0
    total_days_gain  = port_df["Day's Gain ($)"].sum()
    total_realized   = port_df["Realized P/L ($)"].sum()

    fundamentals_dict = {}
    with st.spinner("Calculating Portfolio Cash Flows..."):
        for t in unique_tickers:
            fundamentals_dict[t] = fetch_fundamentals_data(t)

    risk = compute_weighted_metrics(
        port_df,
        meta_df,
        snapshot_data=market_data,
        fundamentals_data=fundamentals_dict,
    )
    beta_str  = f"{risk['beta']:.2f}" if pd.notna(risk["beta"]) else "—"
    yield_str = f"{risk['div_yield_pct']:.2f}%" if pd.notna(risk["div_yield_pct"]) else "—"
    fcf_str   = f"{risk['fcf_yield_pct']:.2f}%" if pd.notna(risk.get("fcf_yield_pct")) else "—"
    ocf_str   = f"{risk['ocf_yield_pct']:.2f}%" if pd.notna(risk.get("ocf_yield_pct")) else "—"

    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8, gap="small")
    uc = "var(--green)" if total_unreal >= 0 else "var(--red)"
    dc = "var(--green)" if total_days_gain >= 0 else "var(--red)"
    fcf_val = risk.get("fcf_yield_pct")
    ocf_val = risk.get("ocf_yield_pct")
    fcf_color = "var(--green)" if pd.notna(fcf_val) and fcf_val >= 3 else ("var(--text)" if pd.notna(fcf_val) and fcf_val >= 0 else "var(--red)")
    ocf_color = "var(--green)" if pd.notna(ocf_val) and ocf_val >= 4 else ("var(--text)" if pd.notna(ocf_val) and ocf_val >= 0 else "var(--red)")
    for col, label, value, colour in [
        (k1, "Cost Basis",     f"${total_cost:,.0f}",       "var(--text)"),
        (k2, "Market Value",   f"${total_mkt:,.0f}",        "var(--text)"),
        (k3, "Total P/L",      f"${total_unreal:+,.0f}",    uc),
        (k4, "Day's Gain",     f"${total_days_gain:+,.0f}", dc),
        (k5, "Port Beta",      beta_str,                    "var(--text)"),
        (k6, "Div Yield",      yield_str,                   "var(--text)"),
        (k7, "FCF Yield",      fcf_str,                     fcf_color),
        (k8, "OCF Yield",      ocf_str,                     ocf_color),
    ]:
        col.markdown(f'<div class="glass-card" style="padding:10px;"><div class="kpi-title" style="font-size:0.65rem;">{label}</div><div class="kpi-value" style="font-size:.9rem;color:{colour};">{value}</div></div>', unsafe_allow_html=True)

    if abs(total_realized) > 0.005:
        rc = "var(--green)" if total_realized >= 0 else "var(--red)"
        st.markdown(f'<div style="padding:.3rem .8rem;border-radius:4px;border:1px solid {rc}33;background:{rc}11;color:{rc};font-size:.78rem;font-weight:600;display:inline-block;margin-bottom:.5rem;">Realized P/L (FIFO): ${total_realized:+,.2f}</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 2.8], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Allocation</div>', unsafe_allow_html=True)
            donut_src = port_df[port_df["Current Value"].notna() & (port_df["Current Value"] > 0)][["Ticker", "Current Value", "Sector"]].copy()
            if not donut_src.empty:
                donut_src["Weight (%)"] = (donut_src["Current Value"] / donut_src["Current Value"].sum() * 100).round(1)
                donut = alt.Chart(donut_src).mark_arc(innerRadius=68, stroke="#1C2030", strokeWidth=2).encode(
                    theta=alt.Theta("Current Value:Q", stack=True),
                    color=alt.Color(
                        "Ticker:N",
                        scale=alt.Scale(scheme="tableau10"),
                        legend=alt.Legend(title=None, orient="bottom", columns=2, labelFontSize=10),
                    ),
                    tooltip=[
                        alt.Tooltip("Ticker:N"),
                        alt.Tooltip("Current Value:Q", title="Value ($)", format="$,.2f"),
                        alt.Tooltip("Weight (%):Q", format=".1f"),
                        alt.Tooltip("Sector:N"),
                    ],
                )
                st.altair_chart(apply_chart_theme(donut, height=300), use_container_width=True)

                st.markdown('<div class="section-label" style="margin-top:1.5rem;">Sector Exposure</div>', unsafe_allow_html=True)
                sector_src = donut_src.groupby("Sector", as_index=False)["Current Value"].sum().sort_values("Current Value", ascending=True)
                sector_chart = alt.Chart(sector_src).mark_bar(
                    color="#2962FF",
                    opacity=0.9,
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                ).encode(
                    x=alt.X("Current Value:Q", title=None, axis=None),
                    y=alt.Y("Sector:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("Sector:N"), alt.Tooltip("Current Value:Q", format="$,.2f")],
                )
                st.altair_chart(apply_chart_theme(sector_chart, height=max(120, len(sector_src) * 38)), use_container_width=True)

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Open Positions</div>', unsafe_allow_html=True)
            
            def _pl_colour(val):
                if not isinstance(val, (int, float)) or np.isnan(val): return ""
                return "color: #26A69A; font-weight:600;" if val >= 0 else "color: #EF5350; font-weight:600;"
                
            ledger_df = port_df[[
                "Ticker", "Sector", "Shares", "Avg Cost", "Current Price", 
                "Current Value", "Wt %", "Unrealized P/L ($)", "Unrealized P/L (%)", 
                "Day's Gain ($)", "Day's Gain (%)", "7D Trend", "52W Range"
            ]].copy()
            
            ledger_df = ledger_df.rename(columns={
                "Shares": "Qty", "Avg Cost": "Cost", "Current Price": "Last", 
                "Current Value": "Value", "Unrealized P/L ($)": "P/L ($)", 
                "Unrealized P/L (%)": "P/L (%)", "Day's Gain ($)": "Day ($)", 
                "Day's Gain (%)": "Day (%)"
            })

            st.dataframe(
                ledger_df.style.map(_pl_colour, subset=["P/L ($)", "P/L (%)", "Day ($)", "Day (%)"]),
                hide_index=True,
                use_container_width=True,
                height=480,
                column_config={
                    "Qty": st.column_config.NumberColumn(format="%.4f"),
                    "Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Last": st.column_config.NumberColumn(format="$%.2f"),
                    "Value": st.column_config.NumberColumn(format="$%,.2f"),
                    "Wt %": st.column_config.NumberColumn(format="%.2f%%"),
                    "P/L ($)": st.column_config.NumberColumn(format="$+,.2f"),
                    "P/L (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Day ($)": st.column_config.NumberColumn(format="$+,.2f"),
                    "Day (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "7D Trend": st.column_config.LineChartColumn("7D Trend", y_min=0),
                    "52W Range": st.column_config.ProgressColumn(
                        "52W Range",
                        help="Position relative to 52-week Low/High",
                        min_value=0,
                        max_value=1,
                        format=" "
                    ),
                }
            )

    # ── Lower Asset Detail Row ──
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    st.markdown("### Selected Asset Details")
    selected_ticker = st.selectbox("Select Holding", options=unique_tickers, key="portfolio_holding_select")
    from marketLayer5 import fetch_fundamentals_data
    selected_fundamentals = fetch_fundamentals_data(selected_ticker)
    
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-label">Selected Asset Statistics</div>', unsafe_allow_html=True)
            ticker_symbol = selected_ticker
            ref_data = fetch_range_reference_data(ticker_symbol)
            
            if not ref_data.empty:
                latest = ref_data.iloc[-1]
                low_52 = ref_data["Low"].min()
                high_52 = ref_data["High"].max()
                curr = latest["Close"]
                
                sc1, sc2 = st.columns(2)
                with sc1: render_range_bar("Day's Range", latest["Low"], latest["High"], curr, color="#2962FF")
                with sc2: render_range_bar("52-Week Range", low_52, high_52, curr, color="#F59E0B")
                
                fund = selected_fundamentals.get("snapshot", {})
                
                st.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)
                stats = [
                    ("Prev Close", f"${latest.get('Open', 0):,.2f}"),
                    ("Market Cap", format_human_value(fund.get("market_cap", 0))),
                    ("P/E (TTM)", f"{fund.get('trailing_pe', 0):.2f}x"),
                    ("EPS (TTM)", f"{fund.get('trailing_eps', 0):.2f}"),
                ]
                for lbl, val in stats:
                    st.markdown(f'<div class="stat-row" style="display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05);"><span style="color:#8ea0b8;font-size:0.88rem;">{lbl}</span><span style="font-weight:600;font-size:0.95rem;">{val}</span></div>', unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-label">Earnings & Revenue Performance</div>', unsafe_allow_html=True)
            ticker_symbol = selected_ticker
            render_earnings_dots(ticker_symbol)
            build_rev_earning_chart(selected_fundamentals.get("annual"))
            st.markdown('<div style="height:15px;"></div>', unsafe_allow_html=True)

    render_cash_flow_analysis(selected_ticker, selected_fundamentals)

    with st.expander("📋 Transaction Ledger", expanded=False):
        tx_label_map = {int(r.id): f"#{int(r.id):>4} │ {r.action:<4} │ {r.ticker:<6} │ {float(r.shares):.4f} @ ${float(r.execution_price):.2f} │ {r.timestamp}" for r in txns.itertuples(index=False)}
        tx_display = txns[["id","ticker","action","shares","execution_price","timestamp"]].copy()
        tx_display.columns = ["ID","Ticker","Side","Qty","Price ($)","Timestamp"]
        st.dataframe(tx_display.style.format({"Qty":"{:.4f}","Price ($)":"${:.2f}"}), hide_index=True, use_container_width=True, height=220)
        st.caption("⚠️ Deletion is irreversible.")
        del_id = st.selectbox("Select transaction to delete:", options=list(tx_label_map.keys()), format_func=lambda x: tx_label_map[x])
        if st.button("Delete Entry", type="primary"):
            db_delete_transaction(del_id); st.toast(f"Transaction #{del_id} deleted."); st.rerun()


# ---------------------------------------------------------------------------
# Render: Advisory tab
# ---------------------------------------------------------------------------

def render_advisory_tab(analysis=None):
    if "logged_in_user" not in st.session_state:
        st.info("Παρακαλώ συνδεθείτε μέσω του Portfolio tab."); return

    txns = db_get_transactions(st.session_state.logged_in_user)
    if txns.empty:
        st.info("Προσθέστε συναλλαγές για να δείτε institutional risk analysis."); return

    st.markdown('<div class="section-label">Institutional Advisory & Risk Report</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.9rem;color:var(--muted);margin-bottom:20px;">Hedge-fund grade: CVaR, Harmonic P/E, ERP, Stress Testing, Rolling Correlations.</div>', unsafe_allow_html=True)

    repo = LiveYFinanceRepository()
    pipeline = run_portfolio_pipeline(txns, repo, analysis=analysis)
    port_df   = pipeline["port_df"]
    quant     = pipeline["quant"]
    valuation = pipeline["valuation"]
    total_val = pipeline["total_val"]
    insights  = pipeline["insights"]
    earnings_notes = pipeline.get("earnings_notes", [])

    # ── Earnings Pulse UI (Aspect 10) ──
    if earnings_notes:
        with st.container(border=True):
            st.markdown('<div class="section-label">📊 Holdings Earnings Pulse</div>', unsafe_allow_html=True)
            for note in earnings_notes:
                st.markdown(f"""
                <div style="padding:10px; background:rgba(41,98,255,0.05); border-left:3px solid #2962FF; border-radius:4px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#D1D4DC;">{note['ticker']}</span>
                        <span style="font-size:0.65rem; color:var(--muted);">{note['timestamp']}</span>
                    </div>
                    <div style="font-size:0.8rem; color:#D1D4DC; margin-top:4px;">{note['note']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("View Full Journal History", use_container_width=True):
                st.session_state.show_journal = True
                
    if st.session_state.get("show_journal", False):
        with st.expander("📖 Earnings Intelligence Journal", expanded=True):
            from DataLayer6 import db_get_earnings_journal
            journal_df = db_get_earnings_journal(limit=20)
            if not journal_df.empty:
                st.dataframe(journal_df, use_container_width=True, hide_index=True)
            else:
                st.info("No journal entries yet.")
            if st.button("Close Journal"):
                st.session_state.show_journal = False
                st.rerun()
    constraints = pipeline.get("constraints", {})

    if port_df.empty:
        st.info("Δεν βρέθηκαν ενεργές θέσεις."); return

    # ── PhD Evaluation & Smart Rebalancer ──────────────────────────────────
    if constraints:
        kills = constraints.get("kill_switches", [])
        if kills:
            st.markdown('<div class="section-label" style="margin-top:8px;">🛑 Layer 1: Kill Switches</div>', unsafe_allow_html=True)
            for k in kills:
                st.error(k, icon="🚨")
                
        alloc_df = constraints.get("allocations", pd.DataFrame())
        if not isinstance(alloc_df, pd.DataFrame):
            alloc_df = pd.DataFrame()
            
        if not alloc_df.empty:
            with st.container(border=True):
                st.markdown('<div class="section-label" style="margin-top:12px;">🧠 Layer 3: Risk-Adjusted Allocation Engine</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.78rem;color:var(--muted);margin-bottom:8px;">Ανάλυση Gradients και Score για βέλτιστη κατανομή Target Weights βάσει Priority.</div>', unsafe_allow_html=True)
                
                def _color_drift(val):
                    if pd.isna(val): return ""
                    try:
                        v = float(str(val).replace("%", "").replace("+", ""))
                        return "color: #26a69a;" if v > 2.5 else "color: #ef5350;" if v < -2.5 else "color: #8ea0b8;"
                    except: return ""
                
                def _color_score(val):
                    if pd.isna(val): return ""
                    try:
                        v = float(str(val).split("/")[0])
                        return "color: #26a69a; font-weight:700;" if v >= 75 else "color: #ef5350; font-weight:700;" if v <= 35 else ""
                    except: return ""
                    
                fmt = {"Conviction": "{:.1f}", "Current Wt (%)": "{:.1f}%", "Target Wt (%)": "{:.1f}%", "Drift (%)": "{:+.1f}%"}
                styler = alloc_df.style.format(fmt)
                styler = styler.map(_color_drift, subset=["Drift (%)"]).map(_color_score, subset=["Conviction"])
                st.dataframe(styler, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label" style="margin-top:8px;">📊 Risk Engine</div>', unsafe_allow_html=True)
    with st.container(border=True):
        risk_data = analysis.get("quant", {})
        var_pct  = risk_data.get("var_95_pct", np.nan)
        var_abs  = risk_data.get("var_95", np.nan)
        cvar_pct = risk_data.get("cvar_95_pct", np.nan)
        cvar_abs = risk_data.get("cvar_95", np.nan)

        r1, r2, r3, r4, r5 = st.columns(5, gap="small")
        
        ewma_vol = risk_data.get("ewma_vol", risk_data.get("ewma_volatility_annual", risk_data.get("portfolio_vol_annual", np.nan)))
        rolling_alert = risk_data.get("rolling_corr_alert", False)
        n_high_corr = len(risk_data.get("correlation_warnings", []))
        macro_score = risk_data.get("macro_context", {}).get("spy_macro_climate_score", 0.5)

        ewma_tip = "EWMA (Εκθετικός Κινητός Μέσος): Δίνει μεγαλύτερη βαρύτητα στις πρόσφατες μέρες. Έτσι, αν η αγορά έγινε απότομα ασταθής χθες, το νούμερο αυτό θα αντιδράσει πιο γρήγορα από το απλό ιστορικό Volatility."
        corr_tip = "Συσχέτιση (Correlation): Δείχνει αν οι μετοχές σας κινούνται όλες μαζί (υψηλό ρίσκο) ή ανεξάρτητα (καλή διαφοροποίηση). Κόκκινο σημαίνει ότι μια πτώση θα παρασύρει όλο το χαρτοφυλάκιο."
        macro_tip = "Macro Risk: Αξιολογεί πόσο εκτεθειμένο είναι το χαρτοφυλάκιο στις τρέχουσες πιέσεις των επιτοκίων και του πληθωρισμού."

        with r1: st.markdown(f'<div class="glass-card"><div class="kpi-title" style="font-size:0.65rem;">VaR 95% (1-Day)</div><div class="kpi-value" style="font-size:0.95rem;color:#EF5350;">{"${:,.0f}".format(var_abs) if pd.notna(var_abs) else "N/A"}</div><div class="kpi-delta">{f"-{var_pct:.2f}%" if pd.notna(var_pct) else ""}</div></div>', unsafe_allow_html=True)
        with r2: st.markdown(f'<div class="glass-card"><div class="kpi-title" style="font-size:0.65rem;">CVaR / Exp. Shortfall</div><div class="kpi-value" style="font-size:0.95rem;color:#EF5350;">{"${:,.0f}".format(cvar_abs) if pd.notna(cvar_abs) else "N/A"}</div><div class="kpi-delta">{f"-{cvar_pct:.2f}%" if pd.notna(cvar_pct) else ""}</div></div>', unsafe_allow_html=True)
        with r3: st.markdown(f'<div class="glass-card"><div class="kpi-title" style="font-size:0.65rem; cursor:help;" title="{ewma_tip}">EWMA Volatility ❔</div><div class="kpi-value" style="font-size:0.95rem;">{f"{ewma_vol:.1f}%" if pd.notna(ewma_vol) else "N/A"}</div></div>', unsafe_allow_html=True)

        with r4:
            corr_color = "#EF5350" if (n_high_corr > 0 or rolling_alert) else "#26A69A"
            corr_label = f"⚠️ {n_high_corr} ζεύγη" if n_high_corr else "✅ Διαφοροποιημένο"
            st.markdown(f'<div class="glass-card"><div class="kpi-title" style="font-size:0.65rem; cursor:help;" title="{corr_tip}">Correlation ❔</div><div class="kpi-value" style="font-size:0.85rem;color:{corr_color};">{corr_label}</div></div>', unsafe_allow_html=True)

        with r5:
            m_color = "#EF5350" if macro_score < 0.4 else "#26A69A"
            st.markdown(f'<div class="glass-card"><div class="kpi-title" style="font-size:0.65rem; cursor:help;" title="{macro_tip}">Macro Risk ❔</div><div class="kpi-value" style="font-size:0.85rem;color:{m_color};">{"Υψηλό" if macro_score < 0.4 else "Κανονικό"}</div></div>', unsafe_allow_html=True)

        explain_cols = st.columns(2, gap="medium")
        with explain_cols[0]:
            st.markdown(
                f'<div class="commentary-card"><strong>VaR 95%</strong>: Αν οι συνθήκες μείνουν περίπου όπως σήμερα, αυτό είναι το ποσό που μπορεί να χαθεί μέσα σε μία δύσκολη ημέρα στις περισσότερες περιπτώσεις. '
                f'Εδώ δείχνει περίπου <strong>{"${:,.0f}".format(var_abs) if pd.notna(var_abs) else "N/A"}</strong>. '
                f'Το <strong>CVaR</strong> πηγαίνει ένα βήμα πιο πέρα: δεν κοιτά μόνο το κακό σενάριο, αλλά τον μέσο όρο των πολύ κακών ημερών. '
                f'Γι’ αυτό το CVaR είναι συνήθως χειρότερο από το VaR και δείχνει τι μπορεί να συμβεί όταν τα πράγματα ξεφεύγουν.</div>',
                unsafe_allow_html=True,
            )
        with explain_cols[1]:
            st.markdown(
                f'<div class="commentary-card"><strong>EWMA Volatility</strong>: δείχνει πόσο νευρική είναι τελευταία η αγορά και αντιδρά πιο γρήγορα στις πρόσφατες κινήσεις. '
                f'<strong>Correlation</strong>: δείχνει αν οι θέσεις σου πέφτουν ή ανεβαίνουν όλες μαζί. '
                f'Όταν η συσχέτιση ανεβαίνει, χάνεται η διαφοροποίηση. '
                f'<strong>Macro Risk</strong>: δείχνει πόσο ευάλωτο είναι το χαρτοφυλάκιο σε επιτόκια, πληθωρισμό και γενικό market regime.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Τι σημαίνουν απλά οι δείκτες του Risk Engine;"):
            st.markdown(
                """
                - `VaR 95%`: ένα πρακτικό όριο για το πόσα μπορεί να χάσεις σε μια κακή ημέρα υπό φυσιολογική πίεση αγοράς.
                - `CVaR / Expected Shortfall`: τι χάνεις κατά μέσο όρο όταν η ημέρα είναι ακόμη χειρότερη από το VaR.
                - `EWMA Volatility`: πόσο έντονες είναι οι πρόσφατες διακυμάνσεις. Όσο ανεβαίνει, τόσο πιο νευρική γίνεται η θέση.
                - `Correlation`: αν οι θέσεις κινούνται όλες μαζί, το χαρτοφυλάκιο γίνεται πιο εύθραυστο.
                - `Macro Risk`: αν το χαρτοφυλάκιο είναι ευαίσθητο σε rate shocks, inflation pressure ή αλλαγή οικονομικού καθεστώτος.
                """
            )

        if "risk_df" in analysis and not analysis["risk_df"].empty:
            st.markdown('<div class="section-label" style="margin-top:15px; font-size: 0.9rem;">🎯 Risk Decomposition (MCR) & Context</div>', unsafe_allow_html=True)
            st.dataframe(
                analysis["risk_df"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Asset", width="small"),
                    "Weight (%)": st.column_config.NumberColumn("Capital Wt", format="%.1f%%"),
                    "Volatility (%)": st.column_config.NumberColumn("Asset Vol", format="%.1f%%"),
                    "Risk Contribution (%)": st.column_config.NumberColumn("Risk Contrib (RC)", format="%.1f%%"),
                    "52w Position (%)": st.column_config.ProgressColumn(
                        "52w Range (Context)",
                        help="0% = 52w Low, 100% = 52w High",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100
                    )
                }
            )

    # ── Valuation KPIs ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px;">📈 Valuation Engine</div>', unsafe_allow_html=True)
    h_pe = valuation.get("harmonic_forward_pe", np.nan)
    a_pe = valuation.get("arithmetic_forward_pe", np.nan)
    spy_pe = valuation.get("spy_forward_pe", np.nan)
    bench_name = valuation.get("benchmark_name", "SPY")
    erp  = valuation.get("equity_risk_premium_pct", np.nan)
    ey   = valuation.get("earnings_yield_pct", np.nan)
    rf   = valuation.get("risk_free_rate_pct", np.nan)
    erp_alert = valuation.get("erp_alert", False)
    bias = valuation.get("pe_bias_pct", np.nan)
    v1, v2, v3, v4 = st.columns(4, gap="medium")
    v1.metric("Harmonic Forward P/E", f"{h_pe:.1f}x" if pd.notna(h_pe) else "N/A", help=f"Σωστός Αρμονικός Μέσος. Arithmetic = {a_pe:.1f}x (+{bias:.1f}% bias)" if pd.notna(a_pe) and pd.notna(bias) else "Harmonic Mean Forward P/E")
    v2.metric(f"{bench_name} Forward P/E", f"{spy_pe:.1f}x" if pd.notna(spy_pe) else "N/A", delta=f"{h_pe-spy_pe:+.1f}x spread" if (pd.notna(h_pe) and pd.notna(spy_pe)) else None)
    v3.metric("Earnings Yield", f"{ey:.2f}%" if pd.notna(ey) else "N/A", help="1 / Forward P/E")
    v4.metric(f"{'🔴' if erp_alert else '🟢'} Equity Risk Premium", f"{erp:.2f}%" if pd.notna(erp) else "N/A", delta=f"RF: {rf:.2f}%" if pd.notna(rf) else None)

    # ── Stress Testing ─────────────────────────────────────────────────────
    stress = quant.get("stress_results", {})
    scenarios = stress.get("scenarios", {})
    if scenarios:
        st.markdown('<div class="section-label" style="margin-top:20px;">🧪 Stress Testing</div>', unsafe_allow_html=True)
        with st.container(border=True):
            sc_cols = st.columns(len(scenarios), gap="medium")
            for col, (sc_name, sc_data) in zip(sc_cols, scenarios.items()):
                pnl, pct = sc_data["pnl"], sc_data["pct"]
                color = "#26A69A" if pnl >= 0 else "#EF5350"
                border_color = "#26A69A" if pnl >= 0 else "#EF5350"
                # Task 5: Stress test cards with dynamic border and tooltip
                col.markdown(f'<div class="stress-card" title="{sc_data["description"]}" style="border-left:3px solid {border_color};"><div class="kpi-title" style="font-size:0.72rem;">{sc_name.split("(")[0].strip()}</div><div class="kpi-value" style="font-size:1rem;color:{color};">${pnl:+,.0f}</div><div class="kpi-delta">{pct:+.1f}%</div></div>', unsafe_allow_html=True)

    # ── Interactive Macro Scenario Engine ──────────────────────────────────
    import sys
    # Fetch returns logic normally from marketLayer5
    from marketLayer5 import fetch_historical_returns
    from riskengine8_2 import simulate_dynamic_macro_shock as simulate_macro_shock

    st.markdown('<div class="section-label" style="margin-top:30px;">🌍 Interactive Macro Engine</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:var(--muted);margin-bottom:10px;">Υπολογισμός σε πραγματικό χρόνο της ζημίας/κέρδους των θέσεών σας βάσει 1y Beta & Correlation όταν το Benchmark μεταβάλλεται.</div>', unsafe_allow_html=True)

    unique_tickers = list(port_df["Ticker"].unique()) if not port_df.empty else []
    returns_df = fetch_historical_returns(unique_tickers)
    bench_ticker = st.session_state.get("benchmark", "SPY").upper()

    shock_val = st.slider(f"Macro Shock Slider: {bench_ticker} Return (%)", min_value=-50, max_value=50, value=-10, step=1)
    impact = simulate_macro_shock(port_df, returns_df, benchmark=bench_ticker, shock_pct=shock_val)
    
    if impact and "total_simulated_value" in impact:
        simulated_pnl = impact["total_expected_pl"]
        total_sim = impact["total_simulated_value"]
        # Task 5: Dynamic color for Macro result
        pnl_color = "#EF5350" if simulated_pnl < 0 else "#26A69A"
        with st.container(border=True):
            st.markdown(f'<div class="macro-impact"><span style="font-size:1rem;color:var(--muted);">Simulated Portfolio Value: </span><span style="font-size:1.45rem;font-weight:700;">${total_sim:,.2f}</span> <span style="color:{pnl_color};font-size:1.25rem;font-weight:700;margin-left:10px;">${simulated_pnl:+,.2f}</span></div>', unsafe_allow_html=True)
            chart_data = []
            for t, data in impact["asset_impacts"].items():
                if data["current_value"] > 0:
                    chart_data.append({"Asset": t, "P/L ($)": data["pl_impact"], "Drop (%)": data["implied_shock_pct"]})
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                # Task 5: Bar chart cornerRadius and smooth animation
                bars = alt.Chart(chart_df).mark_bar(cornerRadius=2).encode(
                    x=alt.X("P/L ($):Q", title="Portfolio Impact ($)"),
                    y=alt.Y("Asset:N", sort=alt.EncodingSortField(field="P/L ($)", order="ascending"), title=None),
                    color=alt.condition(alt.datum["P/L ($)"] > 0, alt.value("#26A69A"), alt.value("#EF5350")),
                    tooltip=["Asset", alt.Tooltip("P/L ($):Q", format="$,.2f"), alt.Tooltip("Drop (%):Q", format=".2f")],
                )
                st.altair_chart(apply_chart_theme(bars, height=max(150, len(chart_data) * 35)), use_container_width=True)

    # ── Rolling Correlations ───────────────────────────────────────────────
    rolling_summary = quant.get("rolling_corr_current", {})
    rolling_alert = quant.get("rolling_corr_alert", False)
    if rolling_summary:
        with st.expander("🔄 Rolling 30-Day Correlation Details", expanded=rolling_alert):
            rc_df = pd.DataFrame([{"Ζεύγος": k, "30d Corr": v["current_30d"], "Ιστορικό Avg": v["historical_avg"], "Δ": round(v["current_30d"] - v["historical_avg"], 3)} for k, v in rolling_summary.items()]).sort_values("30d Corr", ascending=False)
            st.dataframe(rc_df, hide_index=True, use_container_width=True)

    # === Deep Analysis: Co-integration & Relative Performance (Moved from rail) ===
    st.markdown('<div class="section-label" style="margin-top:30px;">📉 Benchmark Sensitivity & Co-integration</div>', unsafe_allow_html=True)
    benchmark_note = build_benchmark_sensitivity_text(analysis)
    st.markdown(f'<div class="commentary-card">{benchmark_note if benchmark_note else "Δεν υπάρχουν διαθέσιμα στοιχεία co-integration."}</div>', unsafe_allow_html=True)

    rel_df = pd.DataFrame({
        "Date": analysis["normalized_stock"].index, 
        "Stock": analysis["normalized_stock"].values, 
        "Benchmark": analysis["normalized_benchmark"].values
    })
    st.altair_chart(apply_chart_theme(build_dual_line_chart(rel_df, "Relative Performance vs Benchmark"), height=300), use_container_width=True)

    # ── Interactive Macro Scenario Matrix ──────────────────────────────────
    render_macro_prudential_matrix(analysis)

    # ── Advisory Insights ──────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:30px;">💼 Senior Analyst Advisory</div>', unsafe_allow_html=True)
    if insights:
        for insight in insights:
            st.info(insight, icon="💡")
    else:
        st.markdown("Δεν υπάρχουν triggers αυτή τη στιγμή.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Quant Terminal", page_icon=":bar_chart:", layout="wide")
    inject_styles()
    init_portfolio_db()

    with st.sidebar:
        st.markdown("### Controls")
        ticker     = st.text_input("Ticker",    value="NVDA").strip().upper() or "NVDA"
        benchmark  = st.text_input("Benchmark", value="QQQ").strip().upper()  or "QQQ"
        beta_window = int(st.number_input("Rolling Beta Window", min_value=20, max_value=252, value=60, step=5))
        run = st.button("Run Dashboard", width="stretch")

    if "run_dashboard" not in st.session_state:
        st.session_state.run_dashboard = True
    if run:
        st.session_state.update(run_dashboard=True, ticker=ticker, benchmark=benchmark, beta_window=beta_window)

    ticker     = st.session_state.get("ticker",     ticker)
    benchmark  = st.session_state.get("benchmark",  benchmark)
    beta_window = st.session_state.get("beta_window", beta_window)

    render_hero()
    if not st.session_state.run_dashboard:
        st.stop()

    with st.spinner(f"Loading {ticker}…"):
        try:
            repo = LiveYFinanceRepository()
            analysis = build_analysis(ticker, benchmark, beta_window, repo)
        except ValueError as e:
            st.error(f"Σφάλμα δεδομένων: {e}"); st.stop()
        except Exception as e:
            st.error(f"Απρόσμενο σφάλμα κατά την επικοινωνία με το Data Feed: {str(e)}")
            st.exception(e) # Show full traceback in UI for debugging
            st.stop()

    left, right = st.columns([0.75, 2.25], gap="large")
    with left:
        render_kpi_rail(analysis["valuation_cards"])
        render_benchmark_multiples(benchmark, analysis["benchmark_financials"])
    with right:
        render_secondary_metrics(analysis)
        tabs = st.tabs(["Overview", "Market Pulse", "Financials", "Live Briefing", "📂 Portfolio", "Advisory & Risk"])
        with tabs[0]: render_overview_tab(analysis)
        with tabs[1]: render_market_pulse_tab(ticker, analysis)
        with tabs[2]: render_financials_tab(analysis)
        with tabs[3]: render_live_briefing_tab(analysis)
        with tabs[4]: render_portfolio_tab()
        with tabs[5]: render_advisory_tab(analysis)


def render_macro_prudential_matrix(analysis=None):
    st.markdown('<div class="section-label" style="margin-top:40px;">🌐 GENERAL MONETARY IMPACT MATRIX</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:0.85rem;color:#8ea0b8;margin-bottom:20px;">
            Auto macro state engine με trailing 12m path. Το panel δεν ζητά πια χειροκίνητα assumptions,
            αλλά διαβάζει yields, inflation proxies και growth proxies για να δώσει investor-facing context.
        </div>
        """,
        unsafe_allow_html=True,
    )

    macro = (analysis or {}).get("macro_report", {}) if isinstance(analysis, dict) else {}
    if not macro:
        try:
            from marketLayer5 import build_macro_environment_report
            macro = build_macro_environment_report()
        except Exception:
            macro = {}

    theme = macro.get("theme", "Macro data unavailable")
    x_val = float(macro.get("x_growth_score", 0.0) or 0.0)
    y_val = float(macro.get("y_inflation_score", 0.0) or 0.0)
    data_quality = str(macro.get("data_quality", "minimal")).upper()
    summary_lines = macro.get("summary_lines", []) or []
    investor_takeaway = macro.get("investor_takeaway", "Δεν υπάρχει ακόμη επαρκές macro interpretation.")
    path_points = macro.get("macro_vector_12m", []) or []

    if y_val >= 0 and x_val < 0:
        adv_color = "#EF5350"
        adv_bg = "rgba(239, 83, 80, 0.08)"
        adv_title = "STAGFLATION RISK"
    elif y_val >= 0 and x_val >= 0:
        adv_color = "#F59E0B"
        adv_bg = "rgba(245, 158, 11, 0.08)"
        adv_title = "LATE-CYCLE TIGHTENING"
    elif y_val < 0 and x_val >= 0:
        adv_color = "#26A69A"
        adv_bg = "rgba(38, 166, 154, 0.08)"
        adv_title = "DISINFLATIONARY GROWTH"
    else:
        adv_color = "#42A5F5"
        adv_bg = "rgba(66, 165, 245, 0.08)"
        adv_title = "DEFENSIVE SLOWDOWN"

    with st.container(border=True):
        col_text, col_chart = st.columns([1.05, 1.95], gap="large")

        with col_text:
            st.markdown('<h3 style="font-size:1.05rem; margin-bottom: 12px; color:#ffffff;">Current Macro Theme</h3>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background-color:{adv_bg}; border-left: 3px solid {adv_color}; padding: 12px; border-radius: 6px; margin-bottom: 14px;">
                    <div style="color:{adv_color}; font-size:0.78rem; font-weight:800; letter-spacing:0.06em;">{adv_title}</div>
                    <div style="color:#FFFFFF; font-size:1rem; font-weight:700; margin-top:4px;">{theme}</div>
                    <div style="color:#8ea0b8; font-size:0.78rem; margin-top:6px;">Data quality: {data_quality}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_cols = st.columns(2, gap="small")
            metric_cols[0].markdown(f'<div class="glass-card"><div class="kpi-title">10Y Yield</div><div class="kpi-value" style="font-size:1rem;">{macro.get("yield_10y_current", 0.0):.2f}%</div><div class="kpi-sub">12m Δ: {macro.get("yield_10y_12m_change_bps", 0.0):+.0f} bps</div></div>', unsafe_allow_html=True)
            metric_cols[1].markdown(f'<div class="glass-card"><div class="kpi-title">SPY 12M</div><div class="kpi-value" style="font-size:1rem;">{macro.get("spy_12m_return", 0.0):+.1f}%</div><div class="kpi-sub">3m Δ: {macro.get("spy_3m_return", 0.0):+.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown('<div style="margin-top:14px; font-size:0.8rem; color:#8ea0b8; text-transform:uppercase; letter-spacing:0.05em;">What The Engine Reads</div>', unsafe_allow_html=True)
            for line in summary_lines[:3]:
                st.markdown(f'<div style="margin-top:8px; color:#D1D4DC; font-size:0.84rem; line-height:1.5;">• {line}</div>', unsafe_allow_html=True)

            st.markdown('<div style="margin-top:16px; font-size:0.8rem; color:#8ea0b8; text-transform:uppercase; letter-spacing:0.05em;">Investor Reading</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="commentary-card" style="margin-top:8px;">{investor_takeaway}</div>', unsafe_allow_html=True)

        with col_chart:
            bounds = pd.DataFrame([
                {"x1": -20, "x2": 0, "y1": 0, "y2": 20, "fill": "#7f1d1d"},
                {"x1": 0, "x2": 20, "y1": 0, "y2": 20, "fill": "#7c2d12"},
                {"x1": -20, "x2": 0, "y1": -20, "y2": 0, "fill": "#1e3a8a"},
                {"x1": 0, "x2": 20, "y1": -20, "y2": 0, "fill": "#064e3b"},
            ])
            label_data = pd.DataFrame([
                {"x": -10, "y": 10, "label": "STAGFLATION"},
                {"x": 10, "y": 10, "label": "OVERHEATING"},
                {"x": -10, "y": -10, "label": "DEFLATION"},
                {"x": 10, "y": -10, "label": "GOLDILOCKS"},
            ])
            axis_anchor = pd.DataFrame([{"x": 0, "y": 0}])
            point_data = pd.DataFrame([{
                "x": x_val,
                "y": y_val,
                "Theme": theme,
                "Growth Score": x_val,
                "Inflation Score": y_val,
            }])
            path_df = pd.DataFrame(path_points)

            rects = alt.Chart(bounds).mark_rect(opacity=0.16).encode(
                x=alt.X("x1:Q", scale=alt.Scale(domain=[-20, 20]), title="Growth / Output Impulse"),
                x2="x2:Q",
                y=alt.Y("y1:Q", scale=alt.Scale(domain=[-20, 20]), title="Inflation / Rate Pressure"),
                y2="y2:Q",
                color=alt.Color("fill:N", scale=None, legend=None),
            )
            vline = alt.Chart(axis_anchor).mark_rule(color="#3A4757", strokeWidth=2).encode(x="x:Q")
            hline = alt.Chart(axis_anchor).mark_rule(color="#3A4757", strokeWidth=2).encode(y="y:Q")
            labels = alt.Chart(label_data).mark_text(fontSize=13, color="#8ea0b8", fontWeight=600, opacity=0.72).encode(
                x="x:Q", y="y:Q", text="label:N"
            )

            layers = [rects, vline, hline, labels]
            if not path_df.empty:
                path_line = alt.Chart(path_df).mark_line(color="#D1D4DC", strokeWidth=2.2, point=False).encode(
                    x="x:Q", y="y:Q", order=alt.Order("label:N")
                )
                path_nodes = alt.Chart(path_df).mark_point(size=70, color="#9FB3C8", filled=True).encode(
                    x="x:Q", y="y:Q", tooltip=["label:N", alt.Tooltip("x:Q", format=".1f", title="Growth"), alt.Tooltip("y:Q", format=".1f", title="Inflation")]
                )
                layers.extend([path_line, path_nodes])

            glow = alt.Chart(point_data).mark_circle(size=900, opacity=0.28, color=adv_color).encode(x="x:Q", y="y:Q")
            core = alt.Chart(point_data).mark_circle(size=180, opacity=1.0, color=adv_color, stroke="#FFFFFF", strokeWidth=2.4).encode(
                x="x:Q", y="y:Q",
                tooltip=[
                    alt.Tooltip("Theme:N", title="Theme"),
                    alt.Tooltip("Growth Score:Q", format=".1f"),
                    alt.Tooltip("Inflation Score:Q", format=".1f"),
                ],
            )
            layers.extend([glow, core])

            final_chart = alt.layer(*layers).properties(height=430, background="#000000").configure_axis(
                gridColor="rgba(255,255,255,0.05)",
                domainColor="#3A4757",
                tickColor="#3A4757",
                labelColor="#8A8F9A",
                titleColor="#8A8F9A",
                titlePadding=15,
                labelFontSize=12,
                titleFontSize=13,
            ).configure_view(strokeWidth=0)

            st.altair_chart(final_chart, use_container_width=True)
            st.markdown('<div style="margin-top:10px; color:#8ea0b8; font-size:0.78rem;">Το path δείχνει πού μετακινήθηκε το macro regime στο trailing 12μηνο, ενώ η φωτεινή κουκκίδα δείχνει το current state.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
