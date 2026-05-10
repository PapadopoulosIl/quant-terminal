import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import format_human_value, format_percent, format_ratio
from marketLayer5 import fetch_earnings_dates

THEME = {
    "bg": "#131722",
    "panel": "#1C2030",
    "panel_2": "#1E2538",
    "border": "#2A2E39",
    "text": "#D1D4DC",
    "muted": "#787B86",
    "primary": "#2962FF",
    "accent": "#F59E0B",
    "success": "#26A69A",
    "danger": "#EF5350",
    "benchmark": "#72F1FF",
}

ALTAIR_CONFIG = alt.Config(
    background=THEME["bg"],
    axis=alt.AxisConfig(
        labelColor=THEME["muted"],
        titleColor=THEME["muted"],
        gridColor=THEME["border"]
    ),
    legend=alt.LegendConfig(
        labelColor=THEME["text"],
        titleColor=THEME["text"]
    )
)

def apply_chart_theme(chart: alt.Chart, height: int | None = None) -> alt.Chart:
    themed = chart.configure(**ALTAIR_CONFIG.to_dict()).configure_view(strokeWidth=0)
    return themed.properties(height=height) if height is not None else themed

def render_range_bar(label: str, low: float, high: float, current: float, color: str = "#FFFFFF"):
    """Yahoo-style horizontal range bar με marker (όπως στην πρώτη εικόνα)"""
    if any(pd.isna(v) for v in [low, high, current]) or high <= low:
        st.markdown(f'<div style="font-size:0.72rem;color:#787B86;margin-bottom:12px;">{label}: N/A</div>', unsafe_allow_html=True)
        return
    
    position = max(0.0, min(100.0, ((current - low) / (high - low)) * 100))
    st.markdown(f"""
    <div style="margin: 12px 0 16px 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#787B86; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
            <span>{label}</span>
            <span>{low:,.2f} — {high:,.2f}</span>
        </div>
        <div style="position:relative; height:4px; background:#2A2E39; border-radius:999px; overflow:visible;">
            <div style="position:absolute; left:{position}%; top:-5px; width:14px; height:14px; 
                        background:{color}; border-radius:999px; border:3px solid #131722; box-shadow:0 0 8px rgba(0,0,0,0.5);"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.88rem; margin-top:10px;">
            <span style="color:#D1D4DC; font-weight:700;">{current:,.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_earnings_dots(ticker_symbol: str):
    """Earnings per Share dots (Actual vs Estimate) with Advanced Projection Modeling"""
    try:
        from marketLayer5 import fetch_earnings_dates
        df = fetch_earnings_dates(ticker_symbol)
        
        if df is None or df.empty:
            st.info("No earnings data available for this asset.")
            return
            
        # 1. Identify Actual column
        actual_col = 'Reported EPS' if 'Reported EPS' in df.columns else 'Actual'
        if actual_col not in df.columns and 'Actual' in df.columns:
             actual_col = 'Actual'
             
        if 'EPS Estimate' not in df.columns or actual_col not in df.columns:
            st.info("Incomplete earnings data (missing Estimates or Actuals).")
            return

        # 2. Timezone-Aware Date Handling & Sorting
        temp_df = df.copy()
        if temp_df.index.name == 'Earnings Date' or isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df = temp_df.reset_index()
        
        date_col = 'Earnings Date' if 'Earnings Date' in temp_df.columns else 'Date'
        if date_col not in temp_df.columns and 'index' in temp_df.columns:
            date_col = 'index'
        
        if date_col not in temp_df.columns:
             st.info("Could not identify Date column in earnings data.")
             return

        # Ensure UTC comparison and sort descending for chronological calculations
        try:
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], utc=True)
        except Exception:
            st.info("Date conversion failed.")
            return
            
        temp_df = temp_df.sort_values(by=date_col, ascending=False).reset_index(drop=True)
        now_utc = pd.Timestamp.now(tz='UTC')

        # 3. Data Pre-Processing Optimization (Enrichment)
        est_series = pd.to_numeric(temp_df['EPS Estimate'], errors='coerce')
        act_series = pd.to_numeric(temp_df[actual_col], errors='coerce')
        
        # Robust Surprise Magnitude Calculation
        temp_df['Surprise_Pct'] = ((act_series - est_series) / est_series.abs()) * 100
        
        # Implied Growth (YoY): Shift(-4) pulls the Actual EPS from 4 quarters ago
        yoy_base_series = act_series.shift(-4)
        temp_df['Implied_YoY_Pct'] = ((est_series - yoy_base_series) / yoy_base_series.abs()) * 100
        
        # Pending logic
        temp_df['Is_Pending'] = (temp_df[date_col] > now_utc) | (pd.isna(act_series))
        
        # Keep top 4 (most recent + pending), then reverse for left-to-right chart rendering
        processed_df = temp_df.head(4).iloc[::-1].copy().reset_index(drop=True)
        
        # 4. Strict Typing and Chart Data Preparation
        chart_data = pd.DataFrame()
        chart_data['Date'] = processed_df[date_col].dt.strftime('%b %y')
        chart_data['Estimate'] = pd.to_numeric(processed_df['EPS Estimate'], errors='coerce').astype(float)
        
        # Force pending actuals to strictly NaN
        chart_data['Actual'] = np.where(processed_df['Is_Pending'], np.nan, 
                                        pd.to_numeric(processed_df[actual_col], errors='coerce')).astype(float)
        
        chart_data['Is_Pending'] = processed_df['Is_Pending'].astype(bool)
        chart_data['Surprise_Pct'] = pd.to_numeric(processed_df['Surprise_Pct'], errors='coerce').astype(float)
        chart_data['Implied_YoY_Pct'] = pd.to_numeric(processed_df['Implied_YoY_Pct'], errors='coerce').astype(float)
        
        # Status
        conditions = [
            chart_data['Is_Pending'],
            chart_data['Actual'] >= chart_data['Estimate'],
            chart_data['Actual'] < chart_data['Estimate']
        ]
        choices = ["PENDING", "Beat", "Miss"]
        chart_data['Status'] = np.select(conditions, choices, default="Unknown")
        
        # Tooltip formatting
        def format_pct(x):
            if pd.isna(x): return "N/A"
            return f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
            
        chart_data['Tooltip_Surprise'] = chart_data['Surprise_Pct'].apply(format_pct)
        chart_data['Tooltip_YoY'] = chart_data['Implied_YoY_Pct'].apply(format_pct)

        # Projection Vector Logic
        chart_data['Projection_Y'] = np.where(chart_data['Is_Pending'], chart_data['Estimate'], chart_data['Actual'])
        chart_data['Is_Projection_Segment'] = False
        
        realized_idx = chart_data[~chart_data['Is_Pending']].index
        pending_idx = chart_data[chart_data['Is_Pending']].index
        if len(realized_idx) > 0 and len(pending_idx) > 0:
             # Connect last realized to first pending
             chart_data.loc[[realized_idx[-1], pending_idx[0]], 'Is_Projection_Segment'] = True

        if chart_data.empty or chart_data['Estimate'].isna().all():
             st.info("Insufficient numeric data to render the earnings visualization.")
             return

        # 5. Altair Chart Adjustments (Futuristic & Analytical UI)
        base = alt.Chart(chart_data).encode(
            x=alt.X("Date:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelColor="#8A8F9A", grid=False))
        )

        # The Expectation Layer (Grey dots for all)
        estimate_dots = base.mark_point(size=90, opacity=0.7, strokeWidth=1.5).encode(
            y=alt.Y("Estimate:Q", title=None, axis=alt.Axis(labelColor="#8A8F9A", grid=False)),
            color=alt.value("#787B86"),
            tooltip=[
                alt.Tooltip("Date:N", title="Quarter"),
                alt.Tooltip("Estimate:Q", format=".2f", title="Consensus Est")
            ]
        )

        # The Surprise Bars (Vertical Rules)
        surprise_rules = base.transform_filter("datum.Is_Pending == false").mark_rule(strokeWidth=2.5, opacity=0.85).encode(
            y="Estimate:Q",
            y2="Actual:Q",
            color=alt.condition("datum.Status == 'Beat'", alt.value("#26A69A"), alt.value("#EF5350"))
        )

        # The Realized Path (Solid line, terminates at last Actual)
        realized_line = base.transform_filter("datum.Is_Pending == false").mark_line(color="#2962FF", strokeWidth=3).encode(
            y="Actual:Q"
        )
        
        # Realized Nodes (Solid filled circles with white stroke)
        realized_nodes = base.transform_filter("datum.Is_Pending == false").mark_point(
            size=140, filled=True, stroke="white", strokeWidth=1.5, opacity=1
        ).encode(
            y="Actual:Q",
            color=alt.condition("datum.Status == 'Beat'", alt.value("#26A69A"), alt.value("#EF5350")),
            tooltip=[
                alt.Tooltip("Date:N", title="Quarter"),
                alt.Tooltip("Status:N", title="Status"),
                alt.Tooltip("Actual:Q", format=".2f", title="Actual"),
                alt.Tooltip("Estimate:Q", format=".2f", title="Estimate"),
                alt.Tooltip("Tooltip_Surprise:N", title="Surprise %")
            ]
        )

        # The Projection Vector (Dashed line connecting reality to expectation)
        projection_vector = base.transform_filter("datum.Is_Projection_Segment == true").mark_line(
            color="#F59E0B", strokeDash=[4, 4], strokeWidth=2, opacity=0.7
        ).encode(
            y="Projection_Y:Q"
        )

        # The Pending Node (Hollow, Amber glow)
        pending_node = base.transform_filter("datum.Is_Pending == true").mark_point(
            size=180, shape="diamond", stroke="#F59E0B", filled=False, strokeWidth=2.5
        ).encode(
            y="Estimate:Q",
            tooltip=[
                alt.Tooltip("Date:N", title="Quarter (PENDING)"),
                alt.Tooltip("Status:N", title="Status"),
                alt.Tooltip("Estimate:Q", format=".2f", title="Consensus Est"),
                alt.Tooltip("Tooltip_YoY:N", title="Implied YoY Growth %")
            ]
        )

        # Composite Rendering
        chart = (estimate_dots + surprise_rules + realized_line + realized_nodes + projection_vector + pending_node).properties(height=240)
        
        st.markdown('<div class="section-label" style="margin-top:20px; color:#D1D4DC;">Earnings Dynamics & Expectations Vector</div>', unsafe_allow_html=True)
        st.altair_chart(apply_chart_theme(chart), use_container_width=True)
        
    except Exception as e:
        st.info("Insufficient or malformed data to construct advanced earnings visualization.")

def build_rev_earning_chart(inc: pd.DataFrame):
    """Revenue vs Earnings Bar Chart (Annual)"""
    try:
        if inc is None or inc.empty:
            st.info("No annual financials available for this asset.")
            return
        
        # Extract Revenue
        rev_row = None
        for key in ['Total Revenue', 'Revenue', 'Operating Revenue']:
            if key in inc.index:
                rev_row = inc.loc[key]
                break
                
        # Extract Net Income
        net_row = None
        for key in ['Net Income', 'Net Income Common Stockholders', 'Net Income Continuous Operations', 'Net Income Including Noncontrolling Interests']:
            if key in inc.index:
                net_row = inc.loc[key]
                break
        
        if rev_row is None or net_row is None:
            st.info("Incomplete financial statement (missing revenue or net income).")
            return

        def _extract_period_year(col):
            if isinstance(col, str) and "(" in col and ")" in col:
                inner = col.split("(", 1)[1].rstrip(")").strip()
                parsed = pd.to_datetime(inner, format="%b '%y", errors="coerce")
                if pd.notna(parsed):
                    return int(parsed.year)
            parsed = pd.to_datetime(col, errors="coerce")
            return int(parsed.year) if pd.notna(parsed) else str(col)

        years = [_extract_period_year(col) for col in rev_row.index[:4]]
        data = pd.DataFrame({
            "Year": years,
            "Revenue": rev_row.values[:4] / 1e9,
            "Earnings": net_row.values[:4] / 1e9
        }).iloc[::-1]
        
        chart_df = data.melt("Year", var_name="Metric", value_name="Value ($B)")
        
        bars = alt.Chart(chart_df).mark_bar(cornerRadius=2).encode(
            x=alt.X("Year:O", title=None),
            y=alt.Y("Value ($B):Q", title="Billions ($)"),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=["Revenue", "Earnings"], range=["#2962FF", "#F59E0B"]), legend=alt.Legend(orient="bottom", title=None)),
            xOffset="Metric:N",
            tooltip=["Year", "Metric", alt.Tooltip("Value ($B):Q", format=".2fb")]
        ).properties(height=200)
        
        st.markdown('<div class="section-label" style="margin-top:20px;">Financials (Annual)</div>', unsafe_allow_html=True)
        st.altair_chart(apply_chart_theme(bars), use_container_width=True)
    except:
        pass

def build_line_area_chart(df, x_col, y_col, color="#72f1ff", fill="#15b57a", title=None, minimal=False, x_type="temporal"):
    axis_x = None if minimal else alt.Axis(labelColor=THEME["muted"], title=None, grid=False)
    axis_y = None if minimal else alt.Axis(labelColor=THEME["muted"], title=None, grid=False)
    base = alt.Chart(df).encode(
        x=alt.X(f"{x_col}:{'T' if x_type == 'temporal' else 'Q'}", axis=axis_x),
        y=alt.Y(f"{y_col}:Q", axis=axis_y),
    )
    chart = base.mark_area(color=fill, opacity=0.12) + base.mark_line(color=color, strokeWidth=2.8, interpolate="monotone")
    if title:
        chart = chart.properties(title=alt.TitleParams(title, fontSize=11, color=THEME["muted"]))
    return apply_chart_theme(chart, height=320 if minimal else 360)

def build_dual_line_chart(df, title):
    folded = alt.Chart(df).transform_fold(["Stock", "Benchmark"], as_=["Series", "Value"])
    color = alt.Color(
        "Series:N",
        scale=alt.Scale(domain=["Stock", "Benchmark"], range=[THEME["accent"], THEME["benchmark"]]),
        legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
    )
    base = folded.encode(
        x=alt.X("Date:T", axis=alt.Axis(labelColor=THEME["muted"], title=None)),
        y=alt.Y("Value:Q", axis=alt.Axis(labelColor=THEME["muted"], title=None)),
        color=color,
        tooltip=["Date:T", "Series:N", "Value:Q"],
    )
    chart = base.mark_area(opacity=0.12) + base.mark_line(strokeWidth=2.5, interpolate="monotone")
    if title:
        chart = chart.properties(title=alt.TitleParams(title, fontSize=11, color=THEME["muted"]))
    return apply_chart_theme(chart, height=320)
