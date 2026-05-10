
import pandas as pd
import datetime as dt
from DataLayer6 import db_log_earnings_event, db_get_ticker_journal
import logging

# Task 10: Event Intelligence Engine (Senior Refined Version)
logger = logging.getLogger(__name__)

def run_earnings_pulse(
    port_df: pd.DataFrame, 
    market_data: dict,
    meta_df: pd.DataFrame = None
) -> list:
    """
    Aspect 10: Earnings Pulse & Journal (Event Intelligence)
    Strategic feature focusing on holdings with earnings or significant AH moves.
    """
    if port_df.empty:
        return []

    pulse_notes = []
    today = dt.date.today()
    today_str = today.isoformat()

    for _, row in port_df.iterrows():
        ticker = row["Ticker"]
        m_data = market_data.get(ticker, {})
        
        curr = m_data.get("current")
        prev = m_data.get("prev_close")
        ah_move = (curr / prev - 1) * 100 if curr and prev else 0.0

        # === TRIGGER 1: Recent Earnings (Last 48h) ===
        # Always check earnings regardless of price move for holdings
        earnings_note = _check_recent_earnings(ticker, today, ah_move, meta_df)
        if earnings_note:
            pulse_notes.append(earnings_note)
            continue

        # === TRIGGER 2: Significant After-Hours Move (>2.0%) without earnings ===
        if abs(ah_move) >= 2.0:
            # Deduplication: check today's journal
            existing = db_get_ticker_journal(ticker, limit=1)
            if not existing.empty and existing.iloc[0].get("event_date") == today_str:
                pulse_notes.append({
                    "ticker": ticker,
                    "note": existing.iloc[0]["note_text"],
                    "timestamp": existing.iloc[0].get("timestamp", "Today")
                })
                continue

            note = f"Significant After-Hours Move: {ah_move:+.1f}% | Price action alert."
            db_log_earnings_event(
                ticker=ticker,
                quarter="N/A",
                event_date=today_str,
                eps_actual=None,
                eps_est=None,
                revenue=0.0,
                ah_return=ah_move,
                note_text=note
            )
            pulse_notes.append({"ticker": ticker, "note": note, "timestamp": "Today"})

    return pulse_notes


def _check_recent_earnings(ticker: str, today: dt.date, ah_move: float, meta_df: pd.DataFrame = None) -> dict | None:
    """Robust earnings detection without direct yfinance usage"""
    from marketLayer5 import fetch_earnings_dates
    try:
        ed = fetch_earnings_dates(ticker)
        if ed.empty:
            return None

        # Get latest quarter with actual data
        recent = ed.dropna(subset=['Actual', 'EPS Estimate']).head(1)
        if recent.empty:
            return None

        report_date = recent.index[0].date()
        # Report must be in the last 48 hours to trigger a fresh 'Pulse'
        if (today - report_date).days > 2:
            return None

        actual = float(recent.iloc[0]['Actual'])
        est = float(recent.iloc[0]['EPS Estimate'])
        beat_pct = (actual / est - 1) * 100 if est != 0 else 0

        # Strategic Enrichment from Finance Ontology (Aspect 2, 3, 4)
        pe_info = ""
        if meta_df is not None and not meta_df.empty:
             match = meta_df[meta_df["ticker"] == ticker]
             if not match.empty:
                 row = match.iloc[0]
                 pe_val = row.get("trailing_pe") or row.get("trailingPE")
                 if pd.notna(pe_val):
                     pe_info = f" | P/E: {pe_val:.1f}x"

        note = f"Q{report_date.month//3 + 1} Beat: {actual:.2f} vs {est:.2f} ({beat_pct:+.1f}%){pe_info} | AH {ah_move:+.1f}%"
        if ah_move > 0.5: note += " → Bullish"
        elif ah_move < -0.5: note += " → Bearish"

        # Prevent duplicate logs for the same quarter/date
        existing = db_get_ticker_journal(ticker, limit=1)
        if not existing.empty and existing.iloc[0].get("event_date") == report_date.isoformat():
             return {"ticker": ticker, "note": existing.iloc[0]["note_text"], "timestamp": "Recent Report"}

        db_log_earnings_event(
            ticker=ticker,
            quarter=f"Q{report_date.month//3 + 1}",
            event_date=report_date.isoformat(),
            eps_actual=actual,
            eps_est=est,
            revenue=0.0,
            ah_return=ah_move,
            note_text=note
        )
        return {"ticker": ticker, "note": note, "timestamp": "Just now"}

    except Exception as e:
        logger.warning(f"Earnings check failed for {ticker}: {e}")
        return None
