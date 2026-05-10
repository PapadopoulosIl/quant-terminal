"""
data_layer.py — Data Layer: DB access + metadata TTL refresh
============================================================
Responsible for:
  - SQLite schema init + migrations
  - Transaction CRUD (append-only ledger)
  - Asset metadata upsert with 24h TTL refresh
  - Returns clean DataFrames — no business logic

Contract:
  Input:  primitive types (str, float, int)
  Output: pd.DataFrame or None — callers treat as read-only
"""
from __future__ import annotations
import datetime as dt
import sqlite3
import json

import numpy as np
import pandas as pd

from utils import safe_float

DB_PATH = "local_portfolio.db"
METADATA_TTL_HOURS = 24
MACRO_TTL_HOURS = 12

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _db_conn():
    return sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)


# ---------------------------------------------------------------------------
# Schema init (migration-safe)
# ---------------------------------------------------------------------------

def init_portfolio_db() -> None:
    with _db_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS transactions
               (id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                action          TEXT NOT NULL CHECK(action IN ('BUY','SELL')),
                shares          REAL NOT NULL CHECK(shares > 0),
                execution_price REAL NOT NULL CHECK(execution_price > 0),
                timestamp       TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS asset_metadata
               (ticker       TEXT PRIMARY KEY,
                sector       TEXT,
                industry     TEXT,
                asset_class  TEXT,
                beta         REAL,
                div_yield    REAL,
                last_updated TEXT)"""
        )
        for col in ["trailing_pe", "forward_pe", "price_to_sales", "projected_growth"]:
            try:
                conn.execute(f"ALTER TABLE asset_metadata ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass
        conn.execute(
            """CREATE TABLE IF NOT EXISTS macro_environment_cache
               (cache_key    TEXT PRIMARY KEY,
                payload      TEXT NOT NULL,
                last_updated TEXT NOT NULL)"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_user ON transactions(username, ticker)"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS earnings_journal
               (id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT NOT NULL,
                quarter      TEXT,
                event_date   TEXT NOT NULL,
                eps_actual   REAL,
                eps_est      REAL,
                revenue      REAL,
                ah_return    REAL,
                note_text    TEXT NOT NULL,
                timestamp    TEXT NOT NULL)"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_ticker ON earnings_journal(ticker, event_date)"
        )


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------

def db_add_user(username: str) -> None:
    with _db_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))


# ---------------------------------------------------------------------------
# Transaction CRUD
# ---------------------------------------------------------------------------

def db_log_transaction(
    username: str, ticker: str, action: str, shares: float, execution_price: float
) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO transactions
               (username, ticker, action, shares, execution_price, timestamp)
               VALUES (?,?,?,?,?,?)""",
            (username, ticker.upper(), action, shares, execution_price, ts),
        )


def db_get_transactions(username: str) -> pd.DataFrame:
    with _db_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM transactions WHERE username=? ORDER BY timestamp ASC",
            conn, params=(username,),
        )


def db_delete_transaction(tx_id: int) -> None:
    with _db_conn() as conn:
        conn.execute("DELETE FROM transactions WHERE id=?", (tx_id,))


# ---------------------------------------------------------------------------
# Asset metadata
# ---------------------------------------------------------------------------

def db_upsert_metadata(ticker: str, info: dict) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO asset_metadata
               (ticker, sector, industry, asset_class, beta, div_yield,
                trailing_pe, forward_pe, price_to_sales, projected_growth, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker) DO UPDATE SET
                 sector=excluded.sector, industry=excluded.industry,
                 asset_class=excluded.asset_class,
                 beta=excluded.beta, div_yield=excluded.div_yield,
                 trailing_pe=excluded.trailing_pe, forward_pe=excluded.forward_pe,
                 price_to_sales=excluded.price_to_sales,
                 projected_growth=excluded.projected_growth,
                 last_updated=excluded.last_updated""",
            (
                ticker.upper(),
                info.get("sector") or "Unknown",
                info.get("industry") or "Unknown",
                info.get("quoteType") or "EQUITY",
                safe_float(info.get("beta")),
                safe_float(info.get("dividendYield")),
                safe_float(info.get("trailingPE")),
                safe_float(info.get("forwardPE")),
                safe_float(info.get("priceToSalesTrailing12Months")),
                safe_float(info.get("earningsGrowth")),
                now,
            ),
        )


def db_get_metadata(tickers: list) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    with _db_conn() as conn:
        return pd.read_sql_query(
            f"SELECT * FROM asset_metadata WHERE ticker IN ({placeholders})",
            conn, params=tickers,
        )


def get_stale_or_missing(tickers: list) -> list[str]:
    """Returns tickers that are missing from DB or whose metadata is >TTL hours old."""
    existing = db_get_metadata(tickers)
    now = dt.datetime.now()
    stale = []
    for t in tickers:
        if existing.empty or t not in existing["ticker"].values:
            stale.append(t); continue
        row = existing[existing["ticker"] == t].iloc[0]
        if "forward_pe" not in row or pd.isna(row.get("forward_pe")):
            stale.append(t); continue
        try:
            last = dt.datetime.fromisoformat(row["last_updated"])
            if (now - last).total_seconds() > METADATA_TTL_HOURS * 3600:
                stale.append(t)
        except (TypeError, ValueError):
            stale.append(t)
    return stale


# ---------------------------------------------------------------------------
# Macro Environment Caching
# ---------------------------------------------------------------------------

def is_macro_cache_stale(key: str = "global_macro") -> bool:
    """Returns True if the macro cache for 'key' is missing or > MACRO_TTL_HOURS old."""
    now = dt.datetime.now()
    with _db_conn() as conn:
        cursor = conn.execute("SELECT last_updated FROM macro_environment_cache WHERE cache_key=?", (key,))
        row = cursor.fetchone()
        if not row:
            return True
            
        try:
            last = dt.datetime.fromisoformat(row[0])
            if (now - last).total_seconds() > MACRO_TTL_HOURS * 3600:
                return True
        except (TypeError, ValueError):
            return True
            
    return False

def db_get_macro_environment(key: str = "global_macro") -> dict:
    """Fetches and deserializes the macro report dictionary from the cache."""
    with _db_conn() as conn:
        cursor = conn.execute("SELECT payload FROM macro_environment_cache WHERE cache_key=?", (key,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return {}
    return {}

def db_upsert_macro_environment(macro_report: dict, key: str = "global_macro") -> None:
    """Serializes and upserts the macro report dictionary to the cache."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    try:
        payload = json.dumps(macro_report)
    except Exception:
        return
        
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO macro_environment_cache (cache_key, payload, last_updated)
               VALUES (?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 payload=excluded.payload,
                 last_updated=excluded.last_updated""",
            (key, payload, now)
        )

def db_log_earnings_event(
    ticker: str, quarter: str, event_date: str, eps_actual: float, 
    eps_est: float, revenue: float, ah_return: float, note_text: str
) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    with _db_conn() as conn:
        conn.execute(
            """INSERT INTO earnings_journal
               (ticker, quarter, event_date, eps_actual, eps_est, revenue, ah_return, note_text, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ticker.upper(), quarter, event_date, eps_actual, eps_est, revenue, ah_return, note_text, now),
        )

def db_get_earnings_journal(limit: int = 10) -> pd.DataFrame:
    with _db_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM earnings_journal ORDER BY event_date DESC, timestamp DESC LIMIT ?",
            conn, params=(limit,),
        )

def db_get_ticker_journal(ticker: str, limit: int = 4) -> pd.DataFrame:
    with _db_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM earnings_journal WHERE ticker=? ORDER BY event_date DESC LIMIT ?",
            conn, params=(ticker.upper(), limit),
        )