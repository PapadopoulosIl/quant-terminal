"""
Technical_engine11.py — Layer B: Pure Technical Indicators & Math
==================================================================
Stateless mathematical transformations on price/volume data.
No I/O. Vectorized performance.
"""

import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using EWM (Exponential Weighted Moving Average)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = (
        up.ewm(com=period - 1, adjust=False).mean()
        / down.ewm(com=period - 1, adjust=False).mean()
    )
    return 100 - 100 / (1 + rs)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_slope(series: pd.Series, period: int = 20) -> float:
    """Calculate the linear slope of a series (normalized)."""
    if len(series) < period:
        return 0.0
    y = series.tail(period).values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope / y[0]) if y[0] != 0 else 0.0

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """Bollinger Bands."""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    return {
        "upper": sma + (std * std_dev),
        "lower": sma - (std * std_dev),
        "mid": sma
    }

def calculate_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Z-Score."""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    return (series - sma) / std

def calculate_std(series: pd.Series, period: int) -> pd.Series:
    """Rolling Standard Deviation."""
    return series.rolling(window=period).std()

def normalize_series(series: pd.Series, base_value: float = 100.0) -> pd.Series:
    """Normalize series to a base value (e.g., 100)."""
    if series.empty:
        return series
    return (series / series.iloc[0]) * base_value

def compute_cumulative_return_series(returns_series: pd.Series, base_value: float = 100.0) -> pd.Series:
    """Compute cumulative return series from percentage returns."""
    if returns_series.empty:
        return returns_series
    return (1 + returns_series).cumprod() * base_value

def extract_technical_features(daily: pd.DataFrame, intraday: pd.DataFrame) -> dict:
    """
    Layer B: Pure Technical Enrichment.
    Extracts high-level technical features from OHLCV data.
    """
    if daily.empty:
        return {}
        
    close = daily["Close"].dropna()
    high = daily["High"].dropna()
    low = daily["Low"].dropna()
    volume = daily["Volume"].dropna() if "Volume" in daily else pd.Series(dtype=float)
    
    current_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
    
    # Returns
    ret_1d = (current_price / prev_close - 1) * 100 if prev_close else np.nan
    ret_1w = (current_price / close.iloc[max(0, len(close) - 6)] - 1) * 100 if len(close) > 5 else np.nan
    ret_1m = (current_price / close.iloc[max(0, len(close) - 22)] - 1) * 100 if len(close) > 21 else np.nan
    
    # YTD
    import datetime as dt
    year_start = close[close.index >= pd.Timestamp(dt.date.today().replace(month=1, day=1))]
    ytd_ret = (current_price / float(year_start.iloc[0]) - 1) * 100 if not year_start.empty else np.nan
    
    # Volatility & ATR
    realized_vol_20 = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    
    # Volume
    latest_volume = float(volume.iloc[-1]) if not volume.empty else np.nan
    vol_20_avg = float(volume.rolling(20).mean().iloc[-1]) if len(volume) > 20 else np.nan
    volume_ratio = (latest_volume / vol_20_avg) if (pd.notna(vol_20_avg) and vol_20_avg) else np.nan
    
    # 52W High/Low
    high_52w = float(high.iloc[-252:].max()) if len(high) >= 1 else np.nan
    low_52w  = float(low.iloc[-252:].min())  if len(low)  >= 1 else np.nan
    dist_high = (current_price / high_52w - 1) * 100 if pd.notna(high_52w) and high_52w else np.nan
    dist_low  = (current_price / low_52w  - 1) * 100 if pd.notna(low_52w)  and low_52w  else np.nan
    
    # Intraday
    intraday_close = intraday["Close"].dropna() if (not intraday.empty and "Close" in intraday) else pd.Series(dtype=float)
    pulse_return = (float(intraday_close.iloc[-1]) / float(intraday_close.iloc[0]) - 1) * 100 if len(intraday_close) > 1 else np.nan
    
    # Trend Score
    sma20 = calculate_sma(close, 20).iloc[-1]
    sma50 = calculate_sma(close, 50).iloc[-1]
    sma200 = calculate_sma(close, 200).iloc[-1]
    
    trend_score = sum([
        1 if current_price > sma20 else 0,
        1 if current_price > sma50 else 0,
        1 if (pd.notna(sma200) and current_price > sma200) else 0,
    ])
    
    return {
        "current_price": current_price, "ret_1d": ret_1d, "ret_1w": ret_1w,
        "ret_1m": ret_1m, "ytd_ret": ytd_ret, "realized_vol_20": realized_vol_20,
        "atr14": atr14, "volume_ratio": volume_ratio, "dist_high": dist_high,
        "dist_low": dist_low, "pulse_return": pulse_return, "trend_score": trend_score
    }
