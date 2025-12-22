from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# yfinance is very convenient for daily market data
import yfinance as yf


@dataclass
class FetchConfig:
    start: str = "1975-01-01"   # aim 50 years
    end: Optional[str] = None  # None => today
    price_field: str = "Adj Close"  # for equities/ETFs; FX sometimes only has Close
    cache_dir: Path = Path("data/raw_cache")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_daily_levels_yf(ticker: str, cfg: FetchConfig) -> pd.Series:
    """
    Downloads daily price levels from Yahoo Finance via yfinance.
    Returns a Series of levels (daily).
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / f"{ticker.replace('^','_')}_daily.csv"

    # Cache: if exists, load
    if cache_path.exists():
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return _ensure_datetime_index(s.to_frame()).iloc[:, 0]

    df = yf.download(
        ticker,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    # Choose best column
    col = cfg.price_field if cfg.price_field in df.columns else "Close"
    s = df[col].dropna().copy()
    s.name = ticker

    # Save cache
    s.to_frame().to_csv(cache_path)
    return s


def daily_to_monthly_levels(daily: pd.Series) -> pd.Series:
    """
    Convert daily levels to month-end levels (last observation per month).
    """
    daily = _ensure_datetime_index(daily.to_frame()).iloc[:, 0]
    monthly = daily.resample("M").last().dropna()
    monthly.name = daily.name
    return monthly


def monthly_levels_to_returns(monthly_levels: pd.Series) -> pd.Series:
    """
    Compute month-over-month returns from month-end levels.
    """
    r = monthly_levels.pct_change().dropna()
    r.name = monthly_levels.name
    return r


def fetch_monthly_returns_yf(ticker: str, cfg: FetchConfig) -> pd.Series:
    daily = download_daily_levels_yf(ticker, cfg)
    monthly = daily_to_monthly_levels(daily)
    return monthly_levels_to_returns(monthly)
