from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

import yfinance as yf


@dataclass
class FetchConfig:
    start: str = "1975-01-01"   # aim 50 years
    end: Optional[str] = None  # None => today
    price_field: str = "Adj Close"  # for equities/ETFs; FX sometimes only has Close
    cache_dir: Path = Path("data/raw_cache")

def _ensure_datetime_index(obj: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    if not isinstance(obj.index, pd.DatetimeIndex):
        obj.index = pd.to_datetime(obj.index)
    return obj.sort_index()


def _extract_single_series(df: pd.DataFrame, preferred_field: str) -> pd.Series:
    """
    Robustly extract one price series from yfinance output.
    Handles both normal columns and MultiIndex columns.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned from yfinance.")

    # Case A: normal columns like ['Open','High','Low','Close','Adj Close','Volume']
    if not isinstance(df.columns, pd.MultiIndex):
        col = preferred_field if preferred_field in df.columns else "Close"
        s = df[col].dropna()
        if not isinstance(s, pd.Series):
            # In rare cases this could still be a DataFrame if columns are duplicated
            s = s.iloc[:, 0]
        return s

    # Case B: MultiIndex columns (field, ticker) e.g. ('Adj Close','^GSPC')
    fields = df.columns.get_level_values(0).unique().tolist()
    use_field = preferred_field if preferred_field in fields else ("Close" if "Close" in fields else fields[0])

    sub = df[use_field]  # returns DataFrame indexed by date, columns=tickers
    if isinstance(sub, pd.DataFrame):
        # Single ticker case -> one column, multi ticker -> many
        if sub.shape[1] >= 1:
            s = sub.iloc[:, 0].dropna()
            return s

    raise ValueError("Could not extract a single price series from yfinance output.")

def download_daily_levels_yf(ticker: str, cfg: FetchConfig) -> pd.Series:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / f"{ticker.replace('^','_').replace('=','_')}_daily.csv"

    if cache_path.exists():
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return _ensure_datetime_index(s)

    df = yf.download(
        ticker,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    s = _extract_single_series(df, cfg.price_field)
    s.name = ticker
    s = _ensure_datetime_index(s)

    s.to_frame(name=ticker).to_csv(cache_path)
    return s

def daily_to_monthly_levels(daily: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Convert daily levels to month-end levels (last observation per month).
    Accepts Series or single-column DataFrame.
    """
    if isinstance(daily, pd.DataFrame):
        if daily.shape[1] == 0:
            raise ValueError("Empty DataFrame passed to daily_to_monthly_levels.")
        daily = daily.iloc[:, 0]

    daily = _ensure_datetime_index(daily).dropna()
    monthly = daily.resample("ME").last().dropna()
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
