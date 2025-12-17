from __future__ import annotations

from pathlib import Path
from typing import Union
import pandas as pd


def load_returns_data(
    file_path: Union[str, Path],
    date_col: str = "date",
) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Returns data file not found: {file_path}")

    df = pd.read_csv(file_path)

    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' not found in CSV.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    #we keep numeric asset columns only
    df = df.select_dtypes(include=["number"]).sort_index()

    if df.isna().any().any():
        print("[WARN] NaNs found; filling forward/backward.")
        df = df.ffill().bfill()

    if df.empty:
        raise ValueError("No numeric return columns found.")

    return df
