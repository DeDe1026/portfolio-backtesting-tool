from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_cpi_index(
    path: Path | str = "data/raw/switzerland_cpi.csv",
) -> pd.Series:
    """
    Load Swiss CPI index level (monthly).
    Expected columns: date, cpi
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CPI file not found: {path}. "
            "Provide a monthly CPI index CSV."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    if "cpi" not in df.columns:
        raise ValueError("CPI CSV must contain column named 'cpi'")

    return df["cpi"]


def compute_monthly_inflation(cpi_index: pd.Series) -> pd.Series:
    """
    Compute month-over-month inflation from CPI index levels.
    """
    infl = cpi_index.pct_change().dropna()
    infl.name = "ch_inflation"
    return infl
