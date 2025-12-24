from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_ch_inflation_rates(
    path: Path | str = "data/raw/switzerland_inflation_monthly.csv",
) -> pd.Series:
    """
    Load Swiss inflation rates (monthly).
    Expected columns: date, ch_inflation
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"inflation rates file not found: {path}. "
            "Provide a monthly inflation rates CSV."
        )

    df = pd.read_csv(path, sep=";", parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    if "ch_inflation" not in df.columns:
        raise ValueError(f"Expected 'ch_inflation' column, found: {list(df.columns)}")

    s = df.set_index("date")["ch_inflation"].astype(float)
    s.name = "ch_inflation"
    return s



