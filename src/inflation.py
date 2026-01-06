from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_ch_inflation_rates(
    path: str | Path = "data/raw/switzerland_inflation_monthly_clean.csv",
) -> pd.Series:
    """
    Load Swiss monthly inflation (MoM) from a CSV.
    Expected columns: date + inflation column (flexible names).
    Returns a Series indexed by month-end timestamps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inflation CSV not found: {path}")

    # delimiter auto-detect
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")

    # Drop totally empty columns (Excel often adds these)
    df = df.dropna(axis=1, how="all")

    # Normalize col names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Auto-detect date column
    date_candidates = ["date", "datum", "monat", "month", "time", "periode", "période"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        # fall back: first column
        date_col = df.columns[0]

    # Auto-detect inflation column
    infl_candidates = ["ch_inflation_mom", "ch_inflation", "inflation", "teuerung", "rate", "value", "wert"]
    infl_col = next((c for c in infl_candidates if c in df.columns), None)
    if infl_col is None:
        if len(df.columns) < 2:
            raise ValueError(f"Cannot detect inflation column. Columns: {list(df.columns)}")
        infl_col = df.columns[1]

    out = df[[date_col, infl_col]].copy()
    out.columns = ["date", "ch_inflation"]

    # Parse date + numeric
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    s = out["ch_inflation"]
    if s.dtype == "object":
        s = (
            s.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
    out["ch_inflation"] = pd.to_numeric(s, errors="coerce")

    out = out.dropna(subset=["date", "ch_inflation"]).sort_values("date")

    # Normalize to month-end index
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp("M")

    # If values are percent units (e.g., 0.2 meaning 0.2%), convert to decimal
    if out["ch_inflation"].median() > 0.05:
        out["ch_inflation"] = out["ch_inflation"] / 100.0

    infl = out.set_index("date")["ch_inflation"]
    infl.name = "ch_inflation"
    return infl



