from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data_fetcher import FetchConfig, fetch_monthly_returns_yf
from src.inflation import load_cpi_index, compute_monthly_inflation



@dataclass
class DatasetConfig:
    start: str = "1975-01-01"
    end: Optional[str] = None
    out_dir: Path = Path("data/raw")
    out_filename: str = "monthly_returns_native.csv"


def build_monthly_returns_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Builds a monthly returns dataset in native currencies.
    You can later convert USD assets to CHF using USDCHF returns.

    Columns:
      - sp500_usd
      - smi_chf
      - usdchf
      - gold_usd
      - us_gov_bonds_usd (ETF proxy)
      - ch_gov_bonds_chf (ETF proxy)
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    fetch_cfg = FetchConfig(start=cfg.start, end=cfg.end)

    # --- Tickers (you can adjust later) ---
    tickers: Dict[str, str] = {
        "sp500_usd": "^GSPC",
        "smi_chf": "^SSMI",          # if your yahoo locale needs different symbol, adjust here
        "usdchf": "USDCHF=X",
        "gold_usd": "XAUUSD=X",      # fallback could be "GC=F"
        "us_gov_bonds_usd": "TLT",   # proxy (20y US Treasuries)
        "ch_gov_bonds_chf": "CSBGC0.SW",  # proxy CHF Swiss govt bonds ETF (7-15y)
    }

    series = {}
    for colname, ticker in tickers.items():
        try:
            r = fetch_monthly_returns_yf(ticker, fetch_cfg)
            r.name = colname
            series[colname] = r
            print(f"[OK] {colname} ({ticker}) rows={len(r)}")
        except Exception as e:
            print(f"[WARN] Failed {colname} ({ticker}): {e}")

    df = pd.concat(series.values(), axis=1).dropna(how="any")

    out_path = cfg.out_dir / cfg.out_filename
    df.to_csv(out_path, index=True)
    print(f"\nSaved monthly dataset to: {out_path}")
    print(df.tail())

        # --- Swiss inflation (optional but recommended) ---
    try:
        cpi = load_cpi_index()
        infl = compute_monthly_inflation(cpi)

        df = df.join(infl, how="inner")
        print("[OK] Swiss inflation added")

    except FileNotFoundError:
        print("[WARN] Swiss CPI file not found — inflation excluded")


    return df
