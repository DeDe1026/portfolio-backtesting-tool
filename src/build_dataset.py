from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data_fetcher import FetchConfig, fetch_monthly_returns_yf
from src.inflation import load_ch_inflation_rates



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

    # --- Tickers  ---
    tickers: Dict[str, list[str]] = {
    "sp500_usd": ["^GSPC"],
    "smi_chf": ["^SSMI"],
    "usdchf": ["USDCHF=X"],
    # Gold: XAUUSD=X often fails; GC=F usually works
    "gold_usd": ["GC=F", "XAUUSD=X"],
    # US Treasuries proxy (20y or aggregated bonds)
    "us_gov_bonds_usd": ["TLT", "BND"],
    # Swiss gov bonds CHF proxy (if one fails, try another CHF bond ETF symbol you prefer, 7-15y or 0-3y)
    "ch_gov_bonds_chf": ["CSBGC0.SW", "CSBGC3.SW"],}


    series = {}
    fail_log = {}

    for colname, candidates in tickers.items():
        ok = False
        last_err = None

        for ticker in candidates:
            try:
                r = fetch_monthly_returns_yf(ticker, fetch_cfg)
                r.name = colname
                series[colname] = r
                print(f"[OK] {colname} ({ticker}) rows={len(r)}")
                ok = True
                break
            except Exception as e:
                last_err = str(e)

        if not ok:
            fail_log[colname] = {"candidates": candidates, "error": last_err}
            print(f"[WARN] Failed {colname} ({candidates}): {last_err}")

    if len(series) == 0:
        raise RuntimeError(
            "No series downloaded successfully. "
            f"Failures: {fail_log}")

    df = pd.concat(series.values(), axis=1).dropna(how="any")

    out_path = cfg.out_dir / cfg.out_filename
    df_out = df.copy()
    df_out.index.name = "date"
    df_out.reset_index().to_csv(out_path, index=False)

    print(f"\nSaved monthly dataset to: {out_path}")
    print(df.tail())

   

        # --- Swiss inflation  ---
    try:
        infl = load_ch_inflation_rates()
        print("[DEBUG] Inflation series head:")
        print(infl.head())
        df = df.join(infl, how="inner")
        print(f"[OK] Added Swiss inflation series: {len(infl)} months")

    except FileNotFoundError:
        print("[WARN] inflation CSV not found; proceeding without inflation series.")

    df_out = df.copy()
    df_out.index.name = "date"
    df_out.reset_index().to_csv(out_path, index=False)



    return df
