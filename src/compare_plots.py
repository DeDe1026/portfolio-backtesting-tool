from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PERCENTILES_11 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def _fmt_chf(x, pos=None) -> str:
    # axis formatting: commas + 2 decimals
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def pick_paths_by_terminal_percentiles(paths: np.ndarray, percentiles: list[int]) -> np.ndarray:
    terminal = paths[:, -1]
    order = np.argsort(terminal)
    n = len(order)
    picked = []
    for p in percentiles:
        k = int(round((p / 100.0) * (n - 1)))
        picked.append(paths[order[k], :])
    return np.vstack(picked)


def save_weights_pie(weights: dict[str, float], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def median_path_series(paths: np.ndarray) -> np.ndarray:
    return np.median(paths, axis=0)


def save_median_paths_plot(
    basic_results: list[dict[str, Any]],
    opt_results: list[dict[str, Any]],
    out_path: Path,
    periods_per_year: int = 12,
) -> None:
    years = np.arange(basic_results[0]["paths"].shape[1]) / periods_per_year
    fig, ax = plt.subplots()

    for r in basic_results:
        ax.plot(years, median_path_series(r["paths"]), label=f"BASIC_{r['mode'].upper()}")

    for r in opt_results:
        ax.plot(years, median_path_series(r["paths"]), label=f"OPT_{r['mode'].upper()}")

    ax.set_xlabel("Years")
    ax.set_ylabel("Wealth (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_chf))
    ax.set_title("Median wealth path — BASIC vs OPTIMIZED")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_representative_paths_plot(
    paths: np.ndarray,
    out_path: Path,
    title: str,
    percentiles: list[int] | None = None,
    periods_per_year: int = 12,
) -> None:
    """
    Saves a plot of 11 percentile-selected paths:
      [1,10,20,30,40,50,60,70,80,90,99]
    """
    if percentiles is None:
        percentiles = PERCENTILES_11

    rep = pick_paths_by_terminal_percentiles(paths, percentiles)
    years = np.arange(rep.shape[1]) / periods_per_year

    fig, ax = plt.subplots()
    for i, p in enumerate(percentiles):
        ax.plot(years, rep[i, :], label=f"P{p}")

    ax.set_xlabel("Years")
    ax.set_ylabel("Wealth (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_chf))
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
