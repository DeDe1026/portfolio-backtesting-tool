from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models import MonteCarloSimulator, PortfolioConfig
from src.optimization import OptimizationConfig, optimize_portfolio


BOOTSTRAP_MODES = ["iid", "block", "regime"]


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw} if n > 0 else {}
    return {k: float(v) / total for k, v in raw.items()}


def apply_fx_conversion_to_chf(
    returns_df: pd.DataFrame,
    fx_col: str = "usdchf",
    usd_asset_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert USD asset returns into CHF returns using USDCHF:
        r_chf = (1+r_usd_asset)*(1+r_fx) - 1
    Keeps fx_col in df but fx_col is NOT investable.
    """
    df = returns_df.copy()
    if fx_col not in df.columns:
        return df

    fx = df[fx_col].astype(float)

    if usd_asset_cols is None:
        usd_asset_cols = [c for c in df.columns if c.endswith("_usd")]

    for c in usd_asset_cols:
        if c not in df.columns:
            continue
        df[c] = (1.0 + df[c].astype(float)) * (1.0 + fx) - 1.0

    return df


def max_drawdown(series: np.ndarray) -> float:
    x = np.asarray(series, dtype=float)
    x = np.maximum(x, 1e-12)
    peak = np.maximum.accumulate(x)
    dd = 1.0 - (x / peak)
    return float(np.max(dd))


def median_path_cagr_vol_mdd(paths: np.ndarray, periods_per_year: int = 12) -> tuple[float, float, float]:
    terminal = paths[:, -1]
    med = np.median(terminal)
    i = int(np.argmin(np.abs(terminal - med)))

    w = paths[i, :]
    rets = w[1:] / np.maximum(w[:-1], 1e-12) - 1.0

    years = (len(w) - 1) / periods_per_year
    if w[-1] <= 0 or w[0] <= 0:
        cagr = -1.0
    else:
        cagr = float((w[-1] / w[0]) ** (1.0 / max(years, 1e-12)) - 1.0)

    vol = float(np.std(rets, ddof=1) * np.sqrt(periods_per_year)) if len(rets) > 2 else float("nan")
    mdd = max_drawdown(w)
    return cagr, vol, mdd


def pick_paths_by_terminal_percentiles(paths: np.ndarray, percentiles: list[int]) -> np.ndarray:
    terminal = paths[:, -1]
    order = np.argsort(terminal)
    n = len(order)
    picked = []
    for p in percentiles:
        k = int(round((p / 100.0) * (n - 1)))
        picked.append(paths[order[k], :])
    return np.vstack(picked)


def run_one_mode(
    sim: MonteCarloSimulator,
    mode: str,
    n_paths: int,
    seed: int,
    alpha: float,
    block_size: int,
    regime_k: int,
    regime_vol_window: int,
    regime_min_samples: int,
) -> dict[str, Any]:
    paths, diag = sim.simulate_paths(
        n_paths=n_paths,
        random_state=seed,
        bootstrap_mode=mode,
        block_size=block_size,
        alpha=alpha,
        regime_k=regime_k,
        regime_vol_window=regime_vol_window,
        regime_min_samples=regime_min_samples,
        return_diagnostics=True,
    )

    terminal = paths[:, -1]
    survival = float(np.mean(terminal > 0.0))
    failed = int(np.sum(terminal <= 0.0))

    neg_months = diag["neg_months"]  # (n_paths, n_periods)

    return {
        "mode": mode,
        "paths": paths,
        "terminal": terminal,
        "survival": survival,
        "failed": failed,
        "n_paths": int(n_paths),
        "neg_months": neg_months,
        "median_terminal": float(np.median(terminal)),
        "p10_terminal": float(np.percentile(terminal, 10)),
        "p90_terminal": float(np.percentile(terminal, 90)),
    }


def build_comp_rows(results_list: list[dict[str, Any]], scenario_label: str) -> list[dict[str, Any]]:
    rows = []
    for r in results_list:
        neg = r["neg_months"]  # (n_paths, n_periods)
        neg_per_path = neg.sum(axis=1)
        rows.append({
            "Scenario": scenario_label,
            "Bootstrap": r["mode"].upper(),
            "Survival": float(r["survival"]),
            "Failed (count)": float(r["failed"]),
            "Failed (%)": float(r["failed"] / r["n_paths"]),
            "Neg months / path (median)": float(np.median(neg_per_path)),
            "Neg months / path (mean)": float(np.mean(neg_per_path)),
            "Total months": float(neg.shape[1]),
            "Median terminal (CHF)": float(r["median_terminal"]),
            "P10 terminal (CHF)": float(r["p10_terminal"]),
            "P90 terminal (CHF)": float(r["p90_terminal"]),
        })
    return rows


def median_path_series(paths: np.ndarray) -> np.ndarray:
    return np.median(paths, axis=0)


def save_weights_pie(weights: dict[str, float], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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
    ax.set_title("Median wealth path — BASIC vs OPTIMIZED")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_basic_and_optimized(
    returns_df_fx: pd.DataFrame,
    assets: list[str],
    weights_guess: dict[str, float],
    initial_capital: float,
    horizon_years: int,
    w_pref: float,
    w_floor: float,
    inflation_toggle: bool,
    inflation_col: str,
    opt_cfg: OptimizationConfig,
    n_paths: int,
    seed: int,
    block_size: int,
    regime_k: int,
    regime_vol_window: int,
    regime_min_samples: int,
) -> dict[str, Any]:
    """
    Runs:
      - BASIC: user weights + floor rule via alpha_basic
      - OPTIMIZED: optuna returns best weights/alpha/withdrawal_rate, then simulate across all 3 bootstrap modes

    Returns dict with:
      - comp_df
      - perf_df
      - basic_results
      - opt_results
      - opt_result (best weights/alpha/withdrawal)
    """
    withdrawal_rate_pref = (12.0 * w_pref) / initial_capital if initial_capital > 0 else 0.0

    # BASIC alpha implied by floor vs preferred
    alpha_basic = 0.0
    if w_pref > 0:
        alpha_basic = max(0.0, min(0.999, 1.0 - (w_floor / w_pref)))

    cfg_basic = PortfolioConfig(
        initial_capital=float(initial_capital),
        withdrawal_rate=float(withdrawal_rate_pref),
        horizon_years=int(horizon_years),
        rebalance_frequency="yearly",
        inflation_aware_withdrawals=bool(inflation_toggle),
        inflation_col=str(inflation_col),
    )

    sim_basic = MonteCarloSimulator(
        returns=returns_df_fx,
        asset_weights=weights_guess,
        config=cfg_basic,
        periods_per_year=12,
    )

    basic_results = []
    for mode in BOOTSTRAP_MODES:
        basic_results.append(
            run_one_mode(
                sim_basic, mode, n_paths, seed, float(alpha_basic),
                block_size, regime_k, regime_vol_window, regime_min_samples
            )
        )

    # OPTIMIZATION base cfg uses preferred withdrawal baseline (as you do in Streamlit)
    base_cfg = PortfolioConfig(
        initial_capital=float(initial_capital),
        withdrawal_rate=float(withdrawal_rate_pref),
        horizon_years=int(horizon_years),
        rebalance_frequency="yearly",
        inflation_aware_withdrawals=bool(inflation_toggle),
        inflation_col=str(inflation_col),
    )

    opt_result = optimize_portfolio(
        returns_df=returns_df_fx,
        assets=assets,
        base_config=base_cfg,
        opt_config=opt_cfg,
        periods_per_year=12,
    )

    best_weights = opt_result["best_weights"]
    best_alpha = float(opt_result["best_alpha"])
    best_wr = float(opt_result["best_withdrawal_rate"])

    cfg_best = PortfolioConfig(
        initial_capital=float(initial_capital),
        withdrawal_rate=float(best_wr),
        horizon_years=int(horizon_years),
        rebalance_frequency="yearly",
        inflation_aware_withdrawals=bool(inflation_toggle),
        inflation_col=str(inflation_col),
    )
    sim_best = MonteCarloSimulator(
        returns=returns_df_fx,
        asset_weights=best_weights,
        config=cfg_best,
        periods_per_year=12,
    )

    opt_results = []
    for mode in BOOTSTRAP_MODES:
        opt_results.append(
            run_one_mode(
                sim_best, mode, n_paths, seed, best_alpha,
                block_size, regime_k, regime_vol_window, regime_min_samples
            )
        )

    comp_df = pd.DataFrame(
        build_comp_rows(basic_results, "BASIC (user weights, floor rule)") +
        build_comp_rows(opt_results, "OPTIMIZED (Bayesian)")
    )

    perf_rows = []
    for scenario_label, res_list in [("BASIC", basic_results), ("OPTIMIZED", opt_results)]:
        for r in res_list:
            cagr, vol, mdd = median_path_cagr_vol_mdd(r["paths"], periods_per_year=12)
            perf_rows.append({
                "Scenario": scenario_label,
                "Bootstrap": r["mode"].upper(),
                "Median-path CAGR": float(cagr),
                "Median-path Vol (ann.)": float(vol),
                "Median-path Max Drawdown": float(mdd),
            })
    perf_df = pd.DataFrame(perf_rows)

    return {
        "comp_df": comp_df,
        "perf_df": perf_df,
        "basic_results": basic_results,
        "opt_results": opt_results,
        "opt_result": opt_result,
        "alpha_basic": float(alpha_basic),
        "withdrawal_rate_pref": float(withdrawal_rate_pref),
    }
