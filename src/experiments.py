from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class CompareConfig:
    n_paths: int = 2000
    random_state: int = 42
    alpha: float = 0.0

    # Block
    block_size: int = 12

    # Regime
    regime_k: int = 3
    regime_vol_window: int = 12
    regime_min_samples: int = 24


def max_drawdown(path: np.ndarray) -> float:
    """
    path: shape (T,)
    Returns max drawdown as a fraction (e.g. 0.35 = -35% from peak).
    """
    peak = np.maximum.accumulate(path)
    dd = (peak - path) / np.clip(peak, 1e-12, None)
    return float(np.max(dd))


def compute_path_metrics(paths: np.ndarray) -> Dict[str, Any]:
    """
    paths shape: (n_paths, T)
    """
    terminal = paths[:, -1]
    survival = float(np.mean(terminal > 0.0))

    dd = np.array([max_drawdown(paths[i, :]) for i in range(paths.shape[0])], dtype=float)

    metrics = {
        "survival_rate": survival,
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_p10": float(np.percentile(terminal, 10)),
        "terminal_p90": float(np.percentile(terminal, 90)),
        "terminal_min": float(np.min(terminal)),
        "terminal_max": float(np.max(terminal)),
        "maxdd_mean": float(np.mean(dd)),
        "maxdd_median": float(np.median(dd)),
        "maxdd_p10": float(np.percentile(dd, 10)),
        "maxdd_p90": float(np.percentile(dd, 90)),
    }
    return metrics


def run_bootstrap_comparison(sim, cfg: CompareConfig) -> tuple[pd.DataFrame, dict]:
    """
    Runs iid vs block vs regime and returns:
      - summary dataframe
      - raw outputs dict (for plotting boxplots etc.)
    """
    outputs = {}

    # IID
    paths_iid = sim.simulate_paths(
        n_paths=cfg.n_paths,
        random_state=cfg.random_state,
        bootstrap_mode="iid",
        alpha=cfg.alpha,
    )
    outputs["iid"] = paths_iid

    # Block (robust): if too little data, shrink block_size to a valid value
    try:
        paths_block = sim.simulate_paths(
            n_paths=cfg.n_paths,
            random_state=cfg.random_state,
            bootstrap_mode="block",
            block_size=cfg.block_size,
            alpha=cfg.alpha,
        )
        outputs["block"] = paths_block
    except ValueError as e:
        # Try fallback block sizes (down to 1) instead of crashing
        fallback_sizes = [6, 3, 2, 1]
        ok = False
        for bs in fallback_sizes:
            try:
                paths_block = sim.simulate_paths(
                    n_paths=cfg.n_paths,
                    random_state=cfg.random_state,
                    bootstrap_mode="block",
                    block_size=bs,
                    alpha=cfg.alpha,
                )
                outputs["block"] = paths_block
                print(f"[WARN] Block bootstrap failed for block_size={cfg.block_size}. "
                    f"Falling back to block_size={bs}.")
                ok = True
                break
            except ValueError:
                continue

        if not ok:
            print(f"[WARN] Block bootstrap skipped: {e}")

    # Regime
    paths_regime = sim.simulate_paths(
        n_paths=cfg.n_paths,
        random_state=cfg.random_state,
        bootstrap_mode="regime",
        alpha=cfg.alpha,
        regime_k=cfg.regime_k,
        regime_vol_window=cfg.regime_vol_window,
        regime_min_samples=cfg.regime_min_samples,
    )
    outputs["regime"] = paths_regime

    rows: List[Dict[str, Any]] = []
    for mode, paths in outputs.items():
        m = compute_path_metrics(paths)
        m["mode"] = mode
        rows.append(m)

    df = pd.DataFrame(rows).set_index("mode").sort_index()
    return df, outputs
