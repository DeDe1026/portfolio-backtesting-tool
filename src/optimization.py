from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import optuna
import pandas as pd

from src.models import MonteCarloSimulator, PortfolioConfig


@dataclass
class OptimizationConfig:
    n_trials: int = 50
    seed: int = 42
    target_survival: float = 0.95
    n_paths_eval: int = 2000

    # Search ranges
    withdrawal_min: float = 0.01
    withdrawal_max: float = 0.08

    alpha_min: float = 0.0
    alpha_max: float = 0.30

    # Bootstrap settings during optimization
    bootstrap_mode: Literal["iid", "block"] = "iid"
    block_size: int = 12


def _normalize_weights(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw, 1e-12, None)
    return raw / raw.sum()


def export_optuna_trials_csv(study: optuna.Study, out_path: str) -> None:
    """
    Export Optuna trial history (params + survival) to a CSV for analysis and reporting.
    """
    rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "objective_value": t.value,
            "state": str(t.state),
        }

        # params
        for k, v in t.params.items():
            row[k] = v

        # tracked metadata
        row["survival_rate"] = t.user_attrs.get("survival_rate")
        row["penalized"] = t.user_attrs.get("penalized", False)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def optimize_portfolio(
    returns_df,
    assets: list[str],
    base_config: PortfolioConfig,
    opt_config: OptimizationConfig,
    periods_per_year: int = 12,
    fixed_bootstrap_mode: Optional[str] = None,
) -> Dict:
    """
    Bayesian optimization (Optuna/TPE) to maximize withdrawal_rate subject to survival constraint.

    Returns dict with:
      - best params
      - best value
      - trial metadata
      - and the Optuna study object for exporting trial history
    """
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets for weight optimization.")

    bootstrap_mode = fixed_bootstrap_mode or opt_config.bootstrap_mode

    def objective(trial: optuna.Trial) -> float:
        # 1) Sample raw weights and normalize
        raw_w = np.array([trial.suggest_float(f"w_{a}", 0.0, 1.0) for a in assets], dtype=float)
        w = _normalize_weights(raw_w)
        weights = {a: float(w[i]) for i, a in enumerate(assets)}

        # 2) Sample withdrawal rate and alpha
        withdrawal_rate = trial.suggest_float("withdrawal_rate", opt_config.withdrawal_min, opt_config.withdrawal_max)
        alpha = trial.suggest_float("alpha", opt_config.alpha_min, opt_config.alpha_max)

        # 3) Build simulator with trial config
        cfg = PortfolioConfig(
            initial_capital=base_config.initial_capital,
            withdrawal_rate=withdrawal_rate,
            horizon_years=base_config.horizon_years,
            rebalance_frequency=base_config.rebalance_frequency,
        )
        sim = MonteCarloSimulator(
            returns=returns_df,
            asset_weights=weights,
            config=cfg,
            periods_per_year=periods_per_year,
        )

        paths = sim.simulate_paths(
            n_paths=opt_config.n_paths_eval,
            random_state=opt_config.seed,
            bootstrap_mode=bootstrap_mode,      # "iid" or "block"
            block_size=opt_config.block_size,
            alpha=alpha,
        )

        terminal = paths[:, -1]
        survival = float(np.mean(terminal > 0.0))

        # ---- Commit 16 improvement: record trial diagnostics ----
        trial.set_user_attr("survival_rate", survival)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("bootstrap_mode", bootstrap_mode)
        trial.set_user_attr("block_size", opt_config.block_size)

        # Constraint: survival must meet target
        if survival < opt_config.target_survival:
            trial.set_user_attr("penalized", True)

            # heavy penalty to discourage these trials
            penalty = -1_000.0 - (opt_config.target_survival - survival) * 10_000.0
            return penalty

        trial.set_user_attr("penalized", False)

        # Objective: maximize withdrawal rate
        return withdrawal_rate

    sampler = optuna.samplers.TPESampler(seed=opt_config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=opt_config.n_trials)

    best = study.best_trial
    return {
        "best_value": best.value,
        "best_params": best.params,
        "best_trial_number": best.number,
        "n_trials": opt_config.n_trials,
        "target_survival": opt_config.target_survival,
        "bootstrap_mode": bootstrap_mode,
        "block_size": opt_config.block_size,
        "study": study,  # return the study so main.py can export trials
    }

