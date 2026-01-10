from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import optuna
import pandas as pd

from src.models import MonteCarloSimulator, PortfolioConfig


OptimizationMode = Literal["A_survival_weights_only", "B_withdraw_max_subject_survival", "C_survival_weights_alpha"]


@dataclass
class OptimizationConfig:
    n_trials: int = 50
    seed: int = 42
    target_survival: float = 0.95
    n_paths_eval: int = 1000

    # Which optimization logic to use
    mode: OptimizationMode = "B_withdraw_max_subject_survival"

    # Search ranges
    withdrawal_min: float = 0.01
    withdrawal_max: float = 0.08
    # Alpha range
    alpha_min: float = 0.0
    alpha_max: float = 0.50

    # For Mode B: multiply the user's baseline withdrawal rate
    withdraw_mult_min: float = 1.0
    withdraw_mult_max: float = 2.0

    # For Mode A: enforce alpha implied by floor 
    preferred_withdrawal: Optional[float] = None  # CHF/month
    withdrawal_floor: Optional[float] = None      # CHF/month


    # Bootstrap settings during optimization
    bootstrap_mode: Literal["iid", "block"] = "block"
    block_size: int = 12


def _normalize_weights(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw, 1e-12, None)
    return raw / raw.sum()


def export_optuna_trials_csv(study: optuna.Study, out_path: str) -> None:
    rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "objective_value": t.value,
            "state": str(t.state),
        }
        for k, v in t.params.items():
            row[k] = v

        row["survival_rate"] = t.user_attrs.get("survival_rate")
        row["penalized"] = t.user_attrs.get("penalized", False)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def _evaluate_trial(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    base_config: PortfolioConfig,
    withdrawal_rate: float,
    alpha: float,
    opt_config: OptimizationConfig,
    periods_per_year: int,
    bootstrap_mode: str,
) -> tuple[float, float]:
    """
    Returns (survival, median_terminal_wealth) for the trial.
    """
    cfg = PortfolioConfig(
        initial_capital=base_config.initial_capital,
        withdrawal_rate=withdrawal_rate,
        horizon_years=base_config.horizon_years,
        rebalance_frequency=base_config.rebalance_frequency,
        inflation_aware_withdrawals=getattr(base_config, "inflation_aware_withdrawals", False),
        inflation_col=getattr(base_config, "inflation_col", "ch_inflation_mom"),
    )

    sim = MonteCarloSimulator(
        returns=returns_df,
        asset_weights=weights,
        config=cfg,
        periods_per_year=periods_per_year,
    )

    use_floor_rule = (opt_config.mode == "A_survival_weights_only")

    paths = sim.simulate_paths(
        n_paths=opt_config.n_paths_eval,
        random_state=opt_config.seed,
        bootstrap_mode=bootstrap_mode,
        block_size=opt_config.block_size,
        alpha=alpha,
        withdrawal_rule=("neg_to_floor" if use_floor_rule else "alpha_cut"),
        preferred_withdrawal=(opt_config.preferred_withdrawal if use_floor_rule else None),
        withdrawal_floor=(opt_config.withdrawal_floor if use_floor_rule else None),
    )

    terminal = paths[:, -1]
    survival = float(np.mean(terminal > 0.0))
    med_terminal = float(np.median(terminal))
    return survival, med_terminal


def optimize_portfolio(
    returns_df: pd.DataFrame,
    assets: list[str],
    base_config: PortfolioConfig,
    opt_config: OptimizationConfig,
    periods_per_year: int = 12,
    fixed_bootstrap_mode: Optional[str] = None,
) -> Dict:
    """
    Bayesian optimization (Optuna/TPE) with 3 modes:

    A) mode="A_survival_weights_only"
       - optimize weights only
       - alpha is 0 because floor rule is used
       - withdrawal_rate is base_config.withdrawal_rate
       - objective: maximize survival (tie-break with terminal wealth)

    B) mode="B_withdraw_max_subject_survival"
       - user provided a baseline withdrawal_rate (from desired CHF/month)
       - optimize weights + alpha + withdraw_mult (>=1)
       - withdrawal_rate = base_rate * withdraw_mult
       - constraint: survival >= target_survival
       - objective: maximize withdrawal_rate

    C) mode="C_survival_weights_alpha"
       - optimize weights + alpha
       - withdrawal_rate fixed (base_config.withdrawal_rate)
       - objective: maximize survival (tie-break with terminal wealth)
    """
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets for weight optimization.")

    bootstrap_mode = fixed_bootstrap_mode or opt_config.bootstrap_mode
    base_rate = float(base_config.withdrawal_rate)

    def objective(trial: optuna.Trial) -> float:
        # 1) sample weights
        raw_w = np.array([trial.suggest_float(f"w_{a}", 0.0, 1.0) for a in assets], dtype=float)
        w = _normalize_weights(raw_w)
        weights = {a: float(w[i]) for i, a in enumerate(assets)}

        # 2) choose alpha + withdrawal rate depending on mode
        if opt_config.mode == "A_survival_weights_only":
            alpha = 0.0
            withdrawal_rate = base_rate

        elif opt_config.mode == "B_withdraw_max_subject_survival":
            alpha = trial.suggest_float("alpha", opt_config.alpha_min, opt_config.alpha_max)
            mult = trial.suggest_float("withdraw_mult", opt_config.withdraw_mult_min, opt_config.withdraw_mult_max)
            withdrawal_rate = base_rate * mult

        elif opt_config.mode == "C_survival_weights_alpha":
            alpha = trial.suggest_float("alpha", opt_config.alpha_min, opt_config.alpha_max)
            withdrawal_rate = base_rate

        else:
            raise ValueError(f"Unknown opt_config.mode: {opt_config.mode}")

        # 3) evaluate
        survival, med_terminal = _evaluate_trial(
            returns_df=returns_df,
            weights=weights,
            base_config=base_config,
            withdrawal_rate=withdrawal_rate,
            alpha=alpha,
            opt_config=opt_config,
            periods_per_year=periods_per_year,
            bootstrap_mode=bootstrap_mode,
        )

        # record diagnostics
        trial.set_user_attr("survival_rate", survival)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("bootstrap_mode", bootstrap_mode)
        trial.set_user_attr("block_size", opt_config.block_size)

        # ---- Objectives ----
        if opt_config.mode == "B_withdraw_max_subject_survival":
            # Hard survival constraint; penalize heavily if not met
            if survival < opt_config.target_survival:
                trial.set_user_attr("penalized", True)
                shortfall = opt_config.target_survival - survival
                return -1_000.0 - 10_000.0 * shortfall

            trial.set_user_attr("penalized", False)
            # maximize withdrawal rate
            return float(withdrawal_rate)

        else:
            # maximize survival (tie-break by terminal wealth)
            trial.set_user_attr("penalized", False)
            return float(survival + 1e-6 * np.log1p(max(med_terminal, 1e-9)))

    sampler = optuna.samplers.TPESampler(seed=opt_config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=opt_config.n_trials)

    best = study.best_trial
    best_params = dict(best.params)

    # reconstruct best weights
    raw_best = np.array([best_params.get(f"w_{a}", 0.0) for a in assets], dtype=float)
    w_best = _normalize_weights(raw_best)
    best_weights = {a: float(w_best[i]) for i, a in enumerate(assets)}

    # infer alpha and withdrawal rate
    if opt_config.mode == "A_survival_weights_only":
        best_alpha = 0.0
        best_withdraw_mult = 1.0
        best_withdrawal_rate = base_rate

    elif opt_config.mode == "B_withdraw_max_subject_survival":
        best_alpha = float(best_params["alpha"])
        best_withdraw_mult = float(best_params["withdraw_mult"])
        best_withdrawal_rate = base_rate * best_withdraw_mult

    else:  # C
        best_alpha = float(best_params["alpha"])
        best_withdraw_mult = 1.0
        best_withdrawal_rate = base_rate

    return {
        "best_weights": best_weights,
        "best_alpha": best_alpha,
        "best_withdrawal_rate": float(best_withdrawal_rate),
        "best_withdraw_mult": float(best_withdraw_mult),
        "best_trial_number": best.number,
        "n_trials": opt_config.n_trials,
        "target_survival": opt_config.target_survival,
        "mode": opt_config.mode,
        "bootstrap_mode": bootstrap_mode,
        "block_size": opt_config.block_size,
        "study": study,
    }


