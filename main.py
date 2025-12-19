from pathlib import Path
import json

from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig
from src.evaluation import compute_survival_rate, summarize_terminal_wealth
from src.optimization import OptimizationConfig, optimize_portfolio, export_optuna_trials_csv


def run_quick_eval(sim: MonteCarloSimulator, label: str, **kwargs) -> None:
    paths = sim.simulate_paths(**kwargs)
    terminal = paths[:, -1]
    survival = compute_survival_rate(terminal)
    summary = summarize_terminal_wealth(terminal)

    print("\n" + "-" * 60)
    print(label)
    print(f"Survival rate: {survival*100:.1f}%")
    print(f"Median terminal wealth: {summary['median']:.0f}")
    print(f"P10: {summary['p10']:.0f} | P90: {summary['p90']:.0f}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "example_returns.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    returns_df = load_returns_data(data_path)
    assets = list(returns_df.columns)

    base_config = PortfolioConfig(
        initial_capital=1_000_000.0,
        withdrawal_rate=0.04,
        horizon_years=30,
        rebalance_frequency="yearly",)

    # Baseline simulator (fixed weights)
    if len(assets) >= 2:
        weights = {assets[0]: 0.6, assets[1]: 0.4}
    else:
        weights = {assets[0]: 1.0}

    sim = MonteCarloSimulator(returns_df, weights, base_config, periods_per_year=12)

    run_quick_eval(
        sim,
        label="Baseline IID (4% WR, alpha=0)",
        n_paths=2000,
        random_state=42,
        bootstrap_mode="iid",
        alpha=0.0,)

    # --- Optimization ---
    opt_cfg = OptimizationConfig(
        n_trials=50,
        seed=42,
        target_survival=0.95,
        n_paths_eval=2000,
        withdrawal_min=0.01,
        withdrawal_max=0.08,
        alpha_min=0.0,
        alpha_max=0.30,
        bootstrap_mode="iid",
        block_size=3,)  # keep small for dummy dataset; later set to 12+ with real data

    print("\nStarting optimization...")
    opt_result = optimize_portfolio(
        returns_df=returns_df,
        assets=assets,
        base_config=base_config,
        opt_config=opt_cfg,
        periods_per_year=12,)

    # Extract study (so we can export trials)
    study = opt_result.pop("study")

    print("\nOptimization result:")
    print(opt_result)

    # Save best result JSON
    out_best = results_dir / "optuna_best_params.json"
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(opt_result, f, indent=2)
    print(f"\nSaved best params to: {out_best}")

    # ---- Commit 16 improvement: export all trials ----
    out_trials = results_dir / "optuna_trials.csv"
    export_optuna_trials_csv(study, str(out_trials))
    print(f"Saved Optuna trial history to: {out_trials}")

    # Optional: evaluate best solution
    best_params = opt_result["best_params"]

    # Normalize weights again (they were raw 0..1)
    raw_w = [best_params[f"w_{a}"] for a in assets]
    s = sum(raw_w) if sum(raw_w) > 0 else 1.0
    best_weights = {a: float(best_params[f"w_{a}"] / s) for a in assets}

    best_wr = float(best_params["withdrawal_rate"])
    best_alpha = float(best_params["alpha"])

    best_cfg = PortfolioConfig(
        initial_capital=base_config.initial_capital,
        withdrawal_rate=best_wr,
        horizon_years=base_config.horizon_years,
        rebalance_frequency=base_config.rebalance_frequency,)
    best_sim = MonteCarloSimulator(returns_df, best_weights, best_cfg, periods_per_year=12)

    run_quick_eval(
        best_sim,
        label=f"Best found IID (WR={best_wr:.3%}, alpha={best_alpha:.1%})",
        n_paths=5000,
        random_state=123,
        bootstrap_mode="iid",
        alpha=best_alpha,
        block_size=opt_cfg.block_size,)

    run_case(
        sim,
        label="Regime bootstrap (K-means k=3) (alpha=0)",
        results_dir=results_dir,
        n_paths=500,
        random_state=42,
        bootstrap_mode="regime",
        alpha=0.0,)



if __name__ == "__main__":
    main()

