from pathlib import Path
import json
import numpy as np
import argparse

from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig
from src.evaluation import compute_survival_rate, summarize_terminal_wealth
from src.optimization import OptimizationConfig, optimize_portfolio, export_optuna_trials_csv
from src.experiments import CompareConfig, run_bootstrap_comparison
from src.compare_plots import save_survival_bar, save_terminal_boxplot, save_drawdown_boxplot
from src.build_dataset import DatasetConfig, build_monthly_returns_dataset

def run_case(
    sim,
    label: str,
    results_dir: Path,
    n_paths: int,
    random_state: int,
    bootstrap_mode: str,
    alpha: float,
    block_size: int | None = None,
    regime_k: int = 3,
    regime_vol_window: int = 12,
    regime_min_samples: int = 24,):

    print("\n" + "-" * 60)
    print(label)

    paths = sim.simulate_paths(
        n_paths=n_paths,
        random_state=random_state,
        bootstrap_mode=bootstrap_mode,
        block_size=block_size,
        alpha=alpha, 
        regime_k=regime_k,
        regime_vol_window=regime_vol_window,
        regime_min_samples=regime_min_samples,)

    terminal = paths[:, -1]
    survival = float(np.mean(terminal > 0.0))

    summary = {
        "label": label,
        "bootstrap_mode": bootstrap_mode,
        "alpha": alpha,
        "survival_rate": survival,
        "median_terminal_wealth": float(np.median(terminal)),
        "p10_terminal_wealth": float(np.percentile(terminal, 10)),
        "p90_terminal_wealth": float(np.percentile(terminal, 90)),
    }

    print(f"Survival rate: {survival * 100:.1f}%")
    print(f"Median terminal wealth: {summary['median_terminal_wealth']:.0f}")
    print(
        f"P10: {summary['p10_terminal_wealth']:.0f} | "
        f"P90: {summary['p90_terminal_wealth']:.0f}"
    )

    out = results_dir / f"{label.replace(' ', '_').lower()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo portfolio backtesting with multiple bootstrap modes")

    parser.add_argument("--n-paths", type=int, default=2000, help="Number of Monte Carlo paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--alpha", type=float, default=0.0, help="Withdrawal reduction factor")
    parser.add_argument("--block-size", type=int, default=12, help="Block size for block bootstrap")

    parser.add_argument("--regime-k", type=int, default=3, help="Number of regimes (K-means)")
    parser.add_argument(
        "--regime-vol-window",
        type=int,
        default=12,
        help="Rolling volatility window for regime features",)
    parser.add_argument(
        "--regime-min-samples",
        type=int,
        default=24,
        help="Minimum samples required to activate regime bootstrapping",)

    parser.add_argument("--build-data", action="store_true", help="Download/build monthly dataset into data/raw/")
    parser.add_argument("--data-start", type=str, default="1975-01-01", help="Start date for data download (YYYY-MM-DD)")
    parser.add_argument("--data-file", type=str, default="data/raw/monthly_returns_native.csv", help="Path to monthly returns CSV")


    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if args.build_data:
        build_monthly_returns_dataset(
        DatasetConfig(start=args.data_start, out_filename=Path(args.data_file).name))

        print("Data build complete. Exiting because --build-data was set.")
        return
    
    project_root = Path(__file__).resolve().parent

    # Load the dataset the user selected (default: data/raw/monthly_returns_native.csv)
    data_path = (project_root / args.data_file).resolve()

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading returns from: {data_path}")
    returns_df = load_returns_data(data_path)
    print(f"Loaded returns shape: {returns_df.shape}")
    print(f"Columns: {list(returns_df.columns)}")
    print(returns_df.head())
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

    compare_cfg = CompareConfig(
        n_paths=args.n_paths,
        random_state=args.seed,
        alpha=args.alpha,
        block_size=args.block_size,
        regime_k=args.regime_k,
        regime_vol_window=args.regime_vol_window,
        regime_min_samples=args.regime_min_samples,)

        
        # keep small for dummy data; later set 12
        

    summary_df, outputs = run_bootstrap_comparison(sim, compare_cfg)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "compare_bootstrap_summary.csv"
    summary_df.to_csv(summary_csv)
    print("\nSaved comparison summary to:", summary_csv)
    print(summary_df)

    save_survival_bar(summary_df, results_dir / "compare_survival_bar.png")
    save_terminal_boxplot(outputs, results_dir / "compare_terminal_boxplot.png")
    save_drawdown_boxplot(outputs, results_dir / "compare_drawdown_boxplot.png")
    print("Saved comparison plots in:", results_dir)


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
        label="Regime bootstrap (K=3, vol=12)",
        results_dir=results_dir,
        n_paths=500,
        random_state=42,
        bootstrap_mode="regime",
        alpha=0.0,
        regime_k=3,
        regime_vol_window=12,
        regime_min_samples=24,)



if __name__ == "__main__":
    main()

