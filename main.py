from __future__ import annotations

from pathlib import Path
import json
import argparse

from src.data_loader import load_returns_data
from src.optimization import OptimizationConfig
from src.build_dataset import DatasetConfig, build_monthly_returns_dataset
from src.pipeline import (
    normalize_weights,
    apply_fx_conversion_to_chf,
    run_basic_and_optimized,
    save_weights_pie,
    save_median_paths_plot,
)

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
    parser.add_argument("--inflation-aware-withdrawals", action="store_true", help="Increase withdrawals by realized inflation (keeps spending power constant)")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--horizon-years", type=int, default=30)
    parser.add_argument("--w-pref", type=float, default=4000.0)
    parser.add_argument("--w-floor", type=float, default=3000.0)

    parser.add_argument("--opt-mode", choices=["A","B","C"], default="B")
    parser.add_argument("--target-survival", type=float, default=0.95)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--alpha-cap", type=float, default=0.50)
    parser.add_argument("--withdraw-mult-max", type=float, default=2.0)

    parser.add_argument("--fx-col", type=str, default="usdchf")
    parser.add_argument("--usd-assets", type=str, default="", help="Comma-separated USD columns to FX convert")
    parser.add_argument("--inflation-col", type=str, default="ch_inflation")

    parser.add_argument("--weights", type=str, default="", help="Comma-separated asset=weight, e.g. smi_chf=0.5,sp500_usd=0.5")


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
    fx_col = args.fx_col
    inflation_col = args.inflation_col

    usd_assets = [s.strip() for s in args.usd_assets.split(",") if s.strip()]
    if not usd_assets:
        usd_assets = [c for c in returns_df.columns if c.endswith("_usd")]

    returns_df_fx = apply_fx_conversion_to_chf(returns_df, fx_col=fx_col, usd_asset_cols=usd_assets)

    assets = [c for c in returns_df_fx.columns if c not in [inflation_col, fx_col]]
    if len(assets) < 2:
        raise ValueError(f"Need at least 2 investable assets after excluding [{inflation_col}, {fx_col}]. Got: {assets}")

    if args.weights.strip():
        w_raw = {}
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            w_raw[k.strip()] = float(v.strip())
        weights_guess = normalize_weights(w_raw)
    else:
        weights_guess = {a: 1.0 / len(assets) for a in assets}

    if args.opt_mode == "A":
        mode = "A_survival_weights_only"
    elif args.opt_mode == "B":
        mode = "B_withdraw_max_subject_survival"
    else:
        mode = "C_survival_weights_alpha"

    withdrawal_rate_pref = (12.0 * args.w_pref) / args.initial_capital

    alpha_max_floor = 0.0
    if args.w_pref > 0:
        alpha_max_floor = max(0.0, 1.0 - (args.w_floor / args.w_pref))

    fixed_alpha = float(alpha_max_floor) if mode == "A_survival_weights_only" else None
    alpha_max = float(alpha_max_floor if mode == "A_survival_weights_only" else args.alpha_cap)

    opt_cfg = OptimizationConfig(
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        target_survival=float(args.target_survival),
        n_paths_eval=min(int(args.n_paths), 5000),
        mode=mode,
        alpha_min=0.0,
        alpha_max=alpha_max,
        fixed_alpha=fixed_alpha,
        bootstrap_mode="iid",
        block_size=int(args.block_size),
        withdraw_mult_min=1.0,
        withdraw_mult_max=float(args.withdraw_mult_max),
    )


    pipe = run_basic_and_optimized(
        returns_df_fx=returns_df_fx,
        assets=assets,
        weights_guess=weights_guess,
        initial_capital=float(args.initial_capital),
        horizon_years=int(args.horizon_years),
        w_pref=float(args.w_pref),
        w_floor=float(args.w_floor),
        inflation_toggle=bool(args.inflation_aware_withdrawals),
        inflation_col=str(inflation_col),
        opt_cfg=opt_cfg,
        n_paths=int(args.n_paths),
        seed=int(args.seed),
        block_size=int(args.block_size),
        regime_k=int(args.regime_k),
        regime_vol_window=int(args.regime_vol_window),
        regime_min_samples=int(args.regime_min_samples),
    )

    comp_df = pipe["comp_df"]
    perf_df = pipe["perf_df"]
    basic_results = pipe["basic_results"]
    opt_results = pipe["opt_results"]
    opt_result = pipe["opt_result"]

    comp_path = results_dir / "comp_basic_vs_optimized.csv"
    perf_path = results_dir / "perf_basic_vs_optimized.csv"
    comp_df.to_csv(comp_path, index=False)
    perf_df.to_csv(perf_path, index=False)

    # opt_result contains Optuna study (not JSON-serializable)
    opt_result = dict(opt_result)  # shallow copy to avoid side effects
    opt_result.pop("study", None)

    with open(results_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(opt_result, f, indent=2)

    save_weights_pie(opt_result["best_weights"], results_dir / "best_weights_pie.png", "Optimized weights")
    save_median_paths_plot(basic_results, opt_results, results_dir / "median_paths_basic_vs_opt.png")

    print("\n=== Bootstrap comparison (BASIC vs OPTIMIZED) ===")
    print(comp_df.to_string(index=False))

    print("\n=== Median-path performance (net of withdrawals) ===")
    print(perf_df.to_string(index=False))

    print("\nSaved:")
    print(" -", comp_path)
    print(" -", perf_path)
    print(" -", results_dir / "best_params.json")
    print(" -", results_dir / "best_weights_pie.png")
    print(" -", results_dir / "median_paths_basic_vs_opt.png")


if __name__ == "__main__":
    main()

