from pathlib import Path

from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig
from src.evaluation import (
    compute_survival_rate,
    summarize_terminal_wealth,
    plot_terminal_histogram,
    plot_sample_paths,
    save_summary_json,
)


def run_case(sim: MonteCarloSimulator, label: str, results_dir: Path, **kwargs) -> None:
    paths = sim.simulate_paths(**kwargs)
    terminal = paths[:, -1]
    survival = compute_survival_rate(terminal)
    summary = summarize_terminal_wealth(terminal)
    summary["survival_rate"] = survival
    summary["label"] = label
    summary["params"] = kwargs

    print("\n" + "-" * 60)
    print(label)
    print(f"Paths shape: {paths.shape}")
    print(f"Survival rate: {survival*100:.1f}%")
    print(f"Median terminal wealth: {summary['median']:.0f}")
    print(f"P10: {summary['p10']:.0f} | P90: {summary['p90']:.0f}")

    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")

    # Save artifacts
    plot_terminal_histogram(
        terminal_values=terminal,
        title=f"Terminal wealth: {label}",
        out_path=results_dir / f"{safe_label}_terminal_hist.png",
        bins=40,
    )
    plot_sample_paths(
        paths=paths,
        title=f"Sample paths: {label}",
        out_path=results_dir / f"{safe_label}_sample_paths.png",
        n_lines=30,
    )
    save_summary_json(
        summary=summary,
        out_path=results_dir / f"{safe_label}_summary.json",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "example_returns.csv"
    results_dir = project_root / "results"

    returns_df = load_returns_data(data_path)
    print("Loaded returns columns:", list(returns_df.columns))
    print(returns_df.head())

    weights = {"stocks": 0.6, "bonds": 0.4}
    config = PortfolioConfig(
        initial_capital=1_000_000.0,
        withdrawal_rate=0.04,
        horizon_years=30,
        rebalance_frequency="yearly",
    )
    sim = MonteCarloSimulator(returns_df, weights, config, periods_per_year=12)

    run_case(
        sim,
        label="IID monthly bootstrap (alpha=0)",
        results_dir=results_dir,
        n_paths=500,
        random_state=42,
        bootstrap_mode="iid",
        alpha=0.0,
    )

    run_case(
        sim,
        label="Block bootstrap (3 months) (alpha=0)",
        results_dir=results_dir,
        n_paths=500,
        random_state=42,
        bootstrap_mode="block",
        block_size=3,
        alpha=0.0,
    )

    run_case(
        sim,
        label="IID monthly bootstrap with alpha=10%",
        results_dir=results_dir,
        n_paths=500,
        random_state=42,
        bootstrap_mode="iid",
        alpha=0.10,
    )

    print("\nSaved plots + summaries in:", results_dir)


if __name__ == "__main__":
    main()

