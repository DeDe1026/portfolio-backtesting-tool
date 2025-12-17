from pathlib import Path

from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig
from src.evaluation import compute_survival_rate, summarize_terminal_wealth


def run_case(sim: MonteCarloSimulator, label: str, **kwargs) -> None:
    paths = sim.simulate_paths(**kwargs)
    terminal = paths[:, -1]
    survival = compute_survival_rate(terminal)
    summary = summarize_terminal_wealth(terminal)

    print("\n" + "-" * 60)
    print(label)
    print(f"Paths shape: {paths.shape}")
    print(f"Survival rate: {survival*100:.1f}%")
    print(f"Median terminal wealth: {summary['median']:.0f}")
    print(f"P10: {summary['p10']:.0f} | P90: {summary['p90']:.0f}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "example_returns.csv"

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

    # Case 1: iid bootstrapping
    run_case(
        sim,
        label="IID monthly bootstrap (alpha=0)",
        n_paths=500,
        random_state=42,
        bootstrap_mode="iid",
        alpha=0.0,
    )

    # Case 2: block bootstrapping
    run_case(
        sim,
        label="Block bootstrap (12 months) (alpha=0)",
        n_paths=500,
        random_state=42,
        bootstrap_mode="block",
        block_size=4,
        alpha=0.0,
    )

    # Case 3: iid with alpha rule
    run_case(
        sim,
        label="IID monthly bootstrap with alpha=10%",
        n_paths=500,
        random_state=42,
        bootstrap_mode="iid",
        alpha=0.10,
    )


if __name__ == "__main__":
    main()

