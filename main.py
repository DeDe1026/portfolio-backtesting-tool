from pathlib import Path
from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "example_returns.csv"

    returns_df = load_returns_data(data_path)
    print("Loaded returns:")
    print(returns_df.head())

    weights = {"stocks": 0.6, "bonds": 0.4}
    config = PortfolioConfig(initial_capital=1_000_000.0, withdrawal_rate=0.04, horizon_years=30)
    sim = MonteCarloSimulator(returns_df, weights, config)

    paths = sim.simulate_paths(n_paths=10, random_state=42)
    print("Paths shape:", paths.shape)


if __name__ == "__main__":
    main()
