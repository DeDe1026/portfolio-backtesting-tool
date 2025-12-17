from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    withdrawal_rate: float = 0.04
    horizon_years: int = 30
    rebalance_frequency: str = "yearly"


class MonteCarloSimulator:
    def __init__(
        self,
        returns: pd.DataFrame,
        asset_weights: Dict[str, float],
        config: Optional[PortfolioConfig] = None,
        periods_per_year: int = 12,
    ) -> None:
        self.returns = returns
        self.asset_weights = asset_weights
        self.config = config or PortfolioConfig()
        self.periods_per_year = periods_per_year
        self._validate_weights()

    def _validate_weights(self) -> None:
        missing = [a for a in self.asset_weights if a not in self.returns.columns]
        if missing:
            raise ValueError(f"Assets not found in returns data: {missing}")

        w_sum = sum(self.asset_weights.values())
        if not np.isclose(w_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {w_sum}")

    def simulate_paths(self, n_paths: int = 100, random_state: int | None = None) -> np.ndarray:
        n_periods = self.config.horizon_years * self.periods_per_year
        print("[INFO] Placeholder simulate_paths() — returns zeros.")
        return np.zeros((n_paths, n_periods))
