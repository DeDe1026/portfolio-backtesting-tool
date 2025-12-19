from src.regime import RegimeConfig, build_regime_features, fit_kmeans_regimes, compute_regime_transition_matrix, sample_regime_sequence

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd


BootstrapMode = Literal["iid", "block"]


@dataclass
class PortfolioConfig:
    """
    Simulation configuration.

    withdrawal_rate is ANNUAL in decimal, e.g. 0.04 for 4%/year.

    In this implementation, withdrawals are a fixed amount per period,
    computed from initial_capital (classic "Trinity" style):
        withdrawal_per_period = initial_capital * withdrawal_rate / periods_per_year
    """
    initial_capital: float = 1_000_000.0
    withdrawal_rate: float = 0.04
    horizon_years: int = 30
    rebalance_frequency: Literal["none", "yearly"] = "yearly"


class MonteCarloSimulator:
    """
    Monte Carlo simulator with historical bootstrapping.

    Supports:
      - iid bootstrapping: sample months independently with replacement
      - block bootstrapping: sample blocks of consecutive months with replacement
      - multi-asset holdings (so rebalancing frequency actually matters)
      - fixed withdrawals per period
      - optional alpha withdrawal-cut rule on negative portfolio return
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        asset_weights: Dict[str, float],
        config: Optional[PortfolioConfig] = None,
        periods_per_year: int = 12,
    ) -> None:
        self.returns = returns.copy()
        self.asset_weights = asset_weights.copy()
        self.config = config or PortfolioConfig()
        self.periods_per_year = periods_per_year

        self._validate_inputs()

        # Keep asset order stable for matrix operations
        self.assets = list(self.asset_weights.keys())
        self.weights_vec = np.array([self.asset_weights[a] for a in self.assets], dtype=float)

    def _validate_inputs(self) -> None:
        if self.returns.empty:
            raise ValueError("Returns DataFrame is empty.")

        missing = [a for a in self.asset_weights if a not in self.returns.columns]
        if missing:
            raise ValueError(f"Assets not found in returns data columns: {missing}")

        w_sum = float(sum(self.asset_weights.values()))
        if not np.isclose(w_sum, 1.0):
            raise ValueError(f"Asset weights must sum to 1.0, got {w_sum}")

        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")

        if self.config.horizon_years <= 0:
            raise ValueError("horizon_years must be positive.")

        if self.config.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")

        if self.config.withdrawal_rate < 0:
            raise ValueError("withdrawal_rate must be >= 0.")

        if self.config.rebalance_frequency not in ("none", "yearly"):
            raise ValueError("rebalance_frequency must be 'none' or 'yearly'.")

    # -------------------------
    # Bootstrapping index samplers
    # -------------------------

    def _sample_indices_iid(self, rng: np.random.Generator, n_periods: int) -> np.ndarray:
        """
        Sample n_periods indices from historical data with replacement (iid months).
        """
        n_hist = len(self.returns)
        return rng.integers(low=0, high=n_hist, size=n_periods)

    def _sample_indices_block(self, rng: np.random.Generator, n_periods: int, block_size: int) -> np.ndarray:
        """
        Sample indices using block bootstrap:
        pick random start positions, take block_size consecutive indices,
        repeat until reaching n_periods.
        """
        if block_size <= 0:
            raise ValueError("block_size must be >= 1.")
        if block_size == 1:
            return self._sample_indices_iid(rng, n_periods)

        n_hist = len(self.returns)
        if n_hist < block_size:
            raise ValueError(f"Not enough historical data ({n_hist}) for block_size={block_size}.")

        max_start = n_hist - block_size
        indices: list[int] = []

        while len(indices) < n_periods:
            start = int(rng.integers(0, max_start + 1))
            block = list(range(start, start + block_size))
            indices.extend(block)

        return np.array(indices[:n_periods], dtype=int)

    def _sample_indices_regime(
    self,
    rng: np.random.Generator,
    n_periods: int,
    regime_k: int = 3,
    vol_window: int = 12,
    min_samples: int = 24,
) -> np.ndarray:
    """
    Regime bootstrapping:
      1) Cluster historical periods into regimes (K-means) using portfolio return + rolling vol.
      2) Fit a Markov transition matrix over regimes.
      3) Sample a regime path with the Markov chain.
      4) For each period, sample a historical index from the selected regime.

    Fallback: if not enough data for regimes, use iid indices.
    """
    cfg = RegimeConfig(k=regime_k, vol_window=vol_window, random_state=42, min_samples=min_samples)

    feats = build_regime_features(self.returns, self.asset_weights, cfg)
    labels, km, scaler = fit_kmeans_regimes(feats, cfg)

    if km is None:
        # Not enough data for stable regimes → fallback
        return self._sample_indices_iid(rng, n_periods)

    P = compute_regime_transition_matrix(labels, k=cfg.k)
    regime_seq = sample_regime_sequence(rng, n_periods=n_periods, P=P)

    # Map regime -> list of historical indices
    buckets: list[list[int]] = [[] for _ in range(cfg.k)]
    for i, r in enumerate(labels):
        buckets[int(r)].append(i)

    # If any regime bucket empty (can happen), fallback to iid
    if any(len(b) == 0 for b in buckets):
        return self._sample_indices_iid(rng, n_periods)

    # For each period, sample a historical index from that regime
    idx = np.zeros(n_periods, dtype=int)
    for t in range(n_periods):
        reg = int(regime_seq[t])
        idx[t] = int(rng.choice(buckets[reg]))

    return idx


    # -------------------------
    # Core simulation
    # -------------------------

    def simulate_paths(
        self,
        n_paths: int = 1000,
        random_state: Optional[int] = None,
        bootstrap_mode: BootstrapMode = "iid",
        block_size: int = 12,
        alpha: float = 0.0,
        floor_at_zero: bool = True,
    ) -> np.ndarray:
        """
        Simulate portfolio wealth paths.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths.
        random_state : int, optional
            Seed for reproducibility.
        bootstrap_mode : "iid" or "block"
            Bootstrapping method for returns.
        block_size : int
            Block length in periods (months), used when bootstrap_mode="block".
        alpha : float
            Withdrawal cut fraction applied in months where portfolio return < 0.
            Example: alpha=0.10 means withdraw 10% less in negative-return months.
        floor_at_zero : bool
            If True, portfolio wealth cannot go below 0 (ruin stays at 0).

        Returns
        -------
        np.ndarray
            Wealth paths of shape (n_paths, n_periods + 1).
            Column 0 is initial capital, then period-by-period wealth.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if alpha < 0 or alpha >= 1:
            raise ValueError("alpha must be in [0, 1).")

        rng = np.random.default_rng(random_state)
        n_periods = self.config.horizon_years * self.periods_per_year

        # Fixed withdrawal amount per period (classic constant-dollar rule)
        base_withdrawal = self.config.initial_capital * self.config.withdrawal_rate / self.periods_per_year

        # Prepare return matrix for selected assets
        hist_returns = self.returns[self.assets].to_numpy(dtype=float)  # shape (n_hist, n_assets)

        paths = np.zeros((n_paths, n_periods + 1), dtype=float)
        paths[:, 0] = self.config.initial_capital

        for p in range(n_paths):
            # Sample indices
            if bootstrap_mode == "iid":
                idx = self._sample_indices_iid(rng, n_periods)
            elif bootstrap_mode == "block":
                idx = self._sample_indices_block(rng, n_periods, block_size=block_size)
            elif bootstrap_mode == "regime":
                idx = self._sample_indices_regime(rng, n_periods, regime_k=3, vol_window=12, min_samples=24,)
            else:
                raise ValueError(f"Unknown bootstrap_mode: {bootstrap_mode}")

            # Sampled asset returns for this path: (n_periods, n_assets)
            R = hist_returns[idx, :]

            # Holdings per asset
            holdings = self.config.initial_capital * self.weights_vec.copy()

            for t in range(n_periods):
                # 1) Apply asset returns to holdings
                holdings *= (1.0 + R[t, :])

                total_before_withdraw = float(holdings.sum())

                # If ruined
                if floor_at_zero and total_before_withdraw <= 0:
                    holdings[:] = 0.0
                    paths[p, t + 1] = 0.0
                    continue

                # 2) Compute portfolio return for alpha rule
                # portfolio return = weighted return based on holdings weights *at start of month*
                # Here we approximate using current weights after growth (acceptable for this project).
                if total_before_withdraw > 0:
                    current_w = holdings / total_before_withdraw
                    port_r = float(np.dot(current_w, R[t, :]))
                else:
                    port_r = -1.0

                # 3) Withdraw
                withdrawal = base_withdrawal
                if port_r < 0 and alpha > 0:
                    withdrawal *= (1.0 - alpha)

                # Withdraw proportionally from all holdings
                if withdrawal > 0 and total_before_withdraw > 0:
                    withdraw_frac = min(1.0, withdrawal / total_before_withdraw)
                    holdings *= (1.0 - withdraw_frac)

                total_after_withdraw = float(holdings.sum())
                if floor_at_zero and total_after_withdraw < 0:
                    holdings[:] = 0.0
                    total_after_withdraw = 0.0

                # 4) Rebalance (yearly)
                if self.config.rebalance_frequency == "yearly":
                    # rebalance at end of each year (after withdrawals)
                    if (t + 1) % self.periods_per_year == 0 and total_after_withdraw > 0:
                        holdings = total_after_withdraw * self.weights_vec.copy()

                paths[p, t + 1] = total_after_withdraw

        return paths

