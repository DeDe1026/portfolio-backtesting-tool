from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeConfig:
    k: int = 3  # number of regimes
    vol_window: int = 12  # rolling window for volatility features (in periods)
    random_state: int = 42
    min_samples: int = 24  # minimum rows required to do regimes; else fallback


def _monthly_portfolio_return(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets], dtype=float)
    X = returns[assets].to_numpy(dtype=float)
    port = X @ w
    return pd.Series(port, index=returns.index, name="port_ret")


def build_regime_features(
    returns: pd.DataFrame,
    weights: dict[str, float],
    cfg: RegimeConfig,
) -> pd.DataFrame:
    """
    Build simple, explainable features for clustering regimes.
    Features (per period):
      - portfolio return
      - rolling volatility of portfolio return
    """
    port_ret = _monthly_portfolio_return(returns, weights)

    # rolling volatility; use min_periods=2 so small datasets still get some values
    roll_vol = port_ret.rolling(cfg.vol_window, min_periods=2).std()

    feats = pd.DataFrame(
        {
            "port_ret": port_ret,
            "roll_vol": roll_vol,
        },
        index=returns.index,
    )

    # Fill remaining NaNs (first few rows) conservatively
    feats = feats.fillna(method="bfill").fillna(method="ffill")
    return feats


def fit_kmeans_regimes(
    features: pd.DataFrame,
    cfg: RegimeConfig,
) -> tuple[np.ndarray, Optional[KMeans], Optional[StandardScaler]]:
    """
    Returns:
      labels: array of shape (n_samples,)
      model, scaler (None if not enough samples)
    """
    n = len(features)
    if n < max(cfg.min_samples, cfg.k * 3):
        # Not enough data to form stable clusters; caller should fallback
        labels = np.zeros(n, dtype=int)
        return labels, None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(features.to_numpy(dtype=float))

    km = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init=10)
    labels = km.fit_predict(X)

    return labels.astype(int), km, scaler


def compute_regime_transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    """
    Transition matrix P where P[i,j] = P(next_regime=j | current=i)
    """
    P = np.zeros((k, k), dtype=float)
    if len(labels) < 2:
        P[:] = 1.0 / k
        return P

    for a, b in zip(labels[:-1], labels[1:]):
        if 0 <= a < k and 0 <= b < k:
            P[a, b] += 1.0

    # Row-normalize with smoothing
    for i in range(k):
        row_sum = P[i].sum()
        if row_sum <= 0:
            P[i] = 1.0 / k
        else:
            P[i] = P[i] / row_sum

    return P


def sample_regime_sequence(
    rng: np.random.Generator,
    n_periods: int,
    P: np.ndarray,
    start_regime: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a regime sequence using a Markov chain with transition matrix P.
    """
    k = P.shape[0]
    seq = np.zeros(n_periods, dtype=int)

    if start_regime is None:
        seq[0] = int(rng.integers(0, k))
    else:
        seq[0] = int(start_regime)

    for t in range(1, n_periods):
        current = seq[t - 1]
        seq[t] = int(rng.choice(np.arange(k), p=P[current]))

    return seq
