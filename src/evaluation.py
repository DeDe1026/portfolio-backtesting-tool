from __future__ import annotations
import numpy as np


def compute_survival_rate(terminal_values: np.ndarray, ruin_threshold: float = 0.0) -> float:
    if terminal_values.size == 0:
        return 0.0
    return float(np.mean(terminal_values > ruin_threshold))


def summarize_terminal_wealth(terminal_values: np.ndarray) -> dict:
    if terminal_values.size == 0:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "median": float(np.median(terminal_values)),
        "p10": float(np.percentile(terminal_values, 10)),
        "p90": float(np.percentile(terminal_values, 90)),
    }

