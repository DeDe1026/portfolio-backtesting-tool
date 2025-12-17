import numpy as np


def compute_survival_rate(terminal_values: np.ndarray, ruin_threshold: float = 0.0) -> float:
    if len(terminal_values) == 0:
        return 0.0
    return float(np.mean(terminal_values > ruin_threshold))
