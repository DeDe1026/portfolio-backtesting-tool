from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_survival_rate(terminal_values: np.ndarray, ruin_threshold: float = 0.0) -> float:
    if terminal_values.size == 0:
        return 0.0
    return float(np.mean(terminal_values > ruin_threshold))


def summarize_terminal_wealth(terminal_values: np.ndarray) -> dict:
    if terminal_values.size == 0:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "mean": 0.0}
    return {
        "mean": float(np.mean(terminal_values)),
        "median": float(np.median(terminal_values)),
        "p10": float(np.percentile(terminal_values, 10)),
        "p90": float(np.percentile(terminal_values, 90)),
        "min": float(np.min(terminal_values)),
        "max": float(np.max(terminal_values)),
    }


def plot_terminal_histogram(
    terminal_values: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 50,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(terminal_values, bins=bins)
    plt.title(title)
    plt.xlabel("Terminal wealth")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sample_paths(
    paths: np.ndarray,
    title: str,
    out_path: Path,
    n_lines: int = 30,
) -> None:
    """
    paths shape: (n_paths, n_periods+1)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_paths = paths.shape[0]
    n_lines = min(n_lines, n_paths)

    plt.figure()
    for i in range(n_lines):
        plt.plot(paths[i, :])
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Wealth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_summary_json(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

