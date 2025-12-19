from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_survival_bar(summary_df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(summary_df.index.tolist(), (summary_df["survival_rate"] * 100.0).to_numpy())
    plt.ylabel("Survival rate (%)")
    plt.title("Survival rate by bootstrap mode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_terminal_boxplot(outputs: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [outputs[k][:, -1] for k in outputs.keys()]
    plt.figure()
    plt.boxplot(data, labels=list(outputs.keys()))
    plt.ylabel("Terminal wealth")
    plt.title("Terminal wealth distribution by bootstrap mode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    dd = (peak - path) / np.clip(peak, 1e-12, None)
    return float(np.max(dd))


def save_drawdown_boxplot(outputs: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dd_data = []
    labels = []
    for k, paths in outputs.items():
        dd = np.array([_max_drawdown(paths[i, :]) for i in range(paths.shape[0])], dtype=float)
        dd_data.append(dd)
        labels.append(k)

    plt.figure()
    plt.boxplot(dd_data, labels=labels)
    plt.ylabel("Max drawdown (fraction)")
    plt.title("Max drawdown distribution by bootstrap mode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
