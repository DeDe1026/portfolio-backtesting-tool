from __future__ import annotations

import sys
from pathlib import Path

# --- Make sure project root is on PYTHONPATH so `import src...` works ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig
from src.evaluation import compute_survival_rate, summarize_terminal_wealth

# Optional: only if you want optimization from the UI
# from src.optimization import OptimizationConfig, optimize_portfolio

st.set_page_config(page_title="Swiss Portfolio Backtesting Tool", layout="wide")

def run_one_mode(sim, mode: str, n_paths: int, seed: int, alpha: float,
                 block_size: int, regime_k: int, regime_vol_window: int, regime_min_samples: int):
    paths = sim.simulate_paths(
        n_paths=n_paths,
        random_state=seed,
        bootstrap_mode=mode,
        block_size=block_size,
        alpha=alpha,
        regime_k=regime_k,
        regime_vol_window=regime_vol_window,
        regime_min_samples=regime_min_samples,
    )
    terminal = paths[:, -1]
    survival = float(np.mean(terminal > 0))
    return {
        "mode": mode,
        "survival_rate": survival,
        "median_terminal": float(np.median(terminal)),
        "p10_terminal": float(np.percentile(terminal, 10)),
        "p90_terminal": float(np.percentile(terminal, 90)),
        "paths": paths,  # keep for charts if you want
    }


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        # default: equal weights
        n = len(w)
        return {k: 1.0 / n for k in w} if n > 0 else {}
    return {k: float(v) / total for k, v in w.items()}


def main():
    st.title("Monte Carlo Portfolio Backtesting (Swiss-friendly)")

    project_root = Path(__file__).resolve().parents[1]
    default_data_path = project_root / "data" / "raw" / "monthly_returns_native.csv"

    with st.sidebar:
        st.header("Data")
        data_file = st.text_input("Path to monthly returns CSV", str(default_data_path))
        inflation_toggle = st.checkbox("Inflation-adjust withdrawals (real spending)", value=True)

     
        st.header("Bootstrap configuration")

        # iid has no extra params

        st.subheader("Block bootstrap")
        block_size = st.number_input("Block size (months)", min_value=2, max_value=60, value=12, step=1)

        st.subheader("Regime bootstrap")
        regime_k = st.number_input("Regime K (clusters)", min_value=2, max_value=10, value=3, step=1)
        regime_vol_window = st.number_input("Volatility window (months)", min_value=3, max_value=60, value=12, step=1)
        regime_min_samples = st.number_input("Min samples per regime", min_value=12, max_value=120, value=24, step=1)


        st.header("Simulation")
        n_paths = st.number_input("Monte Carlo paths", min_value=100, max_value=50000, value=5000, step=500)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    # -------- Load data ----------
    try:
        returns_df = load_returns_data(Path(data_file))
    except Exception as e:
        st.error(f"Could not load returns data: {e}")
        st.stop()

    # Identify investable assets: exclude inflation col if present
    inflation_col = "ch_inflation"
    assets = [c for c in returns_df.columns if c != inflation_col]

    if len(assets) == 0:
        st.error("No investable asset columns found in dataset.")
        st.stop()

    st.subheader("Dataset preview")
    st.write(f"Columns: {returns_df.columns.tolist()}")
    st.dataframe(returns_df.tail(10))

    # -------- User inputs ----------
    st.subheader("User Inputs")

    colA, colB, colC = st.columns(3)

    with colA:
        initial_capital = st.number_input(
            "Initial capital (CHF)",
            min_value=10_000.0,
            max_value=100_000_000.0,
            value=1_000_000.0,
            step=10_000.0,
            format="%.2f",
        )
        horizon_years = st.number_input(
            "Time horizon (years)",
            min_value=1,
            max_value=80,
            value=30,
            step=1,
        )

    with colB:
        w_pref = st.number_input(
            "Preferred monthly withdrawal (CHF)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=4000.0,
            step=100.0,
            format="%.2f",
        )
        w_floor = st.number_input(
            "Minimum monthly withdrawal floor (CHF)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=3000.0,
            step=100.0,
            format="%.2f",
        )

    with colC:
        st.markdown("**Alpha constraints**")
        if w_pref <= 0:
            alpha_max = 0.0
        else:
            alpha_max = max(0.0, 1.0 - (w_floor / w_pref))

        st.write(f"Max alpha allowed by floor: **{alpha_max:.2%}**")
        alpha = st.slider(
            "Alpha (withdrawal cut in negative months)",
            min_value=0.0,
            max_value=float(alpha_max),
            value=float(min(0.10, alpha_max)),
            step=0.01,
            help="In negative-return months, withdrawal is reduced by alpha, but cannot breach the floor implied by your inputs.",
        )

    if w_floor > w_pref:
        st.warning("Your floor is above your preferred withdrawal. Set floor ≤ preferred, or the model has no feasible alpha.")
        st.stop()

    # Convert CHF/month to withdrawal_rate used by your engine
    withdrawal_rate = (12.0 * w_pref) / initial_capital if initial_capital > 0 else 0.0

    st.markdown(
        f"- Implied annual withdrawal rate: **{withdrawal_rate:.2%}**  \n"
        f"- Monthly base withdrawal (before alpha): **CHF {w_pref:,.2f}**"
    )

    # -------- Portfolio weights UI ----------
    st.subheader("Initial Portfolio Weights")

    st.caption("Set raw weights (any scale). They will be automatically normalized to sum to 100%.")

    weight_cols = st.columns(min(4, len(assets)))
    raw_weights: dict[str, float] = {}

    for i, a in enumerate(assets):
        with weight_cols[i % len(weight_cols)]:
            raw_weights[a] = st.number_input(
                f"{a} weight",
                min_value=0.0,
                max_value=1_000.0,
                value=(1.0 / len(assets)) * 100.0,
                step=1.0,
                format="%.2f",
            )

    weights = normalize_weights(raw_weights)

    st.caption(f"Weight sum check: {sum(weights.values()):.6f} (should be 1.0)")


    w_df = pd.DataFrame(
        {"asset": list(weights.keys()), "weight": list(weights.values())}
    ).sort_values("weight", ascending=False)

    st.write("Normalized weights (sum = 100%):")
    st.dataframe(w_df.assign(weight_pct=lambda d: (100*d["weight"]).round(2)).drop(columns=["weight"]))

    # -------- Run Simulation ----------
    st.subheader("Run Simulation")

    run_btn = st.button("Run Monte Carlo", type="primary")

    if run_btn:
        cfg = PortfolioConfig(
            initial_capital=float(initial_capital),
            withdrawal_rate=float(withdrawal_rate),
            horizon_years=int(horizon_years),
            rebalance_frequency="yearly",
            inflation_aware_withdrawals=bool(inflation_toggle),
            inflation_col="ch_inflation",
        )

    sim = MonteCarloSimulator(
        returns=returns_df,
        asset_weights=weights,
        config=cfg,
        periods_per_year=12,
    )

    results = []
    for mode in ["iid", "block", "regime"]:
        res = run_one_mode(
            sim=sim,
            mode=mode,
            n_paths=int(n_paths),
            seed=int(seed),
            alpha=float(alpha),
            block_size=int(block_size),
            regime_k=int(regime_k),
            regime_vol_window=int(regime_vol_window),
            regime_min_samples=int(regime_min_samples),
        )
        results.append(res)

    # Build comparison table
    comp = pd.DataFrame([{
        "Bootstrap": r["mode"],
        "Survival %": 100 * r["survival_rate"],
        "Median terminal (CHF)": r["median_terminal"],
        "P10 terminal (CHF)": r["p10_terminal"],
        "P90 terminal (CHF)": r["p90_terminal"],
    } for r in results])

    st.subheader("Bootstrap comparison")
    st.dataframe(comp.style.format({
        "Survival %": "{:.1f}",
        "Median terminal (CHF)": "{:,.0f}",
        "P10 terminal (CHF)": "{:,.0f}",
        "P90 terminal (CHF)": "{:,.0f}",
    }))

    # Metrics row (optional)
    c1, c2, c3 = st.columns(3)
    best_survival = comp.sort_values("Survival %", ascending=False).iloc[0]
    c1.metric("Best survival", f'{best_survival["Bootstrap"]} ({best_survival["Survival %"]:.1f}%)')

    best_median = comp.sort_values("Median terminal (CHF)", ascending=False).iloc[0]
    c2.metric("Best median terminal", f'{best_median["Bootstrap"]} (CHF {best_median["Median terminal (CHF)"]:,.0f})')

    worst_p10 = comp.sort_values("P10 terminal (CHF)", ascending=True).iloc[0]
    c3.metric("Worst downside (P10)", f'{worst_p10["Bootstrap"]} (CHF {worst_p10["P10 terminal (CHF)"]:,.0f})')

    # Charts: show sample paths for each mode
    st.subheader("Sample paths by bootstrap mode (first 50)")
    for r in results:
        st.markdown(f"**{r['mode'].upper()}**")
        st.line_chart(r["paths"][:50, :].T)



if __name__ == "__main__":
    main()
