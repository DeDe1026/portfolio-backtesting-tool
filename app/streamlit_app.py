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
import matplotlib.pyplot as plt

from src.data_loader import load_returns_data
from src.models import MonteCarloSimulator, PortfolioConfig
from src.optimization import OptimizationConfig, optimize_portfolio


st.set_page_config(page_title="Swiss Portfolio Backtesting Tool", layout="wide")


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw} if n > 0 else {}
    return {k: float(v) / total for k, v in raw.items()}


def run_one_mode(
    sim: MonteCarloSimulator,
    mode: str,
    n_paths: int,
    seed: int,
    alpha: float,
    block_size: int,
    regime_k: int,
    regime_vol_window: int,
    regime_min_samples: int,
) -> dict:
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
    survival = float(np.mean(terminal > 0.0))
    return {
        "mode": mode,
        "paths": paths,
        "terminal": terminal,
        "survival": survival,
        "median_terminal": float(np.median(terminal)),
        "p10_terminal": float(np.percentile(terminal, 10)),
        "p90_terminal": float(np.percentile(terminal, 90)),
    }


def median_path_cagr_and_vol(paths: np.ndarray, periods_per_year: int = 12) -> tuple[float, float]:
    """
    Pick the path whose terminal wealth is closest to the median terminal wealth.
    Compute:
      - CAGR of wealth series (net of withdrawals, because withdrawals already applied)
      - annualized volatility of monthly wealth returns
    """
    terminal = paths[:, -1]
    med = np.median(terminal)
    i = int(np.argmin(np.abs(terminal - med)))
    w = paths[i, :]
    rets = w[1:] / np.maximum(w[:-1], 1e-12) - 1.0

    years = (len(w) - 1) / periods_per_year
    if w[-1] <= 0 or w[0] <= 0:
        cagr = -1.0
    else:
        cagr = float((w[-1] / w[0]) ** (1.0 / max(years, 1e-12)) - 1.0)

    vol = float(np.std(rets, ddof=1) * np.sqrt(periods_per_year)) if len(rets) > 2 else float("nan")
    return cagr, vol


def pie_chart(weights: dict[str, float], title: str):
    fig, ax = plt.subplots()
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
    ax.set_title(title)
    st.pyplot(fig)


def main():
    st.title("Swiss Portfolio Backtesting Tool (Monte Carlo + Bootstrap + Bayesian Optimization)")

    project_root = Path(__file__).resolve().parents[1]
    default_data_path = project_root / "data" / "raw" / "monthly_returns_native.csv"

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Data")
        data_file = st.text_input("Path to monthly returns CSV", str(default_data_path))

        st.header("Inflation")
        inflation_toggle = st.checkbox("Inflation-adjust withdrawals (real spending)", value=True)
        inflation_col = st.text_input("Inflation column name in dataset", value="ch_inflation")

        st.header("Bootstrap configuration")
        st.subheader("Block bootstrap")
        block_size = st.number_input("Block size (months)", min_value=2, max_value=60, value=12, step=1)

        st.subheader("Regime bootstrap")
        regime_k = st.number_input("Regime K (clusters)", min_value=2, max_value=10, value=3, step=1)
        regime_vol_window = st.number_input("Volatility window (months)", min_value=3, max_value=60, value=12, step=1)
        regime_min_samples = st.number_input("Min samples per regime", min_value=12, max_value=240, value=24, step=1)

        st.header("Simulation")
        n_paths = st.number_input("Monte Carlo paths", min_value=200, max_value=50000, value=5000, step=500)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

        st.header("Optimization")
        opt_mode_ui = st.selectbox(
            "Optimization logic",
            [
                "A) Preferred + Floor → optimize weights to maximize survival",
                "B) Preferred only + target survival → optimize weights + alpha to maximize withdrawal",
                "C) Preferred only → optimize weights + alpha to maximize survival",
            ],
        )

        target_survival = st.slider("Target survival (used in Mode B)", 0.50, 0.99, 0.95, 0.01)
        n_trials = st.number_input("Optuna trials", min_value=10, max_value=400, value=80, step=10)

        st.caption("Alpha search cap for modes that optimize alpha (B/C).")
        alpha_cap = st.slider("Alpha max (optimization cap)", 0.0, 0.80, 0.50, 0.05)

        st.caption("Withdrawal multiplier range for Mode B (>=1 means trying to raise withdrawals).")
        withdraw_mult_max = st.slider("Max withdrawal multiplier (Mode B)", 1.0, 3.0, 2.0, 0.1)

    # ---------------- Load data ----------------
    try:
        returns_df = load_returns_data(Path(data_file))
    except Exception as e:
        st.error(f"Could not load returns data: {e}")
        st.stop()

    # assets exclude inflation
    assets = [c for c in returns_df.columns if c != inflation_col]
    if len(assets) == 0:
        st.error("No investable assets found (all columns are treated as inflation or dataset is empty).")
        st.stop()

    st.subheader("Dataset preview")
    st.write(f"Columns: {returns_df.columns.tolist()}")
    st.dataframe(returns_df.tail(12))

    # ---------------- User inputs ----------------
    st.subheader("User Inputs (CHF)")

    colA, colB, colC = st.columns(3)

    with colA:
        initial_capital = st.number_input(
            "Initial capital (CHF)",
            min_value=10_000.0,
            max_value=200_000_000.0,
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
            help="Used mainly in Mode A to infer the max feasible alpha.",
        )

    with colC:
        st.markdown("**Initial weights (user guess)**")
        st.caption("You set raw weights; they will be normalized to 100% before optimization/simulation.")

    if w_floor > w_pref and w_pref > 0:
        st.warning("Floor is above preferred withdrawal. Set floor ≤ preferred.")
        st.stop()

    # implied withdrawal rate based on preferred CHF/month
    withdrawal_rate_pref = (12.0 * w_pref) / initial_capital if initial_capital > 0 else 0.0

    st.markdown(
        f"- Implied annual withdrawal rate from preferred amount: **{withdrawal_rate_pref:.2%}**  \n"
        f"- Preferred monthly withdrawal: **CHF {w_pref:,.2f}**"
    )

    # ---------------- Weight inputs ----------------
    st.subheader("Initial Portfolio Weights (user input)")

    weight_cols = st.columns(min(4, len(assets)))
    raw_weights: dict[str, float] = {}

    for i, a in enumerate(assets):
        with weight_cols[i % len(weight_cols)]:
            raw_weights[a] = st.number_input(
                f"{a} (raw)",
                min_value=0.0,
                max_value=1000.0,
                value=100.0 / len(assets),
                step=1.0,
                format="%.2f",
            )

    weights_guess = normalize_weights(raw_weights)

    w_df = pd.DataFrame({"asset": list(weights_guess.keys()), "weight": list(weights_guess.values())})
    w_df["weight_pct"] = (100.0 * w_df["weight"]).round(2)
    w_df = w_df.sort_values("weight_pct", ascending=False)

    st.write("Normalized weights (user input):")
    st.dataframe(w_df[["asset", "weight_pct"]])

    # ---------------- Run button ----------------
    st.subheader("Run Optimization + Compare Bootstrap Modes")

    run_btn = st.button("Run", type="primary")

    if not run_btn:
        st.stop()

    # ---------------- Determine optimization mode ----------------
    if opt_mode_ui.startswith("A)"):
        opt_mode = "A_survival_weights_only"
    elif opt_mode_ui.startswith("B)"):
        opt_mode = "B_withdraw_max_subject_survival"
    else:
        opt_mode = "C_survival_weights_alpha"

    # For Mode A: alpha is constrained by floor vs preferred
    if w_pref <= 0:
        alpha_max_floor = 0.0
    else:
        alpha_max_floor = max(0.0, 1.0 - (w_floor / w_pref))

    # In Mode A we "fix" alpha to the max allowed by floor (your earlier logic)
    fixed_alpha = float(alpha_max_floor) if opt_mode == "A_survival_weights_only" else None

    # base config uses preferred withdrawal as baseline
    base_cfg = PortfolioConfig(
        initial_capital=float(initial_capital),
        withdrawal_rate=float(withdrawal_rate_pref),
        horizon_years=int(horizon_years),
        rebalance_frequency="yearly",
        inflation_aware_withdrawals=bool(inflation_toggle),
        inflation_col=str(inflation_col),
    )

    # optimization config
    opt_cfg = OptimizationConfig(
        n_trials=int(n_trials),
        seed=int(seed),
        target_survival=float(target_survival),
        n_paths_eval=min(int(n_paths), 5000),  # keep it reasonable in UI
        mode=opt_mode,
        alpha_min=0.0,
        alpha_max=float(alpha_max_floor if opt_mode == "A_survival_weights_only" else alpha_cap),
        fixed_alpha=fixed_alpha,
        bootstrap_mode="iid",       # optimize on IID for speed; compare later with all 3
        block_size=int(block_size),
        withdraw_mult_min=1.0,
        withdraw_mult_max=float(withdraw_mult_max),
    )

    with st.spinner("Running Bayesian optimization (Optuna/TPE)..."):
        opt_result = optimize_portfolio(
            returns_df=returns_df,
            assets=assets,
            base_config=base_cfg,
            opt_config=opt_cfg,
            periods_per_year=12,
        )

    best_weights = opt_result["best_weights"]
    best_alpha = float(opt_result["best_alpha"])
    best_withdrawal_rate = float(opt_result["best_withdrawal_rate"])

    # Convert best withdrawal rate to CHF/month (for readability)
    best_monthly_chf = (best_withdrawal_rate * initial_capital) / 12.0

    st.success("Optimization complete.")

    st.subheader("Optimized portfolio (from Bayesian optimization)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", opt_result["mode"])
    c2.metric("Optimized alpha", f"{best_alpha:.1%}")
    c3.metric("Optimized annual withdrawal rate", f"{best_withdrawal_rate:.2%}")
    c4.metric("Implied monthly withdrawal", f"CHF {best_monthly_chf:,.0f}")

    pie_chart(best_weights, "Optimized weights (pie chart)")

    # ---------------- Compare all 3 bootstrap modes with optimized params ----------------
    cfg_best = PortfolioConfig(
        initial_capital=float(initial_capital),
        withdrawal_rate=float(best_withdrawal_rate),
        horizon_years=int(horizon_years),
        rebalance_frequency="yearly",
        inflation_aware_withdrawals=bool(inflation_toggle),
        inflation_col=str(inflation_col),
    )

    sim_best = MonteCarloSimulator(
        returns=returns_df,
        asset_weights=best_weights,
        config=cfg_best,
        periods_per_year=12,
    )

    with st.spinner("Running simulations (IID, Block, Regime) for comparison..."):
        results = []
        for mode in ["iid", "block", "regime"]:
            results.append(
                run_one_mode(
                    sim=sim_best,
                    mode=mode,
                    n_paths=int(n_paths),
                    seed=int(seed),
                    alpha=float(best_alpha),
                    block_size=int(block_size),
                    regime_k=int(regime_k),
                    regime_vol_window=int(regime_vol_window),
                    regime_min_samples=int(regime_min_samples),
                )
            )

    comp = pd.DataFrame([{
        "Bootstrap": r["mode"].upper(),
        "Survival": r["survival"],
        "Median terminal (CHF)": r["median_terminal"],
        "P10 terminal (CHF)": r["p10_terminal"],
        "P90 terminal (CHF)": r["p90_terminal"],
    } for r in results])

    st.subheader("Bootstrap comparison (optimized params applied)")
    st.dataframe(comp.style.format({
        "Survival": "{:.1%}",
        "Median terminal (CHF)": "CHF {:,.0f}",
        "P10 terminal (CHF)": "CHF {:,.0f}",
        "P90 terminal (CHF)": "CHF {:,.0f}",
    }))

    # ---------------- Median-path return/vol ----------------
    st.subheader("Median-path performance (net of withdrawals)")

    perf_rows = []
    for r in results:
        cagr, vol = median_path_cagr_and_vol(r["paths"], periods_per_year=12)
        perf_rows.append({
            "Bootstrap": r["mode"].upper(),
            "Median-path CAGR": cagr,
            "Median-path Vol (ann.)": vol,
        })

    perf = pd.DataFrame(perf_rows)
    st.dataframe(perf.style.format({
        "Median-path CAGR": "{:.2%}",
        "Median-path Vol (ann.)": "{:.2%}",
    }))

    # ---------------- Plots: X axis in years ----------------
    st.subheader("Sample paths (first 50) with X-axis in years")

    for r in results:
        st.markdown(f"**{r['mode'].upper()}**")
        paths = r["paths"][:50, :]
        x_years = np.arange(paths.shape[1]) / 12.0
        df_plot = pd.DataFrame(paths.T, index=x_years)
        df_plot.index.name = "Years"
        st.line_chart(df_plot)

    # ---------------- Helpful note about "average yearly withdrawal" ----------------
    st.info(
        "To display the *realized average yearly withdrawal* (and an optimized alpha that maximizes realized withdrawals), "
        "the simulator must return the withdrawal stream per path (e.g., simulate_paths(..., return_withdrawals=True)). "
        "Right now this UI uses the withdrawal rate/alpha parameters but cannot compute realized withdrawals unless the "
        "engine exposes them."
    )


if __name__ == "__main__":
    main()

