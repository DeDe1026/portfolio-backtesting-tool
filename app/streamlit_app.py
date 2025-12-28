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
import altair as alt


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

def apply_fx_conversion_to_chf(
    returns_df: pd.DataFrame,
    fx_col: str = "usdchf",
    usd_asset_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert USD-denominated asset returns into CHF returns using USDCHF monthly returns:
        r_chf = (1+r_usd_asset)*(1+r_fx) - 1
    Assumes fx_col is monthly % return of USDCHF (CHF per USD).

    Returns a COPY with USD assets converted. fx_col is kept (non-investable).
    """
    df = returns_df.copy()

    if fx_col not in df.columns:
        return df

    fx = df[fx_col].astype(float)

    if usd_asset_cols is None:
        # sensible defaults based on your column names
        usd_asset_cols = [c for c in df.columns if c.endswith("_usd")]

    for c in usd_asset_cols:
        if c not in df.columns:
            continue
        df[c] = (1.0 + df[c].astype(float)) * (1.0 + fx) - 1.0

    return df

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
    out = sim.simulate_paths(
        n_paths=n_paths,
        random_state=seed,
        bootstrap_mode=mode,
        block_size=block_size,
        alpha=alpha,
        regime_k=regime_k,
        regime_vol_window=regime_vol_window,
        regime_min_samples=regime_min_samples,
        return_diagnostics=True,
    )
    paths, diag = out  # <-- because return_diagnostics=True

    terminal = paths[:, -1]
    survival = float(np.mean(terminal > 0.0))
    failed = int(np.sum(terminal <= 0.0))

    neg_months = diag["neg_months"]          # shape (n_paths, n_periods)
    withdrawals = diag["withdrawals"]        # shape (n_paths, n_periods)

    neg_count_per_path = neg_months.sum(axis=1)
    total_months = neg_months.shape[1]

    return {
        "mode": mode,
        "paths": paths,
        "terminal": terminal,
        "survival": survival,
        "failed": failed,
        "n_paths": int(n_paths),
        "neg_months_total": int(neg_months.sum()),
        "neg_months_per_path_median": float(np.median(neg_count_per_path)),
        "Neg_months_per_path_mean": float(np.mean(neg_count_per_path)),
        "neg_months_total_months": int(total_months),
        "withdrawals": withdrawals,
        "neg_months": neg_months,
        "median_terminal": float(np.median(terminal)),
        "p10_terminal": float(np.percentile(terminal, 10)),
        "p90_terminal": float(np.percentile(terminal, 90)),
    }

def pick_paths_by_terminal_percentiles(paths: np.ndarray, percentiles: list[int]) -> np.ndarray:
    terminal = paths[:, -1]
    order = np.argsort(terminal)
    n = len(order)
    picked = []
    for p in percentiles:
        k = int(round((p / 100.0) * (n - 1)))
        picked.append(paths[order[k], :])
    return np.vstack(picked)

def paths_df_year_index(paths: np.ndarray, periods_per_year: int = 12) -> pd.DataFrame:
    years = np.arange(paths.shape[1]) / periods_per_year
    df = pd.DataFrame(paths.T, index=years)
    df.index.name = "Years"
    return df

def median_path_cagr_vol_mdd(paths: np.ndarray, periods_per_year: int = 12) -> tuple[float, float, float]:
    """
    Pick the path whose terminal wealth is closest to the median terminal wealth.
    Compute:
      - CAGR of wealth series (net of withdrawals, because withdrawals already applied)
      - annualized volatility of monthly wealth returns
      - max drawdown of wealth series
    0..1 as fraction.
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
    mdd = max_drawdown(w)
    return cagr, vol, mdd

def altair_lines_from_wide(df: pd.DataFrame, title: str, y_title: str = "CHF"):
    """
    df: index is Years; columns are series names; values are CHF wealth.
    Produces an Altair line chart with formatted tooltips.
    """
    plot_df = df.reset_index().melt(id_vars=[df.index.name or "index"], var_name="Series", value_name="Value")
    x_col = df.index.name or "index"
    plot_df = plot_df.rename(columns={x_col: "Years"})

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("Years:Q", title="Years"),
            y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=",.0f")),
            color=alt.Color("Series:N"),
            tooltip=[
                alt.Tooltip("Years:Q", format=",.2f"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Value:Q", format=",.2f"),
            ],
        )
        .properties(title=title, height=420)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

def pie_chart(weights: dict[str, float], title: str):
    fig, ax = plt.subplots()
    ax.pie(list(weights.values()), labels=list(weights.keys()), autopct="%1.1f%%")
    ax.set_title(title)
    st.pyplot(fig)

def build_comp_rows(results_list, scenario_label: str):
    rows = []
    for r in results_list:
        neg = r["neg_months"]  # (n_paths, n_periods)
        neg_per_path = neg.sum(axis=1)

        rows.append({
            "Scenario": scenario_label,
            "Bootstrap": r["mode"].upper(),
            "Survival": float(r["survival"]),
            "Failed (count)": float(r["failed"]),
            "Failed (%)": float(r["failed"] / r["n_paths"]),
            "Neg months / path (median)": float(np.median(neg_per_path)),
            "Neg months / path (mean)": float(np.mean(neg_per_path)),
            "Total months": float(neg.shape[1]),
            "Median terminal (CHF)": float(r["median_terminal"]),
            "P10 terminal (CHF)": float(r["p10_terminal"]),
            "P90 terminal (CHF)": float(r["p90_terminal"]),
        })
    return rows

def max_drawdown(series: np.ndarray) -> float:
    """
    Max drawdown of a wealth series (0..1 as fraction).
    """
    x = np.asarray(series, dtype=float)
    x = np.maximum(x, 1e-12)
    peak = np.maximum.accumulate(x)
    dd = 1.0 - (x / peak)
    return float(np.max(dd))

def median_path_series(paths: np.ndarray) -> np.ndarray:
    return np.median(paths, axis=0)

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

    fx_col = "usdchf"

    # Let user choose which columns are USD-denominated
    default_usd_assets = [c for c in returns_df.columns if c.endswith("_usd")]
    usd_assets = st.sidebar.multiselect(
        "USD-denominated assets (will be converted to CHF using usdchf)",
        options=[c for c in returns_df.columns if c not in [fx_col, "ch_inflation"]],
        default=default_usd_assets,
    )

    # Apply conversion
    returns_df_fx = apply_fx_conversion_to_chf(returns_df, fx_col=fx_col, usd_asset_cols=usd_assets)

    # Define investable assets: exclude inflation + fx (usdchf)
    assets = [c for c in returns_df_fx.columns if c not in [inflation_col, fx_col]]

    if len(assets) == 0:
        st.error("No investable assets found (all columns are treated as inflation or dataset is empty).")
        st.stop()

    st.write(f"Columns (fx-adjusted): {returns_df_fx.columns.tolist()}")
    st.dataframe(returns_df_fx.tail(12))

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
            returns_df=returns_df_fx,
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
        returns=returns_df_fx,
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

    # ---------------- BASIC (no optimization) simulation ----------------
    alpha_basic = 0.0
    if w_pref > 0:
        alpha_basic = max(0.0, min(0.999, 1.0 - (w_floor / w_pref)))

    cfg_basic = PortfolioConfig(
        initial_capital=float(initial_capital),
        withdrawal_rate=float(withdrawal_rate_pref),
        horizon_years=int(horizon_years),
        rebalance_frequency="yearly",
        inflation_aware_withdrawals=bool(inflation_toggle),
        inflation_col=str(inflation_col),
    )

    sim_basic = MonteCarloSimulator(
        returns=returns_df_fx,
        asset_weights=weights_guess,
        config=cfg_basic,
        periods_per_year=12,
    )

    with st.spinner("Running BASIC simulations (IID, Block, Regime) for comparison..."):
        basic_results = []
        for mode in ["iid", "block", "regime"]:
            basic_results.append(
                run_one_mode(
                    sim=sim_basic,
                    mode=mode,
                    n_paths=int(n_paths),
                    seed=int(seed),
                    alpha=float(alpha_basic),
                    block_size=int(block_size),
                    regime_k=int(regime_k),
                    regime_vol_window=int(regime_vol_window),
                    regime_min_samples=int(regime_min_samples),
                )
            )

    st.subheader("Scenario definitions")

    st.markdown(
        f"""
    **BASIC:** user weights, preferred withdrawal CHF {w_pref:,.2f} / month, and in negative months withdrawal becomes floor CHF {w_floor:,.2f}  
    → implemented as **alpha_basic = {alpha_basic:.2%}**

    **OPTIMIZED:** Bayesian optimization chooses weights + alpha + withdrawal rate (per selected optimization mode).
    """
    )

    # ---------------- Bootstrap comparison table ----------------
    comp = pd.DataFrame(
        build_comp_rows(basic_results, "BASIC (user weights, floor rule)") +
        build_comp_rows(results, "OPTIMIZED (Bayesian)")
    )

    st.subheader("Bootstrap comparison (BASIC vs OPTIMIZED)")
    st.dataframe(
        comp.style.format({
            "Survival": "{:.2%}",
            "Failed (%)": "{:.2%}",
            "Failed (count)": "{:,.2f}",
            "Neg months / path (median)": "{:,.2f}",
            "Neg months / path (mean)": "{:,.2f}",
            "Total months": "{:,.2f}",
            "Median terminal (CHF)": "CHF {:,.2f}",
            "P10 terminal (CHF)": "CHF {:,.2f}",
            "P90 terminal (CHF)": "CHF {:,.2f}",
        })
    )


    # ---------------- Median-path return/vol/max drawdown ----------------
    st.subheader("Median-path performance (net of withdrawals)")

    perf_rows = []
    for scenario_label, res_list in [
        ("BASIC (user)", basic_results),
        ("OPTIMIZED", results),
    ]:
        for r in res_list:
            cagr, vol, mdd = median_path_cagr_vol_mdd(r["paths"], periods_per_year=12)
            perf_rows.append({
                "Scenario": scenario_label,
                "Bootstrap": r["mode"].upper(),
                "Median-path CAGR": float(cagr),
                "Median-path Vol (ann.)": float(vol),
                "Median-path Max Drawdown": float(mdd),
            })


    perf = pd.DataFrame(perf_rows)

    st.dataframe(
        perf.style.format({
            "Median-path CAGR": "{:.2%}",
            "Median-path Vol (ann.)": "{:.2%}",
            "Median-path Max Drawdown": "{:.2%}",
        })
    )

    # ---------------- Plots: X axis in years ----------------
    st.subheader("Representative paths (11 percentiles) — BASIC and OPTIMIZED")

    percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    for scenario_label, res_list in [
        ("BASIC", basic_results),
        ("OPTIMIZED", results),
    ]:
        st.markdown(f"### {scenario_label}")
        for r in res_list:
            st.markdown(f"**{r['mode'].upper()}**")
            rep = pick_paths_by_terminal_percentiles(r["paths"], percentiles)
            df_plot = paths_df_year_index(rep, periods_per_year=12)
            df_plot.columns = [f"P{p}" for p in percentiles]

            # Use Altair so tooltips are nicely formatted
            altair_lines_from_wide(df_plot, title=f"{scenario_label} — {r['mode'].upper()} representative paths")

    st.subheader("Median path comparison of all bootstrap modes for OPTIMIZED")

    years = np.arange(results[0]["paths"].shape[1]) / 12.0
    median_df = pd.DataFrame(index=years)
    median_df.index.name = "Years"


    # OPTIMIZED
    for r in results:
        median_df[f"OPT_{r['mode'].upper()}"] = median_path_series(r["paths"])

    altair_lines_from_wide(median_df, "Median path comparison of all bootstrap modes for OPTIMIZED")



    # ---------------- Helpful note about "average yearly withdrawal" ----------------
    st.info(
        "To display the *realized average yearly withdrawal* (and an optimized alpha that maximizes realized withdrawals), "
        "the simulator must return the withdrawal stream per path (e.g., simulate_paths(..., return_withdrawals=True)). "
        "Right now this UI uses the withdrawal rate/alpha parameters but cannot compute realized withdrawals unless the "
        "engine exposes them."
    )


if __name__ == "__main__":
    main()

