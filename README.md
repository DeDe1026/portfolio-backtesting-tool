# Swiss Portfolio Backtesting Tool
# Monte Carlo Portfolio Simulator

## Research Question
How does sustainability, for a multi-asset, multi-currency portfolio under fixed, inflation-adjusted withdrawals vary across different historical bootstrapping assumptions, and how can portfolio allocations be optimized to maximize survival and/or spending?

## What
A Monte Carlo portfolio backtesting tool (inspired by PortfolioVisualizer) with:
- daily returns fetched from yfinance library and computed as monthly returns
- historical simple iid, block and regime bootstrapping through K-means ML
- FX-adjusted US assets to get CHF returns 
- Inflation-adjusted withdrawals with preferred and floor withdrawals logic
- Bayesian optimization through Optuna for asset allocation with yearly rebalancing, "alpha" withdrawal reduction factor when monthly returns are negative, and withdrawal parameters
- 3 different optimization modes/logic, depending on the investor's goal
- Streamlit interactive UI
- Reproducible CLI Pipeline with CSV/PNG of terminal wealth percentiles and portfolio performance metrics outputs

## Bootstrapping Methods

| Mode     | Description                                              |
| -------- | --------------------------------------------------       |
| `iid`    | Independent monthly resampling                           |
| `block`  | Block bootstrapping (serial correlation preserved)       |
| `regime` | K-means volatility regimes + Markov transitions matrix   |

## Withdrawal Logic
Preferred + Floor Rule (True Floor)

Used in:

- BASIC simulation

- Optimization Mode A

Logic:

- Positive month → preferred withdrawal

- Negative month → floor withdrawal

- Floor ≤ preferred enforced

- Withdrawals are fixed CHF/month (not percentage of wealth)

This is not a percentage floor — it is a true spending floor

## Optimization Modes (Optuna)
- Mode A — Survival First (Floor-Aware)

    - Optimize weights only

    - Uses preferred + floor withdrawal rule

    - Objective: maximize survival probability

    - Alpha is not optimized

- Mode B — Max Withdrawal Subject to Survival

    - Optimize weights + alpha + withdrawal multiplier

    - Constraint: survival ≥ target

    - Objective: maximize withdrawal rate

- Mode C — Survival with Alpha

    - Optimize weights + alpha

    - Withdrawal fixed

    - Objective: maximize survival

## Datas type (price returns, no dividends)
- ^GSPC (S&P 500)
- USDCHF=X (USD/CHF)
- ^SSMI (SMI)
- GC=F (Gold spot in USD)
- Swiss gov bonds ETF proxy (CHF): CSBGC0.SW (7-15 years)
- US gov bonds ETF proxy (USD): TLT (20y+) 

## Setup
conda env create -f environment.yml
conda activate portfolio-project

## Usage
python main.py 
streamlit run app/streamlit_app.py

Expected output: 
- basic vs optimized bootstrap survival comparison CSV file
- median-path CAGR/ Vol/ Max Drawdown metrics CSV file
- Full Optuna trial history CSV file
- Asset allocation Pie chart PNG file 
- Terminal wealth distribution for different percentiles and for each bootstrapping method for both basic and optimized simulations, PNG files
- Best optimization results JSON file
- streamlit browser page launched when Streamlit is run

```
## Project structure
portfolio-backtesting-tool/
│
├── app/
│   └── streamlit_app.py          # Interactive UI
│
├── data/
│   |── raw_cache/                # daily asset datas
│   └── raw/
│       └── monthly_returns_native.csv
│       └── switzerland_inflation_monthly_clean.csv
│       └── switzerland_inflation_monthly.csv
│
├── results/
│   ├── comp_basic_vs_optimized.csv # clean and raw
│   ├── perf_basic_vs_optimized.csv # clean and raw
│   ├── optuna_trials.csv
│   ├── best_params.json
│   └── *.png                       # pie chart and wealth paths distribution
│
├── scripts/
│   └── clean_fso_inflation_csv.py  # Inflation CSV loading and cleaning
│
├── src/
│   ├── models.py                 # Monte Carlo engine (core logic)
│   ├── pipeline.py               # BASIC + OPTIMIZED orchestration
│   ├── optimization.py           # Optuna optimization logic
│   ├── compare_plots.py          # All plotting logic (CLI outputs)
│   ├── inflation.py              # Inflation computation logic 
│   ├── data_fetcher.py           # Data fetching from yfinance
│   ├── data_loader.py            # CSV loading + validation
│   ├── regime.py                 # Regime bootstrapping
│   └── build_dataset.py          # Monthly returns data builder
│
├── .gitignore                    # Github tracking limits framework
├── AI_Usage.md               
├── environemnt.yml               # Dependencies
├── main.py                       # CLI entry point
├── Proposal.md                   
└── README.md
```
## Results
Depend on inputs values (see Technical report charts, plots and tables for results of a specific scenario)

## Requirements
- Python 3.11
- Numpy, Pandas, Matplotlib, Scikit-learn, Scipy, Jupyter, Optuna, Pip, Streamlit, Yfinance, Pandas-datareader