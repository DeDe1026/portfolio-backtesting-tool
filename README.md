# portfolio-backtesting-tool
# Monte Carlo Portfolio Simulator

## What
A Monte Carlo portfolio backtesting tool (inspired by PortfolioVisualizer) with:
- historical simple iid, block and regime bootstrapping through K-means ML
- terminal wealth distribution and withdrawal simulation
- Bayesian optimization for allocation, "alpha" and withdrawal parameters
- UI (Streamlit)

## Datas type
- ^GSPC (S&P 500)
- USDCHF=X (USD/CHF)
- ^SSMI (SMI)
- XAUUSD=X (Gold spot in USD)
- Swiss gov bonds ETF proxy (CHF): CSBGC0.SW (7-15 years)
- US gov bonds ETF proxy (USD): TLT (20y+) or BND (agg)

## Setup
```bash
conda env create -f environment.yml
conda activate portfolio-project
