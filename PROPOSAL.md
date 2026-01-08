

# Monte-Carlo simulator or when the namesake city is no longer about gambling.

**Category:** Simulation and Modeling

## Problem statement / motivation

Portfolio Visualizer has popularized free Monte-Carlo portfolio projections for U.S. retail investors. But Swiss investors face a different structural environment: different regulatory framework, different tax regime, different retirement pillars, different currency, different inflation rates and very different tax considerations on dividends vs. wealth tax depending on each canton. There is currently no open source or student-level tool equivalent to Portfolio Visualizer for *Swiss* constraints. My motivation is twofold. First, this is genuinely useful for me personally as a Swiss investor. Second, this is exactly the kind of financial engineering + quantitative programming work I want to do professionally, either on the technical side of finance (quantitative research, PM support, portfolio construction) or in future entrepreneurial ventures.

## Planned approach and technologies

I will implement Monte-Carlo simulations using Python (NumPy, pandas) with Bayesian optimization using Optuna to find optimal asset allocation and "alpha" withdrawal reduction factor, to maximize survival and withdrawal rates. I will implement 3 bootstrapping methods for the Monte-Carlo simulation, simple iid, block bootstrapping and regime bootstrapping using K-means machine learning to classify asset returns into 3 different regimes with scikit-learn, scipy. Monthly returns will be computed after fetching daily returns from yfinance for the S&P500, SMI, Gold in USD, Swiss 7-15y government bonds and US Government 20+y bonds. All assets in USD will be converted to CHF month after month using FX USDCHF historical returns, that will be sampled during the simulations, following the same bootstrapping methods. Withdrawals will be inflation-adjusted, using historical swiss inflation, sampled following the same bootstrapping methods. 3 optimization modes will be proposed, optimizing survival rate with an enforced withdrawal floor, optimizing withdrawal rate given a survival rate and optimizing survival rate through alpha and asset allocation. To provide a user interface for easy portfolio configuration, I will implement streamlit. To ensure consistency between running main.py and the streamlit file, I will develop a pipeline python file for both files to call to ensure they use the same logic.

## Expected challenges and how I’ll address them

Main challenge: access and implementation of quality open source historical datasets that match Swiss investor reality, more specifically inflation rates. Solution: If not available on BFS/Statistics, allow modular import of external datasets + synthetic distribution fitting. 

## Success criteria

The simulator will be judged successful if a user can pick a portfolio configuration, have machine learning optimize it, generate ≥1,000 Monte-Carlo future wealth paths, and visualize percentile outcomes, terminal wealth, performance metrics + survival probability under different bootstrapping methods and withdrawal rates.

## Stretch goals (if time permits)

1. GMM regime bootstrapping and comparison with K-means.
2. Dynamic asset allocation rules (risk parity / volatility targeting).
3. Integrating dividend returns + dividend and wealth tax considerations
4. Implementing 1st and 2nd retirement pillars monthly pensions, cash buffers and potential other sources of income in the mix of retirement simulation

