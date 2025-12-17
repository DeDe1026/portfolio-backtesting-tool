

# Monte-Carlo simulator or when the namesake city is no longer about gambling.

**Category:** Simulation and Modeling

## Problem statement / motivation

Portfolio Visualizer has popularized free Monte-Carlo portfolio projections for U.S. retail investors. But Swiss investors face a different structural environment: different regulatory framework, different tax regime, different retirement pillars, different currency, and very different tax considerations on dividends vs. capital gains depending on each canton. There is currently no open source or student-level tool equivalent to Portfolio Visualizer for *Swiss* constraints. My motivation is twofold. First, this is genuinely useful for me personally as a Swiss investor. Second, this is exactly the kind of financial engineering + quantitative programming work I want to do professionally, either on the technical side of finance (quantitative research, PM support, portfolio construction) or in future entrepreneurial ventures.

## Planned approach and technologies

I will implement Monte-Carlo simulations using Python (NumPy, pandas) with user-defined distributions for returns, volatilities, correlations, and glidepaths. Optional layer: I will allow “Swiss specific” constraints such as Taxation, inflation, currency risk, pillar contributions, AVS retirement rents and foreign ETF domiciles. I will render distributions of future wealth, percentiles, drawdown distributions, survival rate and safe withdrawal probabilities.

## Expected challenges and how I’ll address them

Main challenge: access to a quality open source historical dataset that matches Swiss investor reality. Solution: allow modular import of external datasets + synthetic distribution fitting. Also: tax modeling complexity. Solution: start with simplified Swiss assumptions, then modularize tax functions.

## Success criteria

The simulator will be judged successful if a user can pick a portfolio configuration, generate ≥10,000 Monte-Carlo future wealth paths, and visualize percentile outcomes + failure probability under withdrawal rates.

## Stretch goals (if time permits)

1. Bootstrapping historical returns (block bootstrap).
2. Dynamic asset allocation rules (risk parity / volatility targeting).

