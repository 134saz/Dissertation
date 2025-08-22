This repository contains the implementation of portfolio optimisation models used in the dissertation "Optimising Investment Portfolios with Risk Constraints".

This study compares four portfolio optimisation approaches across varying market conditions using FTSE 100 data from 2018-2025:
Mean-Variance Optimisation - Traditional Markowitz model
Static CVaR-Constrained - Linear programming with CVaR constraints
Two-Stage Stochastic Programming - Incorporating uncertainty and transaction costs
Multistage Stochastic Programming - Full dynamic optimisation with GARCH scenario generation

Key Findings:
Regime-dependent performance: No single model dominates across all market conditions
Multistage model: Worst performer during stability (-2.44%) but best during crisis (+0.12%)
Two-Stage model: Most consistent performance with reasonable computational cost (12.4x Mean-Variance)
All optimisation models meaningfully outperformed naive equal-weight diversification
