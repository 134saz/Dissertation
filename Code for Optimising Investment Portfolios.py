# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 18:58:28 2025
Portfolio Optimisation Models: Regime-Dependent Performance Analysis

Implementation of four portfolio optimisation models:
1. Mean-Variance Optimisation
2. Static CVaR-Constrained Linear Programming  
3. Two-Stage Stochastic Programming with CVaR
4. Multistage Stochastic Programming with GARCH scenarios

Author: Sara Drozd
Institution: University of Edinburgh 
"""

import pandas as pd
import numpy as np
from arch import arch_model
from gurobipy import Model, GRB
import time
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# === LOADING DATA AND GENERATING WINDOWS ===

#---Load Data------------------------------------------------------------------
def load_log_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], dayfirst=True, index_col=0)
    df = df.sort_index()
    return df

#---Windows--------------------------------------------------------------------
def load_risk_windows(path: str) -> pd.DataFrame:
    """Load and parse the summary risk windows data"""
    df = pd.read_csv(path)
    return df

def get_rolling_windows(data: pd.DataFrame, train_years=3, test_years=1):
    windows = []
    # Align start date to June 1 of the first year in data
    start_year = data.index.min().year
    start_date = pd.Timestamp(year=start_year, month=6, day=1)

    # The latest possible train start date so that test_end doesn't go beyond data max
    max_start_year = data.index.max().year - train_years - test_years + 1
    yearly_start_dates = pd.date_range(start=start_date, end=pd.Timestamp(year=max_start_year, month=6, day=1), freq='12MS')

    for train_start in yearly_start_dates:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)  # May 31 last training day
        test_start = train_end + pd.Timedelta(days=1)  # June 1 start testing
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)  # May 31 last testing day

        if test_end > data.index.max():
            break

        train_data = data[(data.index >= train_start) & (data.index <= train_end)]
        test_data = data[(data.index >= test_start) & (data.index <= test_end)]
        windows.append((train_data, test_data))

    return windows

# === RISK METRICS ===

# ---CVaR Helper Function------------------------------------------------------
def calculate_cvar(returns, beta=0.95):
    """
    Calculate CVaR using the mathematical definition from the framework
    CVaR_β[X] = E[X | X ≥ VaR_β[X]]
    where X = -portfolio_return (loss convention)
    
    Args:
        returns: array of portfolio returns 
        beta: confidence level (e.g., 0.95 for 95% CVaR)
    
    Returns:
        CVaR value (positive number representing expected loss magnitude in worst cases)
    """
    returns = np.array(returns)
    # Convert returns to losses: X = -return
    losses = -returns
    
    if len(losses) == 0:
        return 0.0
    
    # Calculate VaR (beta-quantile of losses)
    var_threshold = np.quantile(losses, beta)
    
    # CVaR is expected loss given loss exceeds VaR
    tail_losses = losses[losses >= var_threshold]
    
    if len(tail_losses) == 0:
        return var_threshold  # Edge case: return VaR if no tail
    
    return np.mean(tail_losses)

def calculate_sortino_ratio(returns, target_return=0.0, periods_per_year=52):
    """
    Calculate Sortino ratio: (Mean Return - Target) / Downside Deviation
    
    Args:
        returns: array of returns
        target_return: minimum acceptable return (default 0)
        periods_per_year: for annualization (52 for weekly)
    
    Returns:
        Annualized Sortino ratio
    """
    returns = np.array(returns)
    
    # Mean excess return
    mean_return = np.mean(returns)
    excess_return = mean_return - target_return
    
    # Downside deviation (only negative excess returns)
    downside_returns = returns - target_return
    downside_returns = downside_returns[downside_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if excess_return > 0 else 0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return np.inf if excess_return > 0 else 0
    
    # Annualized Sortino ratio
    sortino = (excess_return * np.sqrt(periods_per_year)) / (downside_deviation * np.sqrt(periods_per_year))
    return sortino

# === SCENARIO GENERATION ===

# ---GARCH---------------------------------------------------------------------
def fit_garch_models(data: pd.DataFrame, max_iter=1000):
    """
    Robust GARCH fitting with persistence checks
    """
    models = {}
    for col in data.columns:
        try:
            # Scale returns to percentage for numerical stability
            scaled_returns = data[col] * 100
            
            # Multiple attempts with different starting values
            best_model = None
            best_persistence = float('inf')
            
            for attempt in range(3):
                try:
                    am = arch_model(scaled_returns, vol='Garch', p=1, q=1)
                    res = am.fit(disp="off", options={'maxiter': max_iter})
                    
                    # Check persistence
                    if 'alpha[1]' in res.params and 'beta[1]' in res.params:
                        persistence = res.params['alpha[1]'] + res.params['beta[1]']
                        
                        # Accept if persistence is reasonable
                        if persistence < 0.99 and persistence < best_persistence:
                            best_model = res
                            best_persistence = persistence
                            break
                except:
                    continue
            
            if best_model is None:
                # Fallback: use historical volatility
                print(f"WARNING: GARCH failed for {col}, using historical volatility")
                models[col] = create_fallback_model(scaled_returns)
            else:
                models[col] = best_model
                
        except Exception as e:
            print(f"ERROR fitting GARCH for {col}: {e}")
            models[col] = create_fallback_model(data[col] * 100)
    
    return models

def create_fallback_model(returns):
    """Create a fallback model when GARCH fails"""
    class FallbackModel:
        def __init__(self, returns):
            self.params = {
                'mu': returns.mean(),
                'omega': returns.var() * 0.1,  # Long-run variance
                'alpha[1]': 0.1,
                'beta[1]': 0.8
            }
            self.conditional_volatility = pd.Series([returns.std()] * len(returns), index=returns.index)
            self.resid = returns - returns.mean()
    
    return FallbackModel(returns)

def make_positive_definite(matrix, min_eigenvalue=1e-8):
    """
    Convert a matrix to the nearest positive definite matrix
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def simulate_paths(models, cov_matrix, periods=12, num_paths=150):
    """
    GARCH simulation with persistence controls
    """
    N = len(models)
    
    # Make correlation matrix positive definite
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    try:
        chol_corr = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        corr_matrix = make_positive_definite(corr_matrix)
        chol_corr = np.linalg.cholesky(corr_matrix)
    
    sim_returns = np.zeros((num_paths, periods, N))

    for path in range(num_paths):
        independent_shocks = np.random.randn(periods, N)
        correlated_shocks = independent_shocks @ chol_corr.T

        for i, (ticker, res) in enumerate(models.items()):
            # Initialize with reasonable values
            if hasattr(res, 'conditional_volatility'):
                last_vol = res.conditional_volatility.iloc[-1]
                last_eps = res.resid.iloc[-1]
            else:
                last_vol = np.sqrt(res.params['omega'] / (1 - res.params['alpha[1]'] - res.params['beta[1]']))
                last_eps = 0
            
            params = res.params
            omega, alpha, beta = params['omega'], params['alpha[1]'], params['beta[1]']
            mu = params['mu']
            
            # Cap persistence to avoid explosive volatility
            if alpha + beta >= 0.99:
                alpha = 0.1
                beta = 0.85
            
            sigma2 = last_vol ** 2
            eps = last_eps

            for t in range(periods):
                # Update GARCH volatility with bounds
                sigma2 = omega + alpha * eps**2 + beta * sigma2
                sigma2 = np.clip(sigma2, omega, omega * 100)  # Reasonable bounds
                sigma = np.sqrt(sigma2)
                
                eps = sigma * correlated_shocks[t, i]
                sim_returns[path, t, i] = mu + eps

    return sim_returns / 100  # Convert back to decimal returns

def group_scenarios_by_history(scenarios, t):
    history_map = {}
    for s in range(scenarios.shape[0]):
        # Round to avoid floating-point artifacts
        history = tuple(np.round(scenarios[s, :t, :], decimals=6).flatten())
        if history not in history_map:
            history_map[history] = []
        history_map[history].append(s)
    return list(history_map.values())  # Each group shares history up to time t

def weekly_to_monthly_returns(weekly_scenarios):
    """Convert weekly scenarios to monthly by compounding every ~4.33 weeks"""
    num_scenarios, num_weeks, num_assets = weekly_scenarios.shape
    
    # Approximate monthly grouping (52 weeks / 12 months ≈ 4.33 weeks per month)
    weeks_per_month = [4, 4, 5, 4, 4, 5, 4, 4, 4, 5, 4, 4]  # Totals to 52
    
    monthly_scenarios = np.zeros((num_scenarios, 12, num_assets))
    
    for s in range(num_scenarios):
        week_idx = 0
        for month in range(12):
            # Compound returns over the weeks in this month
            month_returns = np.zeros(num_assets)
            for week in range(weeks_per_month[month]):
                if week_idx < num_weeks:
                    if week == 0:
                        month_returns = weekly_scenarios[s, week_idx, :]
                    else:
                        # Compound: (1+r1)*(1+r2) - 1 = r1 + r2 + r1*r2
                        month_returns = (1 + month_returns) * (1 + weekly_scenarios[s, week_idx, :]) - 1
                    week_idx += 1
            monthly_scenarios[s, month, :] = month_returns
    
    return monthly_scenarios

def simulate_dynamic_scenario_convergence(models, cov_matrix, periods=12, 
                                          start=50, step=50, max_paths=1000, threshold=0.001):
    """
    Dynamically increase the number of simulated paths until CVaR converges.
    """
    print("Testing scenario convergence dynamically...")
    cvar_estimates = {}
    prev_cvar = None
    num_paths = start

    while num_paths <= max_paths:
        np.random.seed(42)  # Fixed seed for comparison
        scenarios = simulate_paths(models, cov_matrix, periods=periods, num_paths=num_paths)

        N = scenarios.shape[2]
        equal_weights = np.ones(N) / N
        terminal_returns = []

        for s in range(num_paths):
            terminal_wealth = np.sum(equal_weights * (1 + scenarios[s, -1, :]))
            terminal_return = terminal_wealth - 1
            terminal_returns.append(terminal_return)

        cvar = calculate_cvar(terminal_returns, beta=0.95)
        cvar_estimates[num_paths] = cvar
        print(f"  {num_paths:3d} scenarios: CVaR = {cvar:.6f}")

        if prev_cvar is not None:
            diff = abs(cvar - prev_cvar)
            print(f"    Difference from previous: {diff:.6f}")
            if diff < threshold:
                print(f"  ✓ CVaR estimates appear to have converged at {num_paths} scenarios")
                return num_paths, cvar_estimates
        prev_cvar = cvar
        num_paths += step

    print("  ⚠ CVaR did not converge within max scenario limit")
    return num_paths - step, cvar_estimates  # fallback to last one

def validate_correlation_preservation(original_cov, simulated_returns):
    """
    Check how well the simulation preserves the correlation structure
    """
    sim_cov = np.cov(simulated_returns.reshape(-1, simulated_returns.shape[2]).T)
    
    # Compare correlation matrices
    orig_corr = original_cov / np.outer(np.sqrt(np.diag(original_cov)), np.sqrt(np.diag(original_cov)))
    sim_corr = sim_cov / np.outer(np.sqrt(np.diag(sim_cov)), np.sqrt(np.diag(sim_cov)))
    
    corr_diff = np.abs(orig_corr - sim_corr)
    print(f"Max correlation difference: {np.max(corr_diff):.4f}")
    print(f"Mean correlation difference: {np.mean(corr_diff):.4f}")
    
    return corr_diff

# === OTHER FUNCTIONS ===

#---Min allocation-------------------------------------------------------------
def apply_thresholding(weights, min_weight=0.005):
    """
    Apply thresholding and renormalization as described in the framework
    
    Args:
        weights: initial portfolio weights
        min_weight: minimum weight threshold (default 0.005)
    
    Returns:
        Thresholded and renormalized weights
    """
    weights = np.array(weights)
    # Remove assets below threshold
    weights[weights < min_weight] = 0
    # Renormalize remaining weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    return weights

def analyze_portfolio_allocation(weights, tickers, min_weight=1e-5):
    """Analyze portfolio allocation characteristics"""
    weights = np.array(weights)
    active_mask = weights > min_weight
    
    return {
        'num_assets': np.sum(active_mask),
        'active_assets': [tickers[i] for i in range(len(tickers)) if active_mask[i]],
        'active_weights': weights[active_mask],
        'max_weight': np.max(weights),
        'min_active_weight': np.min(weights[active_mask]) if np.any(active_mask) else 0,
        'concentration': np.sum(weights**2),  # Herfindahl index
        'effective_assets': 1 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0
    }

def check_garch_quality(garch_models, train_data):
    """Check if GARCH models are reasonable"""
    print("GARCH MODEL DIAGNOSTICS")
    print("="*30)
    
    for ticker, model in garch_models.items():
        params = model.params
        
        # Check persistence (alpha + beta should be < 1)
        if 'alpha[1]' in params and 'beta[1]' in params:
            persistence = params['alpha[1]'] + params['beta[1]']
            if persistence >= 0.99:
                print(f"WARNING: {ticker} has high persistence ({persistence:.3f}) - may be unreliable")
        
        # Check if volatility forecasts are reasonable
        last_vol = model.conditional_volatility.iloc[-1]
        historical_vol = train_data[ticker].std() * 100  # Convert to same scale
        
        if last_vol > 2 * historical_vol:
            print(f"WARNING: {ticker} GARCH vol ({last_vol:.2f}) much higher than historical ({historical_vol:.2f})")

# === C.I.'s ===

def compute_confidence_intervals_summary(results_df, confidence=0.95):
    """
    Compute confidence intervals for average performance metrics across all models
    """
    models = ['EqualWeight', 'MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR']
    metrics = ['return', 'cvar', 'sortino']
    
    def compute_ci(data, confidence=0.95):
        """Compute mean and confidence interval margin"""
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        # Use t-distribution for small samples (n=7 windows)
        t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = sem * t_value
        return mean, margin, (mean - margin, mean + margin)
    
    # Create summary table
    summary_results = []
    
    for model in models:
        row = {'Model': model}
        
        for metric in metrics:
            data = results_df[f'{model}_{metric}'].values
            mean, margin, (lower, upper) = compute_ci(data, confidence)
            
            # Store results
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci_margin'] = margin
            row[f'{metric}_ci_lower'] = lower
            row[f'{metric}_ci_upper'] = upper
            
        summary_results.append(row)
    
    return pd.DataFrame(summary_results)

def print_ci_summary(ci_df):
    """Print formatted confidence interval summary"""
    print("\n" + "="*80)
    print("95% CONFIDENCE INTERVALS FOR AVERAGE PERFORMANCE")
    print("="*80)
    
    for _, row in ci_df.iterrows():
        model = row['Model']
        print(f"\n{model}:")
        
        # Return
        ret_mean = row['return_mean']
        ret_lower = row['return_ci_lower'] 
        ret_upper = row['return_ci_upper']
        print(f"  Average Return: {ret_mean:.4f} [{ret_lower:.4f}, {ret_upper:.4f}]")
        
        # CVaR
        cvar_mean = row['cvar_mean']
        cvar_lower = row['cvar_ci_lower']
        cvar_upper = row['cvar_ci_upper'] 
        print(f"  Average CVaR:   {cvar_mean:.6f} [{cvar_lower:.6f}, {cvar_upper:.6f}]")
        
        # Sortino
        sortino_mean = row['sortino_mean']
        sortino_lower = row['sortino_ci_lower']
        sortino_upper = row['sortino_ci_upper']
        print(f"  Average Sortino: {sortino_mean:.3f} [{sortino_lower:.3f}, {sortino_upper:.3f}]")

def test_statistical_significance(results_df):
    """
    Test if differences between models are statistically significant
    Using paired t-tests since same windows are used for all models
    """
    models = ['EqualWeight', 'MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR']
    metrics = ['return', 'cvar', 'sortino']
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-tests)")
    print("="*80)
    
    for metric in metrics:
        print(f"\n{metric.upper()} COMPARISONS:")
        print("-" * 40)
        
        # Test MultistageCVaR vs others (since it has highest average return)
        base_model = 'MultistageCVaR'
        base_data = results_df[f'{base_model}_{metric}'].values
        
        for compare_model in models:
            if compare_model != base_model:
                compare_data = results_df[f'{compare_model}_{metric}'].values
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(base_data, compare_data)
                
                # Effect size (Cohen's d for paired samples)
                diff = base_data - compare_data
                cohen_d = np.mean(diff) / np.std(diff, ddof=1)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"  {base_model} vs {compare_model}: "
                      f"t={t_stat:.3f}, p={p_value:.4f} {significance}, "
                      f"Cohen's d={cohen_d:.3f}")

# === ASSET FREQUENCY PLOTTING ===

def plot_asset_frequencies_png(csv_path='asset_frequency.csv'):
    """
    Create PNG plots for asset frequencies:
    1. Individual PNG for each model
    2. Combined PNG with top 30 assets across all models
    """
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get unique models
    models = df['Model'].unique()
    
    print(f"Creating asset frequency plots for models: {models}")
    
    # 1. Individual plots for each model (saved as PNG)
    for model in models:
        model_data = df[df['Model'] == model].copy()
        model_data = model_data.sort_values('Frequency', ascending=False)  # Descending for better view
        
        # Create figure
        plt.figure(figsize=(16, 10))
        
        # Create color map based on frequency
        colors = plt.cm.viridis(model_data['Frequency'] / model_data['Frequency'].max())
        
        bars = plt.bar(range(len(model_data)), model_data['Frequency'], color=colors)
        
        plt.title(f'{model} - Asset Selection Frequency', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Assets', fontsize=14)
        plt.ylabel('Selection Frequency (out of 7 windows)', fontsize=14)
        plt.ylim(0, max(8, model_data['Frequency'].max() + 1))
        
        # Asset names on x-axis (rotated for readability)
        plt.xticks(range(len(model_data)), model_data['Asset'], rotation=45, ha='right', fontsize=10)
        
        # Add frequency values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add percentage labels as well
        for i, (bar, freq) in enumerate(zip(bars, model_data['Frequency'])):
            percentage = (freq / 7) * 100  # Assuming 7 windows
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'({percentage:.0f}%)', ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # Add grid and styling
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save as PNG
        filename = f'{model}_asset_frequency.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename}")
        plt.close()  # Close to free memory
    
    # 2. Combined plot with top 30 assets across all models
    print("\nCreating combined plot with top 30 assets...")
    
    # Find top 30 most frequently selected assets across all models
    asset_totals = df.groupby('Asset')['Frequency'].sum().sort_values(ascending=False)
    top_30_assets = asset_totals.head(30).index.tolist()
    
    # Filter data for top 30 assets
    top_30_data = df[df['Asset'].isin(top_30_assets)].copy()
    
    # Create pivot table for easier plotting
    pivot_data = top_30_data.pivot(index='Asset', columns='Model', values='Frequency').fillna(0)
    
    # Reorder assets by total frequency
    asset_order = pivot_data.sum(axis=1).sort_values(ascending=False).index
    pivot_data = pivot_data.reindex(asset_order)
    
    # Create the combined plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Set up colors for each model
    model_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Create grouped bar chart
    x = np.arange(len(pivot_data.index))
    width = 0.2  # Width of bars
    
    for i, (model, color) in enumerate(zip(models, model_colors)):
        if model in pivot_data.columns:
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, pivot_data[model], width, 
                         label=model, color=color, alpha=0.8)
            
            # Add value labels on bars (only for non-zero values)
            for j, (bar, value) in enumerate(zip(bars, pivot_data[model])):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                           f'{int(value)}', ha='center', va='bottom', 
                           fontsize=8, fontweight='bold')
    
    # Customize the plot
    ax.set_title('Top 30 Most Frequently Selected Assets - Model Comparison', 
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('Assets (FTSE 100 Tickers)', fontsize=14)
    ax.set_ylabel('Selection Frequency (out of 7 windows)', fontsize=14)
    ax.set_ylim(0, max(8, pivot_data.values.max() + 1))
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_filename = 'top_30_assets_comparison.png'
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {combined_filename}")
    plt.close()
    
    # 3. Create a horizontal version for better readability (bonus)
    print("Creating horizontal version of top 30 assets...")
    
    fig, ax = plt.subplots(figsize=(14, 16))
    
    y = np.arange(len(pivot_data.index))
    height = 0.2
    
    for i, (model, color) in enumerate(zip(models, model_colors)):
        if model in pivot_data.columns:
            offset = (i - len(models)/2 + 0.5) * height
            bars = ax.barh(y + offset, pivot_data[model], height, 
                          label=model, color=color, alpha=0.8)
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, pivot_data[model])):
                if value > 0:
                    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2.,
                           f'{int(value)}', ha='left', va='center', 
                           fontsize=8, fontweight='bold')
    
    ax.set_title('Top 30 Most Frequently Selected Assets - Model Comparison (Horizontal)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Assets (FTSE 100 Tickers)', fontsize=14)
    ax.set_xlabel('Selection Frequency (out of 7 windows)', fontsize=14)
    ax.set_xlim(0, max(8, pivot_data.values.max() + 1))
    
    # Set y-axis labels (reversed order for top-to-bottom reading)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot_data.index, fontsize=10)
    
    # Invert y-axis so highest frequency assets appear at top
    ax.invert_yaxis()
    
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save horizontal version
    horizontal_filename = 'top_30_assets_comparison_horizontal.png'
    plt.savefig(horizontal_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {horizontal_filename}")
    plt.close()
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Individual model plots created: {len(models)} files")
    print(f"Combined plots created: 2 files (vertical + horizontal)")
    print(f"Top 30 assets (by total frequency): {top_30_assets[:10]}... (showing first 10)")
    
    # Show some statistics
    print(f"\nTop 5 assets by total selection frequency:")
    for i, (asset, total_freq) in enumerate(asset_totals.head(5).items(), 1):
        print(f"  {i}. {asset}: {total_freq} total selections")
    
    return pivot_data

# === OPTIMISATION MODELS ===

#---Model 1--------------------------------------------------------------------
def solve_mean_variance_model(mu, Sigma, sigma_max, bounds=0.10):
    """
    Model 1: Mean-Variance Model
    
    Formulation:
    max μᵀx
    s.t. xᵀΣx ≤ σ²_max
         x ∈ X (sum to 1, bounds constraints)
    """
    N = len(mu)
    m = Model("Mean_Variance")
    m.setParam('OutputFlag', 0)
    
    # Decision variables: x ∈ R^N
    x = m.addVars(N, lb=0, ub=bounds, name="x")
    
    # Constraint: sum of weights = 1 (admissible portfolio set X)
    m.addConstr(sum(x[i] for i in range(N)) == 1, name="budget")
    
    # Risk constraint: xᵀΣx ≤ σ²_max
    variance = sum(x[i] * Sigma[i,j] * x[j] for i in range(N) for j in range(N))
    m.addConstr(variance <= sigma_max, name="variance_constraint")
    
    # Objective: maximize μᵀx
    m.setObjective(sum(mu[i] * x[i] for i in range(N)), GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        solution = [x[i].X for i in range(N)]
        return apply_thresholding(solution)
    else:
        print(f"Model 1: Optimization failed with status {m.status}")
        return None

#---Model 2--------------------------------------------------------------------
def solve_static_cvar_model(mu, scenarios, beta, cvar_max, bounds=0.10):
    """
    Model 2: Static CVaR-Constrained Portfolio Model
    
    Formulation:
    max μᵀx
    s.t. α + (1/(1-β)K) Σ ξₖ ≤ CVaR_max
         ξₖ ≥ -xᵀr⁽ᵏ⁾ - α  ∀k
         ξₖ ≥ 0  ∀k
         x ∈ X
    """
    N = len(mu)
    K = len(scenarios)
    m = Model("Static_CVaR")
    m.setParam('OutputFlag', 0)
    
    # Decision variables
    x = m.addVars(N, lb=0, ub=bounds, name="x")
    alpha = m.addVar(name="alpha")  # VaR estimate
    xi = m.addVars(K, lb=0, name="xi")  # Excess loss variables
    
    # Budget constraint
    m.addConstr(sum(x[i] for i in range(N)) == 1, name="budget")
    
    # CVaR constraint: α + (1/(1-β)K) Σ ξₖ ≤ CVaR_max
    cvar_expr = alpha + (1/((1-beta)*K)) * sum(xi[k] for k in range(K))
    m.addConstr(cvar_expr <= cvar_max, name="cvar_constraint")
    
    # Excess loss constraints: ξₖ ≥ -xᵀr⁽ᵏ⁾ - α  ∀k
    for k in range(K):
        portfolio_return_k = sum(x[i] * scenarios[k][i] for i in range(N))
        # Loss in scenario k: X_k = -portfolio_return_k
        loss_k = -portfolio_return_k
        m.addConstr(xi[k] >= loss_k - alpha, name=f"excess_loss_{k}")
    
    # Objective: maximize expected return
    m.setObjective(sum(mu[i] * x[i] for i in range(N)), GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        solution = [x[i].X for i in range(N)]
        return apply_thresholding(solution)
    else:
        print(f"Model 2: Optimization failed with status {m.status}")
        return None

#---Model 3--------------------------------------------------------------------
def solve_two_stage_cvar_model(mu, scenarios, beta, cvar_max, bounds=0.10, gamma=0.001):
    """
    Model 3: Two-Stage CVaR-Constrained Stochastic Programming Model
    
    Formulation:
    max μᵀx + (1/K) Σ [(r⁽ᵏ⁾)ᵀy⁽ᵏ⁾ - c⁽ᵏ⁾]
    s.t. α + (1/(1-β)K) Σ ξₖ ≤ CVaR_max
         ξₖ ≥ -[(r⁽ᵏ⁾)ᵀy⁽ᵏ⁾ - c⁽ᵏ⁾] - α  ∀k
         Δ⁽ᵏ⁾ = y⁽ᵏ⁾ - x  ∀k
         z⁽ᵏ⁾ ≥ Δ⁽ᵏ⁾, z⁽ᵏ⁾ ≥ -Δ⁽ᵏ⁾  ∀k
         c⁽ᵏ⁾ = γ × 1ᵀz⁽ᵏ⁾  ∀k
         x ∈ X, y⁽ᵏ⁾ ∈ X  ∀k
    """
    N = len(mu)
    K = len(scenarios)
    m = Model("Two_Stage_CVaR")
    m.setParam('OutputFlag', 0)
    
    # Decision variables
    x = m.addVars(N, lb=0, ub=bounds, name="x")  # First-stage decision
    y = m.addVars(K, N, lb=0, ub=bounds, name="y")  # Second-stage decisions
    alpha = m.addVar(name="alpha")  # VaR estimate
    xi = m.addVars(K, lb=0, name="xi")  # Excess loss variables
    
    # Transaction cost variables
    delta = m.addVars(K, N, lb=-1, ub=1, name="delta")  # Weight changes
    z = m.addVars(K, N, lb=0, name="z")  # Absolute weight changes
    c = m.addVars(K, lb=0, name="c")  # Transaction costs
    
    # Budget constraints
    m.addConstr(sum(x[i] for i in range(N)) == 1, name="first_stage_budget")
    for k in range(K):
        m.addConstr(sum(y[k,i] for i in range(N)) == 1, name=f"second_stage_budget_{k}")
    
    # Weight change constraints: Δ⁽ᵏ⁾ = y⁽ᵏ⁾ - x
    for k in range(K):
        for i in range(N):
            m.addConstr(delta[k,i] == y[k,i] - x[i], name=f"weight_change_{k}_{i}")
            # Absolute value: z⁽ᵏ⁾ ≥ |Δ⁽ᵏ⁾|
            m.addConstr(z[k,i] >= delta[k,i], name=f"abs_pos_{k}_{i}")
            m.addConstr(z[k,i] >= -delta[k,i], name=f"abs_neg_{k}_{i}")
    
    # Transaction cost: c⁽ᵏ⁾ = γ × 1ᵀz⁽ᵏ⁾
    for k in range(K):
        m.addConstr(c[k] == gamma * sum(z[k,i] for i in range(N)), name=f"transaction_cost_{k}")
    
    # CVaR constraint
    cvar_expr = alpha + (1/((1-beta)*K)) * sum(xi[k] for k in range(K))
    m.addConstr(cvar_expr <= cvar_max, name="cvar_constraint")
    
    # Excess loss constraints: ξₖ ≥ -[(r⁽ᵏ⁾)ᵀy⁽ᵏ⁾ - c⁽ᵏ⁾] - α
    for k in range(K):
        portfolio_return_k = sum(y[k,i] * scenarios[k][i] for i in range(N))
        net_return_k = portfolio_return_k - c[k]  # After transaction costs
        loss_k = -net_return_k
        m.addConstr(xi[k] >= loss_k - alpha, name=f"excess_loss_{k}")
    
    # Objective: maximize first-stage return + expected second-stage return - expected transaction costs
    first_stage_return = sum(mu[i] * x[i] for i in range(N))
    expected_second_stage_return = (1/K) * sum(sum(y[k,i] * scenarios[k][i] for i in range(N)) for k in range(K))
    expected_transaction_costs = (1/K) * sum(c[k] for k in range(K))
    
    m.setObjective(first_stage_return + expected_second_stage_return - expected_transaction_costs, GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        # Return both first-stage and second-stage solutions
        first_stage = apply_thresholding([x[i].X for i in range(N)])
        second_stage = [[y[k,i].X for i in range(N)] for k in range(K)]
        return first_stage, second_stage
    else:
        print(f"Model 3: Optimization failed with status {m.status}")
        return None, None

#---Model 4--------------------------------------------------------------------
def solve_multistage_cvar_model(scenarios, beta, cvar_max, bounds=0.10, gamma=0.001):
    """
    Model 4: Multistage CVaR-Constrained Portfolio Model
    
    Formulation:
    max Σ pₛ [1 + Σₜ (xₜˢᵀrₜˢ - cₜˢ)]
    s.t. ξₛ ≥ -Σₜ xₜˢᵀrₜˢ + Σₜ cₜˢ - α  ∀s
         α + (1/(1-β)) Σ pₛξₛ ≤ CVaR_max
         xₜˢ = xₜˢ' if s ~ₜ s' (non-anticipativity)
         xₜˢ ∈ X  ∀t,s
    """
    num_scenarios, T, N = scenarios.shape
    m = Model("Multistage_CVaR")
    m.setParam('OutputFlag', 0)
    
    # Equal probability scenarios
    p_s = 1.0 / num_scenarios
    
    # Decision variables
    x = {}  # Portfolio weights
    c = {}  # Transaction costs
    z = {}  # Absolute weight changes (for transaction costs)
    
    for s in range(num_scenarios):
        for t in range(T):
            for i in range(N):
                x[s, t, i] = m.addVar(lb=0, ub=bounds, name=f"x_{s}_{t}_{i}")
            c[s, t] = m.addVar(lb=0, name=f"c_{s}_{t}")
            if t > 0:  # No transaction costs at t=0
                for i in range(N):
                    z[s, t, i] = m.addVar(lb=0, name=f"z_{s}_{t}_{i}")
    
    # CVaR variables
    alpha = m.addVar(name="alpha")
    xi = m.addVars(num_scenarios, lb=0, name="xi")
    
    # Budget constraints: portfolio weights sum to 1
    for s in range(num_scenarios):
        for t in range(T):
            m.addConstr(sum(x[s, t, i] for i in range(N)) == 1, name=f"budget_{s}_{t}")
    
    # Non-anticipativity constraints
    for t in range(T):
        history_groups = group_scenarios_by_history(scenarios, t)
        for group in history_groups:
            if len(group) > 1:
                for i in range(N):
                    for s1, s2 in combinations(group, 2):
                        m.addConstr(x[s1, t, i] == x[s2, t, i], name=f"nonanticipativity_{s1}_{s2}_{t}_{i}")
    
    # Transaction cost constraints
    for s in range(num_scenarios):
        # No transaction cost at t=0
        m.addConstr(c[s, 0] == 0, name=f"no_initial_cost_{s}")
        
        for t in range(1, T):
            # Absolute weight changes: z[s,t,i] ≥ |x[s,t,i] - x[s,t-1,i]|
            for i in range(N):
                m.addConstr(z[s, t, i] >= x[s, t, i] - x[s, t-1, i], name=f"abs_pos_{s}_{t}_{i}")
                m.addConstr(z[s, t, i] >= x[s, t-1, i] - x[s, t, i], name=f"abs_neg_{s}_{t}_{i}")
            
            # Transaction cost: c[s,t] = γ * Σᵢ z[s,t,i]
            m.addConstr(c[s, t] == gamma * sum(z[s, t, i] for i in range(N)), name=f"transaction_cost_{s}_{t}")
    
    # CVaR constraints
    for s in range(num_scenarios):
        # Terminal wealth calculation
        cumulative_return = sum(sum(x[s, t, i] * scenarios[s, t, i] for i in range(N)) for t in range(T))
        total_costs = sum(c[s, t] for t in range(T))
        terminal_return = cumulative_return - total_costs
        
        # Loss = -terminal_return
        loss_s = -terminal_return
        m.addConstr(xi[s] >= loss_s - alpha, name=f"excess_loss_{s}")
    
    # CVaR constraint
    cvar_expr = alpha + (1/(1-beta)) * sum(p_s * xi[s] for s in range(num_scenarios))
    m.addConstr(cvar_expr <= cvar_max, name="cvar_constraint")
    
    # Objective: maximize expected terminal wealth
    expected_terminal_wealth = sum(p_s * (1 + sum(sum(x[s, t, i] * scenarios[s, t, i] for i in range(N)) - c[s, t] for t in range(T))) for s in range(num_scenarios))
    m.setObjective(expected_terminal_wealth, GRB.MAXIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        # Extract solution: [scenario][time][asset]
        solution = [[[x[s, t, i].X for i in range(N)] for t in range(T)] for s in range(num_scenarios)]
        return solution
    else:
        print(f"Model 4: Optimization failed with status {m.status}")
        return None

# === MAIN EXECUTION ===

def main():
    # Configuration
    DATA_PATH = "2015_2025_returns.csv"
    RISK_WINDOWS_PATH = "risk_parameters.csv"
    beta = 0.95
    bounds = 0.05
    monthly_periods = 12
    gamma = 0.001
    
    # Load data
    print("Loading data and initializing analysis...")
    log_returns = load_log_returns(DATA_PATH)
    risk_windows_df = load_risk_windows(RISK_WINDOWS_PATH)
    windows = get_rolling_windows(log_returns)
    tickers = log_returns.columns.tolist()
    
    # Initialise storage
    results = []
    timing_results = []
    allocation_results = []
    asset_frequency = {f'{model}_{asset}': 0 for model in ['MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR'] for asset in tickers}
    model_names = ['EqualWeight', 'MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR']
    
    print(f"Analysing {len(windows)} windows with {len(tickers)} assets")
    print("=" * 70)
    
    total_start_time = time.time()
    
    for i, (train, test) in enumerate(windows):
        print(f"Window {i+1}/{len(windows)}: {train.index.min().date()} → {test.index.max().date()}")
        
        window_start_time = time.time()
        np.random.seed(13 + i) # Set seed for reproducibility
        
        # Get risk parameters
        row = risk_windows_df.iloc[i]
        sigma_max = row['Variance_Weekly']
        cvar_max_weekly = row['CVaR_Weekly_95']
        cvar_max_monthly = row['CVaR_Monthly_95']
        
        # Estimate parameters from historical data
        mu = train.mean().values
        Sigma = train.cov().values
        hist_scenarios = train.values.tolist()
        
        # GARCH modeling
        garch_start = time.time()
        garch_models = fit_garch_models(train)
        cov_matrix = train.cov().values
        
        # CVaR convergence check
        best_num_paths, _ = simulate_dynamic_scenario_convergence(garch_models, cov_matrix)
        
        # Generate scenarios for multistage model
        weekly_scenarios = simulate_paths(garch_models, cov_matrix, periods=52, num_paths=best_num_paths)
        monthly_garch_scenarios = weekly_to_monthly_returns(weekly_scenarios)
        garch_time = time.time() - garch_start
        
        # Check one GARCH model for quality
        #check_garch_quality(garch_models, train)
         
        # Solve models
        timing = {'window': i+1, 'garch_time': garch_time}
        
        # Model 1: Mean-Variance
        start_time = time.time()
        x1 = solve_mean_variance_model(mu, Sigma, sigma_max, bounds)
        timing['model1_time'] = time.time() - start_time
        
        # Model 2: Static CVaR
        start_time = time.time()
        x2 = solve_static_cvar_model(mu, hist_scenarios, beta, cvar_max_weekly, bounds)
        timing['model2_time'] = time.time() - start_time
        
        # Model 3: Two-Stage CVaR
        start_time = time.time()
        x3_first, x3_second = solve_two_stage_cvar_model(mu, hist_scenarios, beta, cvar_max_weekly, bounds, gamma)
        timing['model3_time'] = time.time() - start_time
        
        # Model 4: Multistage CVaR
        start_time = time.time()
        x4 = solve_multistage_cvar_model(monthly_garch_scenarios, beta, cvar_max_monthly, bounds, gamma)
        timing['model4_time'] = time.time() - start_time
        
        timing['total_window_time'] = time.time() - window_start_time
        
        # Handle failed optimisations and prepare weights
        if x1 is None:
            x1 = np.ones(len(mu)) / len(mu)
        if x2 is None:
            x2 = np.ones(len(mu)) / len(mu)
        if x3_first is None:
            x3_first = np.ones(len(mu)) / len(mu)
        
        # Fix: Handle multistage model output properly
        if x4 is None:
            x4_test = np.ones(len(mu)) / len(mu)
        else:
            # x4 is [scenario][time][asset], we want the first time period weights
            x4_first = np.mean([x4[s][0] for s in range(len(x4))], axis=0)
            x4_test = apply_thresholding(x4_first)
        
        # Analyse allocations
        ew_weights = np.ones(len(mu)) / len(mu)
        portfolios = {
            'EqualWeight': ew_weights,
            'MeanVariance': x1,
            'StaticCVaR': x2,
            'TwoStageCVaR': x3_first,
            'MultistageCVaR': x4_test
        }
        
        # After solving models, check concentration:
        print(f"Window {i+1} Concentration Analysis:")
    
        models = {'EqualWeight': ew_weights, 'MeanVariance': x1, 'StaticCVaR': x2, 
              'TwoStageCVaR': x3_first, 'MultistageCVaR': x4_test}
    
        for name, weights in models.items():
            active_assets = np.sum(weights > 0.001)
            max_weight = np.max(weights)
            herfindahl = np.sum(weights**2)
            top5_weight = np.sum(np.sort(weights)[-5:])  # Top 5 assets weight
        
            print(f"  {name}: {active_assets} assets, max={max_weight:.1%}, "
                  f"HHI={herfindahl:.3f}, top5={top5_weight:.1%}")
        
        window_allocations = {'window': i+1}
        for model, weights in portfolios.items():
            analysis = analyze_portfolio_allocation(weights, tickers)
            window_allocations[f'{model}_num_assets'] = analysis['num_assets']
            window_allocations[f'{model}_concentration'] = analysis['concentration']
            window_allocations[f'{model}_max_weight'] = analysis['max_weight']
            
            # Track asset frequency (exclude equal weight)
            if model != 'EqualWeight':
                for asset in analysis['active_assets']:
                    key = f'{model}_{asset}'
                    if key in asset_frequency:
                        asset_frequency[key] += 1
        
        allocation_results.append(window_allocations)
        
        # Calculate test performance
        test_returns = test.values
        
        performance = {'window': i+1}
        for model, weights in portfolios.items():
            test_r = test_returns @ weights
            performance[f'{model}_return'] = np.prod(1 + test_r) - 1
            performance[f'{model}_cvar'] = calculate_cvar(test_r, beta)
            performance[f'{model}_sortino'] = calculate_sortino_ratio(test_r)
        
        results.append(performance)
        timing_results.append(timing)
        
        # Print window summary
        solve_times = [timing['model1_time'], timing['model2_time'], timing['model3_time'], timing['model4_time']]
        num_assets = [window_allocations[f'{model}_num_assets'] for model in model_names[1:]]  # Exclude EW
        
        print(f"  Assets selected: {num_assets} | Solve times: {[f'{t:.2f}s' for t in solve_times]}")
    
    total_time = time.time() - total_start_time
    
    # Save results
    results_df = pd.DataFrame(results)
    timing_df = pd.DataFrame(timing_results)
    allocation_df = pd.DataFrame(allocation_results)

    
    results_df.to_csv("performance_results.csv", index=False)
    timing_df.to_csv("timing_results.csv", index=False)
    allocation_df.to_csv("allocation_analysis.csv", index=False)
    
    # Create asset frequency analysis
    freq_data = []
    for key, count in asset_frequency.items():
        if count > 0:  # Only include assets that were actually selected
            model, asset = key.split('_', 1)
            freq_data.append({
                'Model': model,
                'Asset': asset,
                'Frequency': count,
                'Percentage': (count / len(windows)) * 100
            })
    
    freq_df = pd.DataFrame(freq_data)
    freq_df.to_csv("asset_frequency.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per window: {total_time/len(windows):.1f}s")
    
    print(f"\nAverage solve times:")
    for i, model in enumerate(['MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR']):
        avg_time = timing_df[f'model{i+1}_time'].mean()
        print(f"  {model}: {avg_time:.3f}s")
    
    print(f"\nAverage number of assets selected:")
    for model in model_names[1:]:  # Exclude EW
        avg_assets = allocation_df[f'{model}_num_assets'].mean()
        print(f"  {model}: {avg_assets:.1f}")
    
    print(f"\nAverage performance:")
    for model in model_names:
        avg_return = results_df[f'{model}_return'].mean()
        avg_cvar = results_df[f'{model}_cvar'].mean()
        avg_sortino = results_df[f'{model}_sortino'].mean()
        print(f"  {model}: Return={avg_return:.4f}, CVaR={avg_cvar:.6f}, Sortino={avg_sortino:.3f}")
    
    print(f"\nMost frequently selected assets (>50% of windows):")
    for model in ['MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR']:
        if not freq_df.empty:
            model_freq = freq_df[freq_df['Model'] == model]
            popular = model_freq[model_freq['Percentage'] > 50].sort_values('Percentage', ascending=False)
            if not popular.empty:
                assets = popular['Asset'].tolist()[:5]  # Top 5
                print(f"  {model}: {assets}")
            else:
                print(f"  {model}: No assets selected >50% of the time")
        else:
            print(f"  {model}: No frequency data available")
    
    print(f"\nFiles saved:")
    print(f"  - performance_results.csv: Portfolio performance metrics")
    print(f"  - timing_results.csv: Detailed timing information")
    print(f"  - allocation_analysis.csv: Portfolio allocation statistics")
    print(f"  - asset_frequency.csv: Asset selection frequency by model")
    
    # List of models
    models = ['EqualWeight', 'MeanVariance', 'StaticCVaR', 'TwoStageCVaR', 'MultistageCVaR']

    # Use DataFrame
    df = results_df
    
    # Compute Confidence Intervals
    ci_df = compute_confidence_intervals_summary(results_df)
    ci_df.to_csv("confidence_intervals_detailed.csv", index=False)

    # Print formatted summary
    print_ci_summary(ci_df)

    # Test statistical significance
    test_statistical_significance(results_df)

    # Also create a clean summary table for dissertation
    def create_dissertation_table(ci_df):
        """Create a clean table"""
        print("\n" + "="*80)
        print("AVERAGE PERFORMANCE WITH 95% CONFIDENCE INTERVALS")
        print("="*80)
        print(f"{'Model':<15} {'Return (%)':<20} {'CVaR (%)':<20} {'Sortino Ratio':<20}")
        print("-" * 75)
    
        for _, row in ci_df.iterrows():
            model = row['Model']
        
            # Format with CI in brackets
            ret_str = f"{row['return_mean']*100:.2f} [{row['return_ci_lower']*100:.2f}, {row['return_ci_upper']*100:.2f}]"
            cvar_str = f"{row['cvar_mean']*100:.3f} [{row['cvar_ci_lower']*100:.3f}, {row['cvar_ci_upper']*100:.3f}]"
            sortino_str = f"{row['sortino_mean']:.3f} [{row['sortino_ci_lower']:.3f}, {row['sortino_ci_upper']:.3f}]"
        
            print(f"{model:<15} {ret_str:<20} {cvar_str:<20} {sortino_str:<20}")

    create_dissertation_table(ci_df)
    
    # --- Plot Returns ---
    plt.figure(figsize=(12, 6))
    for model in models:
        plt.plot(df['window'], df[f'{model}_return'], label=model)

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Portfolio Returns Across Windows")
    plt.xlabel("Window")
    plt.ylabel("Test Period Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot CVaR ---
    plt.figure(figsize=(12, 6))
    for model in models:
        plt.plot(df['window'], df[f'{model}_cvar'], label=model)

    plt.title("Portfolio CVaR (95%) Across Windows")
    plt.xlabel("Window")
    plt.ylabel("CVaR (Expected Loss in Worst 5%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
    
if __name__ == "__main__":
    # Make sure to run this after your main analysis has created asset_frequency.csv
    plot_data = plot_asset_frequencies_png('asset_frequency.csv')