#!/usr/bin/env python3
"""
Benchmark Visualization Script
==============================
Generates comparison plots from benchmark results.

Usage:
    python visualize_benchmark.py [--input benchmark_results.pkl] [--output ./figures]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import Optional

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'PCE': '#2ecc71',
    'NGD': '#3498db', 
    'IPOPT': '#e74c3c',
    'ADAM': '#9b59b6',
    'LBFGS': '#f39c12',
    'SCP': '#1abc9c',
    'SQP': '#34495e'
}


def load_results(pickle_path: str) -> pd.DataFrame:
    """Load benchmark results from pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Results file not found: {pickle_path}")
    return pd.read_pickle(pickle_path)


def prepare_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert results to long format for easier plotting."""
    rows = []
    
    for _, row in df.iterrows():
        seed = row['seed']
        casadi_variant = row['casadi_variant']
        
        # PCE
        rows.append({
            'seed': seed,
            'planner': 'PCE',
            'cost': row['pce_cost'],
            'time': row['pce_time']
        })
        
        # NGD
        rows.append({
            'seed': seed,
            'planner': 'NGD', 
            'cost': row['ngd_cost'],
            'time': row['ngd_time']
        })
        
        # CasADi variant
        rows.append({
            'seed': seed,
            'planner': casadi_variant,
            'cost': row['casadi_cost'],
            'time': row['casadi_time']
        })
    
    return pd.DataFrame(rows)


def plot_cost_comparison_boxplot(df_long: pd.DataFrame, output_dir: str, 
                                  success_threshold: float = 10000.0):
    """Create boxplot comparing costs across all planners."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter out failures for cleaner visualization
    df_success = df_long[df_long['cost'] < success_threshold]
    
    planners = df_success['planner'].unique()
    colors = [COLORS.get(p, '#95a5a6') for p in planners]
    
    box = ax.boxplot(
        [df_success[df_success['planner'] == p]['cost'] for p in planners],
        labels=planners,
        patch_artist=True
    )
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_xlabel('Planner', fontsize=12)
    ax.set_title('Cost Comparison Across Planners (Successful Runs Only)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_boxplot.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'cost_boxplot.pdf'))
    plt.close()
    print(f"[SAVED] cost_boxplot.png/pdf")


def plot_time_comparison(df_long: pd.DataFrame, output_dir: str):
    """Create bar chart comparing computation times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_stats = df_long.groupby('planner')['time'].agg(['mean', 'std'])
    time_stats = time_stats.sort_values('mean')
    
    planners = time_stats.index.tolist()
    means = time_stats['mean'].values
    stds = time_stats['std'].values
    colors = [COLORS.get(p, '#95a5a6') for p in planners]
    
    bars = ax.bar(planners, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xlabel('Planner', fontsize=12)
    ax.set_title('Average Computation Time Comparison', fontsize=14)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'time_comparison.pdf'))
    plt.close()
    print(f"[SAVED] time_comparison.png/pdf")


def plot_success_rate(df_long: pd.DataFrame, output_dir: str, 
                       threshold: float = 10000.0):
    """Create bar chart showing success rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    success_rates = df_long.groupby('planner').apply(
        lambda x: (x['cost'] < threshold).mean() * 100
    ).sort_values(ascending=False)
    
    planners = success_rates.index.tolist()
    rates = success_rates.values
    colors = [COLORS.get(p, '#95a5a6') for p in planners]
    
    bars = ax.bar(planners, rates, color=colors, alpha=0.8)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Planner', fontsize=12)
    ax.set_title(f'Success Rate (Cost < {threshold})', fontsize=14)
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'success_rate.pdf'))
    plt.close()
    print(f"[SAVED] success_rate.png/pdf")


def plot_cost_vs_time_scatter(df_long: pd.DataFrame, output_dir: str,
                               success_threshold: float = 10000.0):
    """Scatter plot of cost vs time for each planner."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    df_success = df_long[df_long['cost'] < success_threshold]
    
    for planner in df_success['planner'].unique():
        subset = df_success[df_success['planner'] == planner]
        color = COLORS.get(planner, '#95a5a6')
        ax.scatter(subset['time'], subset['cost'], 
                   label=planner, alpha=0.6, s=50, c=color)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Cost vs. Computation Time', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_vs_time.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'cost_vs_time.pdf'))
    plt.close()
    print(f"[SAVED] cost_vs_time.png/pdf")


def plot_seed_comparison(df: pd.DataFrame, output_dir: str):
    """Plot cost trajectories across seeds for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get unique CasADi variants
    casadi_variants = df['casadi_variant'].unique()
    
    # Left: PCE vs NGD vs CasADi variants (Cost)
    ax1 = axes[0]
    seeds = sorted(df['seed'].unique())
    
    # PCE and NGD (use first variant's data since they're the same)
    first_variant_df = df[df['casadi_variant'] == casadi_variants[0]]
    ax1.plot(first_variant_df['seed'], first_variant_df['pce_cost'], 
             'o-', label='PCE', color=COLORS['PCE'], alpha=0.7, markersize=4)
    ax1.plot(first_variant_df['seed'], first_variant_df['ngd_cost'],
             's-', label='NGD', color=COLORS['NGD'], alpha=0.7, markersize=4)
    
    # CasADi variants
    for variant in casadi_variants:
        variant_df = df[df['casadi_variant'] == variant]
        color = COLORS.get(variant, '#95a5a6')
        ax1.plot(variant_df['seed'], variant_df['casadi_cost'],
                 '^-', label=variant, color=color, alpha=0.7, markersize=4)
    
    ax1.set_xlabel('Seed', fontsize=11)
    ax1.set_ylabel('Cost', fontsize=11)
    ax1.set_title('Cost per Seed', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_yscale('log')
    
    # Right: Summary statistics as grouped bar chart
    ax2 = axes[1]
    
    planners = ['PCE', 'NGD'] + list(casadi_variants)
    means = []
    stds = []
    
    means.append(first_variant_df['pce_cost'].mean())
    stds.append(first_variant_df['pce_cost'].std())
    means.append(first_variant_df['ngd_cost'].mean())
    stds.append(first_variant_df['ngd_cost'].std())
    
    for variant in casadi_variants:
        variant_df = df[df['casadi_variant'] == variant]
        means.append(variant_df['casadi_cost'].mean())
        stds.append(variant_df['casadi_cost'].std())
    
    colors = [COLORS.get(p, '#95a5a6') for p in planners]
    x = np.arange(len(planners))
    
    bars = ax2.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(planners, rotation=45, ha='right')
    ax2.set_ylabel('Mean Cost', fontsize=11)
    ax2.set_title('Mean Cost Comparison', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'seed_comparison.pdf'))
    plt.close()
    print(f"[SAVED] seed_comparison.png/pdf")


def generate_latex_table(df: pd.DataFrame, output_dir: str,
                          success_threshold: float = 10000.0):
    """Generate LaTeX table for paper inclusion."""
    df_long = prepare_long_format(df)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Motion Planning Benchmark Results}",
        r"\label{tab:benchmark_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Planner & Mean Cost & Std & Best 10\% & Worst 10\% & Time (s) & Success \\",
        r"\midrule"
    ]
    
    for planner in df_long['planner'].unique():
        subset = df_long[df_long['planner'] == planner]
        
        mean_cost = subset['cost'].mean()
        std_cost = subset['cost'].std()
        
        n_10 = max(1, int(len(subset) * 0.1))
        sorted_costs = subset['cost'].sort_values()
        best_10 = sorted_costs.head(n_10).mean()
        worst_10 = sorted_costs.tail(n_10).mean()
        
        avg_time = subset['time'].mean()
        success = (subset['cost'] < success_threshold).mean() * 100
        
        lines.append(
            f"{planner} & {mean_cost:.2f} & {std_cost:.2f} & {best_10:.2f} & "
            f"{worst_10:.2f} & {avg_time:.3f} & {success:.1f}\\% \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    latex_path = os.path.join(output_dir, 'benchmark_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[SAVED] benchmark_table.tex")


def main():
    parser = argparse.ArgumentParser(description="Visualize Benchmark Results")
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='benchmark_results.pkl',
        help='Path to benchmark results pickle file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=10000.0,
        help='Success threshold for cost'
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"[INFO] Loading results from: {args.input}")
    df = load_results(args.input)
    print(f"[INFO] Loaded {len(df)} result rows")
    
    # Prepare long format
    df_long = prepare_long_format(df)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    plot_cost_comparison_boxplot(df_long, args.output, args.threshold)
    plot_time_comparison(df_long, args.output)
    plot_success_rate(df_long, args.output, args.threshold)
    plot_cost_vs_time_scatter(df_long, args.output, args.threshold)
    plot_seed_comparison(df, args.output)
    generate_latex_table(df, args.output, args.threshold)
    
    print(f"\n[DONE] All figures saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())