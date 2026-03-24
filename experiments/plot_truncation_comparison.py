#!/usr/bin/env python3
"""
Plot script for comparing PPO with standard vs correct truncation handling.

This script reads TensorBoard logs and generates publication-quality plots
comparing the two implementations across multiple environments and seeds.

Usage:
    uv run python experiments/plot_truncation_comparison.py --logdir runs/
    
    # Or with custom options:
    uv run python experiments/plot_truncation_comparison.py \
        --logdir runs/ \
        --output-dir experiments/figures \
        --format pdf \
        --smoothing 0.9
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for PPO truncation handling experiment"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs/",
        help="Directory containing TensorBoard logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/figures",
        help="Directory to save generated figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for figures",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="Exponential moving average smoothing weight (0-1)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8, 5],
        help="Figure size in inches (width height)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats (png)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Base font size for plots",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering (useful if LaTeX is not installed)",
    )
    return parser.parse_args()


def smooth(values, weight):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def load_tensorboard_logs(logdir):
    """
    Load all TensorBoard logs from the specified directory.
    
    Returns a dictionary structured as:
    {
        env_id: {
            algorithm: {
                seed: DataFrame with columns [step, episodic_return]
            }
        }
    }
    """
    data = defaultdict(lambda: defaultdict(dict))
    logdir = Path(logdir)
    
    if not logdir.exists():
        raise ValueError(f"Log directory does not exist: {logdir}")
    
    # Find all event files
    run_dirs = [d for d in logdir.iterdir() if d.is_dir()]
    
    for run_dir in run_dirs:
        # Parse run name: {env_id}__{exp_name}__{seed}__{timestamp}
        parts = run_dir.name.split("__")
        if len(parts) < 3:
            continue
            
        env_id = parts[0]
        exp_name = parts[1]
        try:
            seed = int(parts[2])
        except ValueError:
            continue
        
        # Determine algorithm type from exp_name
        # Handles both discrete (ppo_standard / ppo_correct_truncation) and
        # continuous (ppo_continuous_action / ppo_continuous_action_correct_truncation)
        if "correct_truncation" in exp_name:
            algo = "PPO (Correct Truncation)"
        elif "ppo" in exp_name.lower():
            algo = "PPO (Standard)"
        else:
            continue
        
        # Load TensorBoard events
        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            
            # Get episodic return data
            if "charts/episodic_return" in ea.Tags()["scalars"]:
                events = ea.Scalars("charts/episodic_return")
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                df = pd.DataFrame({
                    "step": steps,
                    "episodic_return": values,
                })
                data[env_id][algo][seed] = df
                print(f"Loaded: {env_id} / {algo} / seed {seed} ({len(df)} points)")
        except Exception as e:
            print(f"Warning: Could not load {run_dir}: {e}")
            continue
    
    return data


def interpolate_to_common_steps(data, num_points=500):
    """
    Interpolate all runs to a common set of steps for fair comparison.
    
    Returns a DataFrame with columns: [step, episodic_return, algorithm, seed, env_id]
    """
    all_data = []
    
    for env_id, algos in data.items():
        # Find the common step range across all runs for this environment
        all_max_steps = []
        for algo, seeds in algos.items():
            for seed, df in seeds.items():
                if len(df) > 0:
                    all_max_steps.append(df["step"].max())
        
        if not all_max_steps:
            continue
            
        max_step = min(all_max_steps)  # Use the minimum max to ensure all runs have data
        common_steps = np.linspace(0, max_step, num_points)
        
        for algo, seeds in algos.items():
            for seed, df in seeds.items():
                if len(df) == 0:
                    continue
                    
                # Interpolate to common steps
                interpolated_values = np.interp(
                    common_steps,
                    df["step"].values,
                    df["episodic_return"].values,
                )
                
                run_df = pd.DataFrame({
                    "step": common_steps,
                    "episodic_return": interpolated_values,
                    "algorithm": algo,
                    "seed": seed,
                    "env_id": env_id,
                })
                all_data.append(run_df)
    
    if not all_data:
        raise ValueError("No data found in logs")
        
    return pd.concat(all_data, ignore_index=True)


def apply_smoothing(df, weight, group_cols=["env_id", "algorithm", "seed"]):
    """Apply smoothing to each run independently."""
    def smooth_group(group):
        group = group.copy()
        group["episodic_return_smooth"] = smooth(
            group["episodic_return"].values, weight
        )
        return group
    
    return df.groupby(group_cols, group_keys=False).apply(smooth_group)


def setup_style(args):
    """Set up matplotlib style for publication-quality figures."""
    # Use a clean style
    plt.style.use("seaborn-whitegrid")
    
    # Set up LaTeX rendering if available
    if not args.no_latex:
        try:
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
            plt.rcParams["font.family"] = "serif"
        except Exception:
            print("Warning: LaTeX not available, using default text rendering")
            plt.rcParams["text.usetex"] = False
    
    # Font sizes
    plt.rcParams["font.size"] = args.font_size
    plt.rcParams["axes.titlesize"] = args.font_size + 2
    plt.rcParams["axes.labelsize"] = args.font_size
    plt.rcParams["xtick.labelsize"] = args.font_size - 1
    plt.rcParams["ytick.labelsize"] = args.font_size - 1
    plt.rcParams["legend.fontsize"] = args.font_size - 1
    
    # Line widths
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.linewidth"] = 1
    
    # Colors - using a colorblind-friendly palette
    return {
        "PPO (Standard)": "#0077BB",  # Blue
        "PPO (Correct Truncation)": "#EE7733",  # Orange
    }


def plot_learning_curves(df, env_id, colors, args, show_raw=False):
    """Generate learning curve plot for a single environment."""
    env_data = df[df["env_id"] == env_id].copy()
    
    fig, ax = plt.subplots(figsize=args.figsize)
    
    for algo in ["PPO (Standard)", "PPO (Correct Truncation)"]:
        algo_data = env_data[env_data["algorithm"] == algo]
        if len(algo_data) == 0:
            continue
        
        # Plot individual seeds with low alpha if requested
        if show_raw:
            for seed in algo_data["seed"].unique():
                seed_data = algo_data[algo_data["seed"] == seed]
                ax.plot(
                    seed_data["step"],
                    seed_data["episodic_return"],
                    color=colors[algo],
                    alpha=0.15,
                    linewidth=0.5,
                )
        
        # Compute mean and std across seeds
        grouped = algo_data.groupby("step")["episodic_return_smooth"].agg(["mean", "std"])
        
        # Plot mean line
        ax.plot(
            grouped.index,
            grouped["mean"],
            color=colors[algo],
            label=algo,
            linewidth=2,
        )
        
        # Plot confidence interval (mean ± std)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            color=colors[algo],
            alpha=0.2,
        )
    
    # Formatting
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title(env_id.replace("-", " ").replace("_", " "))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="lower right")
    
    # Ensure y-axis starts at a reasonable value
    ymin, ymax = ax.get_ylim()
    if ymin < 0 and ymax > 0:
        ax.set_ylim(bottom=min(ymin, -0.1 * ymax))
    
    plt.tight_layout()
    return fig


def plot_combined_grid(df, colors, args):
    """Generate a grid of learning curves for all environments."""
    envs = sorted(df["env_id"].unique())
    n_envs = len(envs)
    
    if n_envs == 0:
        return None
    
    # Determine grid layout
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(args.figsize[0] * n_cols * 0.8, args.figsize[1] * n_rows * 0.8),
        squeeze=False,
    )
    
    for idx, env_id in enumerate(envs):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        env_data = df[df["env_id"] == env_id]
        
        for algo in ["PPO (Standard)", "PPO (Correct Truncation)"]:
            algo_data = env_data[env_data["algorithm"] == algo]
            if len(algo_data) == 0:
                continue
            
            grouped = algo_data.groupby("step")["episodic_return_smooth"].agg(["mean", "std"])
            
            ax.plot(
                grouped.index,
                grouped["mean"],
                color=colors[algo],
                label=algo,
                linewidth=1.5,
            )
            ax.fill_between(
                grouped.index,
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                color=colors[algo],
                alpha=0.2,
            )
        
        ax.set_title(env_id)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.set_xlabel("Steps")
        ax.set_ylabel("Return")
    
    # Hide empty subplots
    for idx in range(n_envs, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def compute_statistics(df):
    """Compute final performance statistics for comparison."""
    stats = []
    
    for env_id in df["env_id"].unique():
        env_data = df[df["env_id"] == env_id]
        
        # Use last 10% of steps for final performance
        max_step = env_data["step"].max()
        final_data = env_data[env_data["step"] >= 0.9 * max_step]
        
        for algo in final_data["algorithm"].unique():
            algo_data = final_data[final_data["algorithm"] == algo]
            
            # Compute mean final return per seed, then aggregate
            seed_means = algo_data.groupby("seed")["episodic_return"].mean()
            
            stats.append({
                "Environment": env_id,
                "Algorithm": algo,
                "Mean Return": seed_means.mean(),
                "Std Return": seed_means.std(),
                "N Seeds": len(seed_means),
            })
    
    return pd.DataFrame(stats)


def plot_bar_comparison(stats_df, colors, args):
    """Generate bar plot comparing final performance."""
    envs = sorted(stats_df["Environment"].unique())
    
    fig, ax = plt.subplots(figsize=(max(8, len(envs) * 2), 5))
    
    x = np.arange(len(envs))
    width = 0.35
    
    for i, algo in enumerate(["PPO (Standard)", "PPO (Correct Truncation)"]):
        algo_stats = stats_df[stats_df["Algorithm"] == algo].set_index("Environment")
        means = [algo_stats.loc[env, "Mean Return"] if env in algo_stats.index else 0 for env in envs]
        stds = [algo_stats.loc[env, "Std Return"] if env in algo_stats.index else 0 for env in envs]
        
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=algo,
            color=colors[algo],
            yerr=stds,
            capsize=5,
        )
    
    ax.set_xlabel("Environment")
    ax.set_ylabel("Final Episodic Return")
    ax.set_title("Final Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    return fig


def generate_latex_table(stats_df):
    """Generate a LaTeX table of results."""
    # Pivot the data for a nice table format
    pivot_data = []
    
    for env in sorted(stats_df["Environment"].unique()):
        row = {"Environment": env}
        env_data = stats_df[stats_df["Environment"] == env]
        
        for algo in ["PPO (Standard)", "PPO (Correct Truncation)"]:
            algo_data = env_data[env_data["Algorithm"] == algo]
            if len(algo_data) > 0:
                mean = algo_data["Mean Return"].values[0]
                std = algo_data["Std Return"].values[0]
                row[algo] = f"${mean:.1f} \\pm {std:.1f}$"
            else:
                row[algo] = "-"
        
        pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Generate LaTeX
    latex = pivot_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(pivot_df.columns) - 1),
    )
    
    return latex


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading TensorBoard logs from: {args.logdir}")
    data = load_tensorboard_logs(args.logdir)
    
    if not data:
        print("No data found. Make sure you've run the experiments first.")
        return
    
    print(f"\nFound data for environments: {list(data.keys())}")
    
    # Prepare data
    print("\nInterpolating data to common steps...")
    df = interpolate_to_common_steps(data)
    
    print(f"Applying smoothing (weight={args.smoothing})...")
    df = apply_smoothing(df, args.smoothing)
    
    # Set up plotting style
    colors = setup_style(args)
    
    # Generate individual learning curve plots
    print("\nGenerating plots...")
    for env_id in df["env_id"].unique():
        fig = plot_learning_curves(df, env_id, colors, args)
        filename = output_dir / f"learning_curve_{env_id}.{args.format}"
        fig.savefig(filename, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filename}")
    
    # Generate combined grid plot
    fig = plot_combined_grid(df, colors, args)
    if fig:
        filename = output_dir / f"learning_curves_combined.{args.format}"
        fig.savefig(filename, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filename}")
    
    # Compute and display statistics
    print("\nComputing final performance statistics...")
    stats_df = compute_statistics(df)
    
    # Generate bar plot
    fig = plot_bar_comparison(stats_df, colors, args)
    filename = output_dir / f"performance_comparison.{args.format}"
    fig.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")
    
    # Save statistics
    stats_filename = output_dir / "statistics.csv"
    stats_df.to_csv(stats_filename, index=False)
    print(f"  Saved: {stats_filename}")
    
    # Generate and save LaTeX table
    latex_table = generate_latex_table(stats_df)
    latex_filename = output_dir / "results_table.tex"
    with open(latex_filename, "w") as f:
        f.write(latex_table)
    print(f"  Saved: {latex_filename}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nLaTeX table for paper:")
    print(latex_table)


if __name__ == "__main__":
    main()
