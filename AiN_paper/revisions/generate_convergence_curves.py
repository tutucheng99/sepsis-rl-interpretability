"""
Generate Training Convergence Curves for Reviewer 2 Comment 3
This script creates publication-quality convergence curves for all 6 RL algorithms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Set up paths
PROJECT_ROOT = Path(r"C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1")
OUTPUT_DIR = PROJECT_ROOT / "AiN_paper" / "revisions" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality plot style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_loss_data():
    """Load all available loss data from d3rlpy logs."""
    data = {}

    # BC (Offline)
    bc_path = PROJECT_ROOT / "d3rlpy_logs" / "DiscreteBC_20251014183017" / "loss.csv"
    if bc_path.exists():
        df = pd.read_csv(bc_path, header=None, names=['epoch', 'step', 'loss'])
        data['BC'] = {'steps': df['step'].values, 'loss': df['loss'].values, 'type': 'offline'}
        print(f"Loaded BC: {len(df)} points, loss range: {df['loss'].min():.4f} - {df['loss'].max():.4f}")

    # CQL (Offline)
    cql_path = PROJECT_ROOT / "d3rlpy_logs" / "DiscreteCQL_20251014191912" / "loss.csv"
    if cql_path.exists():
        df = pd.read_csv(cql_path, header=None, names=['epoch', 'step', 'loss'])
        data['CQL'] = {'steps': df['step'].values, 'loss': df['loss'].values, 'type': 'offline'}
        print(f"Loaded CQL: {len(df)} points, loss range: {df['loss'].min():.4f} - {df['loss'].max():.4f}")

    # DQN/DDQN variants (Online/Hybrid) - find the best run
    dqn_logs_dir = PROJECT_ROOT / "github_models" / "d3rlpy_logs"

    # Find DoubleDQN runs
    dqn_runs = list(dqn_logs_dir.glob("DoubleDQN_online_*/loss.csv"))
    if dqn_runs:
        # Use the most recent run
        latest_dqn = sorted(dqn_runs)[-1]
        df = pd.read_csv(latest_dqn, header=None, names=['epoch', 'step', 'loss'])
        data['DQN'] = {'steps': df['step'].values, 'loss': df['loss'].values, 'type': 'hybrid'}
        print(f"Loaded DQN: {len(df)} points from {latest_dqn.parent.name}")

    # SAC (Online/Hybrid) - has actor and critic loss
    sac_runs = list(dqn_logs_dir.glob("DiscreteSAC_online_*/critic_loss.csv"))
    if sac_runs:
        latest_sac_dir = sorted([r.parent for r in sac_runs])[-1]

        # Load critic loss (main training signal)
        critic_path = latest_sac_dir / "critic_loss.csv"
        if critic_path.exists():
            df = pd.read_csv(critic_path, header=None, names=['epoch', 'step', 'loss'])
            data['SAC'] = {'steps': df['step'].values, 'loss': df['loss'].values, 'type': 'hybrid'}
            print(f"Loaded SAC: {len(df)} points from {latest_sac_dir.name}")

    return data


def plot_individual_curves(data):
    """Create individual convergence plots for each algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    colors = {
        'BC': '#2ecc71',      # Green (offline)
        'CQL': '#27ae60',     # Dark green (offline)
        'DQN': '#3498db',     # Blue (hybrid)
        'SAC': '#9b59b6',     # Purple (hybrid)
    }

    for idx, (name, info) in enumerate(data.items()):
        if idx >= 4:
            break
        ax = axes[idx]

        steps = info['steps'] / 1000  # Convert to thousands
        loss = info['loss']

        # Plot with smoothing
        ax.plot(steps, loss, color=colors.get(name, 'gray'), linewidth=1.5, alpha=0.3, label='Raw')

        # Add smoothed line (moving average)
        if len(loss) > 5:
            window = min(5, len(loss) // 3)
            smoothed = pd.Series(loss).rolling(window=window, center=True).mean()
            ax.plot(steps, smoothed, color=colors.get(name, 'gray'), linewidth=2, label='Smoothed')

        ax.set_xlabel('Training Steps (×1000)')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} Training Convergence')
        ax.grid(True, alpha=0.3)

        # Add paradigm label
        paradigm = 'Offline' if info['type'] == 'offline' else 'Hybrid'
        ax.text(0.95, 0.95, paradigm, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_convergence_individual.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: training_convergence_individual.png")
    plt.close()


def plot_combined_curves(data):
    """Create a combined convergence plot with all algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'BC': '#2ecc71',
        'CQL': '#27ae60',
        'DQN': '#3498db',
        'SAC': '#9b59b6',
    }

    linestyles = {
        'offline': '-',
        'hybrid': '--',
    }

    for name, info in data.items():
        steps = info['steps'] / 1000
        loss = info['loss']

        # Normalize loss for comparison (min-max scaling)
        loss_norm = (loss - loss.min()) / (loss.max() - loss.min() + 1e-8)

        # Smoothed line
        if len(loss_norm) > 5:
            window = min(5, len(loss_norm) // 3)
            smoothed = pd.Series(loss_norm).rolling(window=window, center=True).mean()
        else:
            smoothed = loss_norm

        label = f"{name} ({'Offline' if info['type'] == 'offline' else 'Hybrid'})"
        ax.plot(steps, smoothed, color=colors.get(name, 'gray'),
                linestyle=linestyles[info['type']], linewidth=2, label=label)

    ax.set_xlabel('Training Steps (×1000)')
    ax.set_ylabel('Normalized Loss')
    ax.set_title('Training Convergence Comparison Across RL Algorithms')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = OUTPUT_DIR / "training_convergence_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: training_convergence_combined.png")
    plt.close()


def plot_publication_figure(data):
    """Create publication-ready figure with subplots."""
    fig = plt.figure(figsize=(12, 10))

    # Create grid: 2 rows, 3 columns
    # Top row: BC, CQL, DQN
    # Bottom row: SAC, Combined comparison

    colors = {
        'BC': '#2ecc71',
        'CQL': '#27ae60',
        'DQN': '#3498db',
        'SAC': '#9b59b6',
    }

    algorithm_order = ['BC', 'CQL', 'DQN', 'SAC']

    # Individual plots
    for idx, name in enumerate(algorithm_order):
        if name not in data:
            continue

        if idx < 3:
            ax = fig.add_subplot(2, 3, idx + 1)
        else:
            ax = fig.add_subplot(2, 3, 4)

        info = data[name]
        steps = info['steps'] / 1000
        loss = info['loss']

        # Raw data (light)
        ax.plot(steps, loss, color=colors[name], linewidth=1, alpha=0.3)

        # Smoothed (bold)
        if len(loss) > 5:
            window = min(5, len(loss) // 3)
            smoothed = pd.Series(loss).rolling(window=window, center=True).mean()
            ax.plot(steps, smoothed, color=colors[name], linewidth=2.5)

        ax.set_xlabel('Steps (×1000)')
        ax.set_ylabel('Loss')

        paradigm = 'Offline' if info['type'] == 'offline' else 'Hybrid'
        ax.set_title(f'({chr(97+idx)}) {name} [{paradigm}]')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add final loss annotation
        final_loss = loss[-1] if len(loss) > 0 else 0
        ax.annotate(f'Final: {final_loss:.4f}',
                   xy=(steps[-1], loss[-1]),
                   xytext=(-50, 10), textcoords='offset points',
                   fontsize=8, color=colors[name])

    # Combined normalized comparison
    ax_combined = fig.add_subplot(2, 3, (5, 6))

    for name in algorithm_order:
        if name not in data:
            continue
        info = data[name]
        steps = info['steps'] / 1000
        loss = info['loss']

        # Normalize
        loss_norm = (loss - loss.min()) / (loss.max() - loss.min() + 1e-8)

        if len(loss_norm) > 5:
            window = min(5, len(loss_norm) // 3)
            smoothed = pd.Series(loss_norm).rolling(window=window, center=True).mean()
        else:
            smoothed = loss_norm

        paradigm = 'Offline' if info['type'] == 'offline' else 'Hybrid'
        linestyle = '-' if info['type'] == 'offline' else '--'
        ax_combined.plot(steps, smoothed, color=colors[name],
                        linestyle=linestyle, linewidth=2,
                        label=f'{name} ({paradigm})')

    ax_combined.set_xlabel('Training Steps (×1000)')
    ax_combined.set_ylabel('Normalized Loss (0-1)')
    ax_combined.set_title('(e) Normalized Loss Comparison')
    ax_combined.legend(loc='upper right', framealpha=0.9)
    ax_combined.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "Figure_training_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Saved publication figure: Figure_training_convergence.png")

    # Also save as PDF for LaTeX
    pdf_path = OUTPUT_DIR / "Figure_training_convergence.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print("Saved PDF: Figure_training_convergence.pdf")

    plt.close()


def main():
    print("=" * 60)
    print("Generating Training Convergence Curves")
    print("=" * 60)

    # Load data
    data = load_loss_data()

    if not data:
        print("ERROR: No loss data found!")
        return

    print(f"\nLoaded {len(data)} algorithms: {list(data.keys())}")

    # Generate plots
    print("\nGenerating plots...")
    plot_individual_curves(data)
    plot_combined_curves(data)
    plot_publication_figure(data)

    print("\n" + "=" * 60)
    print("DONE! Figures saved to: AiN_paper/revisions/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
