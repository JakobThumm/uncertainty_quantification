#!/usr/bin/env python3
"""
Plot OOD scores for H36M (ID) vs Tiger Pose (OOD) datasets.

This script loads the computed OOD scores from a cloudpickle file and creates
histograms comparing the ID and OOD distributions.
"""

import argparse
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_ood_histogram(scores_dict, save_path=None, bins=50, alpha=0.6):
    """
    Plot histogram of OOD scores with ID and OOD data in different colors.

    Args:
        scores_dict: Dictionary containing scores for different distributions
        save_path: Path to save the plot (None = display only)
        bins: Number of histogram bins
        alpha: Transparency of histogram bars
    """
    # Extract ID and OOD scores
    id_scores = None
    ood_scores = []
    ood_names = []

    for key, value in scores_dict.items():
        if key in ['eigenvals', 'args_dict', 'score_fun']:
            continue
        if '_QF' in key:  # Skip quadratic form scores
            continue

        if key == 'ID':
            id_scores = np.array(value)
        else:
            ood_scores.append(np.array(value))
            ood_names.append(key)

    if id_scores is None:
        raise ValueError("No ID scores found in scores_dict")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot ID distribution
    ax.hist(id_scores, bins=bins, alpha=alpha, color='blue',
            label=f'ID (H36M) - mean: {id_scores.mean():.4f}',
            density=True, edgecolor='black', linewidth=0.5)

    # Plot OOD distributions
    colors = ['red', 'orange', 'purple', 'green', 'brown']
    for i, (scores, name) in enumerate(zip(ood_scores, ood_names)):
        color = colors[i % len(colors)]
        ax.hist(scores, bins=bins, alpha=alpha, color=color,
                label=f'OOD ({name}) - mean: {scores.mean():.4f}',
                density=True, edgecolor='black', linewidth=0.5)

    # Add vertical lines for means
    ax.axvline(id_scores.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    for i, scores in enumerate(ood_scores):
        color = colors[i % len(colors)]
        ax.axvline(scores.mean(), color=color, linestyle='--', linewidth=2, alpha=0.8)

    ax.set_xlabel('OOD Score (Uncertainty)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('OOD Score Distribution: ID vs OOD', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_ood_comparison(scores_dict, save_path=None):
    """
    Create multiple comparison plots: histogram, box plot, and statistics.

    Args:
        scores_dict: Dictionary containing scores for different distributions
        save_path: Path to save the plot (None = display only)
    """
    # Extract ID and OOD scores
    id_scores = None
    ood_scores = []
    ood_names = []

    for key, value in scores_dict.items():
        if key in ['eigenvals', 'args_dict', 'score_fun']:
            continue
        if '_QF' in key:
            continue

        if key == 'ID':
            id_scores = np.array(value)
        else:
            ood_scores.append(np.array(value))
            ood_names.append(key)

    if id_scores is None:
        raise ValueError("No ID scores found in scores_dict")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Histogram
    ax1 = plt.subplot(2, 2, 1)
    bins = 50
    alpha = 0.6

    ax1.hist(id_scores, bins=bins, alpha=alpha, color='blue',
             label=f'ID (H36M)', density=True, edgecolor='black', linewidth=0.5)

    colors = ['red', 'orange', 'purple', 'green', 'brown']
    for i, (scores, name) in enumerate(zip(ood_scores, ood_names)):
        color = colors[i % len(colors)]
        ax1.hist(scores, bins=bins, alpha=alpha, color=color,
                 label=f'OOD ({name})', density=True, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('OOD Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Score Distribution (Histogram)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = plt.subplot(2, 2, 2)
    all_scores = [id_scores] + ood_scores
    all_labels = ['ID (H36M)'] + [f'OOD ({name})' for name in ood_names]
    all_colors = ['blue'] + [colors[i % len(colors)] for i in range(len(ood_scores))]

    bp = ax2.boxplot(all_scores, labels=all_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('OOD Score', fontsize=12)
    ax2.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # 3. CDF (Cumulative Distribution Function)
    ax3 = plt.subplot(2, 2, 3)

    sorted_id = np.sort(id_scores)
    cdf_id = np.arange(1, len(sorted_id) + 1) / len(sorted_id)
    ax3.plot(sorted_id, cdf_id, color='blue', linewidth=2, label='ID (H36M)')

    for i, (scores, name) in enumerate(zip(ood_scores, ood_names)):
        color = colors[i % len(colors)]
        sorted_scores = np.sort(scores)
        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax3.plot(sorted_scores, cdf, color=color, linewidth=2, label=f'OOD ({name})')

    ax3.set_xlabel('OOD Score', fontsize=12)
    ax3.set_ylabel('Cumulative Probability', fontsize=12)
    ax3.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    stats_data = []
    stats_data.append(['Distribution', 'Mean', 'Std', 'Min', 'Max', 'Median'])

    id_stats = [
        'ID (H36M)',
        f'{id_scores.mean():.6f}',
        f'{id_scores.std():.6f}',
        f'{id_scores.min():.6f}',
        f'{id_scores.max():.6f}',
        f'{np.median(id_scores):.6f}'
    ]
    stats_data.append(id_stats)

    for name, scores in zip(ood_names, ood_scores):
        ood_stats = [
            f'OOD ({name})',
            f'{scores.mean():.6f}',
            f'{scores.std():.6f}',
            f'{scores.min():.6f}',
            f'{scores.max():.6f}',
            f'{np.median(scores):.6f}'
        ]
        stats_data.append(ood_stats)

    table = ax4.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color ID row
    for i in range(len(stats_data[0])):
        table[(1, i)].set_facecolor('#e6f2ff')

    # Color OOD rows
    for row_idx in range(2, len(stats_data)):
        for i in range(len(stats_data[0])):
            table[(row_idx, i)].set_facecolor('#ffe6e6')

    ax4.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('OOD Detection: H36M (ID) vs Tiger Pose (OOD)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive plot to {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot OOD scores from cloudpickle file')
    parser.add_argument('--scores_file', type=str, required=True,
                       help='Path to cloudpickle file containing scores')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the plot (default: display only)')
    parser.add_argument('--plot_type', type=str, default='comprehensive',
                       choices=['histogram', 'comprehensive'],
                       help='Type of plot to create')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of histogram bins')

    args = parser.parse_args()

    # Load scores
    print(f"Loading scores from {args.scores_file}...")
    with open(args.scores_file, 'rb') as f:
        scores_dict = cloudpickle.load(f)

    print(f"Loaded scores for distributions: {[k for k in scores_dict.keys() if k not in ['eigenvals', 'args_dict', 'score_fun']]}")

    # Create plot
    if args.plot_type == 'histogram':
        plot_ood_histogram(scores_dict, save_path=args.save_path, bins=args.bins)
    else:
        plot_ood_comparison(scores_dict, save_path=args.save_path)

    print("Done!")


if __name__ == '__main__':
    main()
