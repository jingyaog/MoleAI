#!/usr/bin/env python3
"""
Plot training metrics from train_detector_cnn.py
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(metrics_path, output_dir="plots"):
    """
    Generate training visualization plots from metrics JSON.

    Args:
        metrics_path: Path to training_metrics_{backbone}.json
        output_dir: Directory to save plots
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    epochs = list(range(1, len(metrics["train_loss"]) + 1))
    train_loss = metrics["train_loss"]
    train_acc = metrics["train_acc"]
    test_acc = metrics["test_acc"]

    # Extract backbone name from filename
    backbone = metrics_path.split("_")[-1].replace(".json", "")

    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Training Metrics', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    axes[0].plot(epochs, train_loss, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    # Add value annotations
    for i, (x, y) in enumerate(zip(epochs, train_loss)):
        axes[0].annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)

    # Plot 2: Training Accuracy
    axes[1].plot(epochs, train_acc, 'g-o', linewidth=2, markersize=8, label='Train Acc')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)
    axes[1].set_ylim([0, 105])

    # Add value annotations
    for i, (x, y) in enumerate(zip(epochs, train_acc)):
        axes[1].annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)

    # Plot 3: Test Accuracy
    axes[2].plot(epochs, test_acc, 'r-o', linewidth=2, markersize=8, label='Test Acc')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(epochs)
    axes[2].set_ylim([0, 105])

    # Add value annotations
    for i, (x, y) in enumerate(zip(epochs, test_acc)):
        axes[2].annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)

    # Add best accuracy line
    best_acc = max(test_acc)
    axes[2].axhline(y=best_acc, color='r', linestyle='--', alpha=0.5,
                   label=f'Best: {best_acc:.1f}%')
    axes[2].legend()

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'training_metrics_{backbone}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    # Create a combined accuracy plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, 'g-o', linewidth=2, markersize=8, label='Train Accuracy')
    ax.plot(epochs, test_acc, 'r-o', linewidth=2, markersize=8, label='Test Accuracy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Train vs Test Accuracy - {backbone.upper()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=12)

    plt.tight_layout()

    output_path2 = os.path.join(output_dir, f'accuracy_comparison_{backbone}.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path2}")

    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Training Summary for {backbone.upper()}")
    print(f"{'='*50}")
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nFinal Metrics:")
    print(f"  Train Loss:     {train_loss[-1]:.4f}")
    print(f"  Train Accuracy: {train_acc[-1]:.2f}%")
    print(f"  Test Accuracy:  {test_acc[-1]:.2f}%")
    print(f"\nBest Test Accuracy: {best_acc:.2f}% (Epoch {test_acc.index(best_acc) + 1})")
    print(f"{'='*50}\n")

    # Show plots if not in headless environment
    try:
        plt.show()
    except:
        print("(Running in headless mode - plots saved to disk)")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--metrics_path', type=str, default='training_metrics_resnet18.json',
                       help='Path to training metrics JSON file')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    args = parser.parse_args()

    if not os.path.exists(args.metrics_path):
        print(f"Error: Metrics file not found: {args.metrics_path}")
        print("\nAvailable metrics files:")
        for f in os.listdir('.'):
            if f.startswith('training_metrics_') and f.endswith('.json'):
                print(f"  - {f}")
        return

    plot_metrics(args.metrics_path, args.output_dir)


if __name__ == '__main__':
    main()
