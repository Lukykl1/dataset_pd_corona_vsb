#!/usr/bin/env python3
"""
feature_visualization.py

Usage:
    python feature_visualization.py <ml_features_csv> [--output_dir <dir>]

Example:
    python feature_visualization.py ml_features.csv --output_dir ./output_plots

This script reads the ml_features.csv file and generates box plots and
violin plots for key numerical features, grouped by fault_type.
The plots are saved to the specified output directory.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize feature distributions from ml_features.csv.")
    parser.add_argument("ml_features_csv", help="Path to the ml_features.csv file.")
    parser.add_argument("--output_dir", default="output_plots",
                        help="Directory to save the generated plots (default: output_plots).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    try:
        df = pd.read_csv(args.ml_features_csv)
    except FileNotFoundError:
        print(f"Error: ML features file not found at {args.ml_features_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.ml_features_csv}: {e}")
        return

    # Identify potential numerical feature columns for plotting
    # Exclude identifiers like 'file', 'channel', 'sample' and the grouping column 'fault_type'
    potential_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and \
                          col not in ['file', 'channel', 'sample', 'fault_type']]
    
    if not potential_features:
        print("No numerical features found to plot.")
        return
    
    print(f"Found features to plot: {potential_features}")

    for feature in potential_features:
        plt.figure(figsize=(12, 8)) # Adjusted for better readability with more classes
        
        # Box Plot
        plt.subplot(2, 1, 1)
        sns.boxplot(x='fault_type', y=feature, data=df, palette="Set2")
        plt.title(f'Box Plot of {feature} by Fault Type')
        plt.xticks(rotation=45, ha="right") # Rotate labels for readability
        plt.tight_layout() # Adjust layout to prevent label overlap

        # Violin Plot
        plt.subplot(2, 1, 2)
        sns.violinplot(x='fault_type', y=feature, data=df, palette="Set2")
        plt.title(f'Violin Plot of {feature} by Fault Type')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_filename = os.path.join(args.output_dir, f'{feature}_distribution_plots.png')
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()

    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()