#!/usr/bin/env python3
"""
feature_visualization.py

Usage:
    python feature_visualization.py <ml_features_csv> [--output_dir <dir>]

Example:
    python feature_visualization.py ml_features.csv --output_dir ./output_plots

This script reads the ml_features.csv file and generates box plots and
violin plots for key numerical features, grouped by fault_type,
separately for each antenna channel.
The plots are saved to the specified output directory.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize feature distributions from ml_features.csv, per channel.")
    parser.add_argument("ml_features_csv", help="Path to the ml_features.csv file.")
    parser.add_argument("--output_dir", default="output_plots",
                        help="Directory to save the generated plots (default: output_plots).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    try:
        df_all_channels = pd.read_csv(args.ml_features_csv)
    except FileNotFoundError:
        print(f"Error: ML features file not found at {args.ml_features_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.ml_features_csv}: {e}")
        return

    if 'channel' not in df_all_channels.columns:
        print("Error: 'channel' column not found in ml_features.csv. Cannot perform per-channel analysis.")
        return

    potential_features = [col for col in df_all_channels.columns if df_all_channels[col].dtype in ['int64', 'float64'] and \
                          col not in ['file', 'channel', 'sample', 'fault_type']]
    
    if not potential_features:
        print("No numerical features found to plot.")
        return
    
    unique_channels = df_all_channels['channel'].unique()
    print(f"Found channels: {unique_channels}")
    print(f"Found features to plot: {potential_features}")

    for channel_name in unique_channels:
        print(f"\nProcessing Channel: {channel_name}")
        df_channel = df_all_channels[df_all_channels['channel'] == channel_name].copy()

        if df_channel.empty:
            print(f"No data found for channel {channel_name}. Skipping.")
            continue
        
        # Create a subdirectory for each channel's plots for better organization
        channel_output_dir = os.path.join(args.output_dir, f"channel_{channel_name}")
        if not os.path.exists(channel_output_dir):
            os.makedirs(channel_output_dir)
            print(f"Created output directory for channel {channel_name}: {channel_output_dir}")

        for feature in potential_features:
            if df_channel[feature].isnull().all(): # Skip if all values for this feature in this channel are NaN
                print(f"  Skipping feature '{feature}' for channel {channel_name} as all values are NaN.")
                continue
            if df_channel[feature].nunique() < 2 and df_channel['fault_type'].nunique() <2: # Skip if not enough variation for a meaningful plot
                print(f"  Skipping feature '{feature}' for channel {channel_name} due to insufficient unique values for plotting.")
                continue


            plt.figure(figsize=(14, 10)) # Adjusted for better readability
            
            # Box Plot
            plt.subplot(2, 1, 1)
            sns.boxplot(x='fault_type', y=feature, data=df_channel, palette="Set2")
            plt.title(f'Channel {channel_name} - Box Plot of {feature} by Fault Type')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

            # Violin Plot
            plt.subplot(2, 1, 2)
            sns.violinplot(x='fault_type', y=feature, data=df_channel, palette="Set2")
            plt.title(f'Channel {channel_name} - Violin Plot of {feature} by Fault Type')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            plot_filename = os.path.join(channel_output_dir, f'{channel_name}_{feature}_distribution_plots.png')
            try:
                plt.savefig(plot_filename)
                print(f"  Saved plot: {plot_filename}")
            except Exception as e:
                print(f"  Error saving plot {plot_filename}: {e}")
            plt.close()

    print(f"\nAll per-channel feature plots saved to subdirectories in {args.output_dir}")

if __name__ == "__main__":
    main()