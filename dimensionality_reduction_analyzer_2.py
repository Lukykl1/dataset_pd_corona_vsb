#!/usr/bin/env python3
"""
dimensionality_reduction_analyzer.py

Usage:
    python dimensionality_reduction_analyzer.py <ml_features_csv> [--output_dir <dir>] [--run_tsne]

Example:
    python dimensionality_reduction_analyzer.py ml_features.csv --output_dir ./output_plots --run_tsne

This script applies PCA and optionally t-SNE to the features in ml_features.csv,
separately for each antenna channel.
It generates:
1. A scree plot for PCA for each channel.
2. A 2D PCA scatter plot for each channel.
3. Optionally, a 2D t-SNE scatter plot for each channel.
   WARNING: Interpret t-SNE plots with extreme caution.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Perform and visualize dimensionality reduction per channel.")
    parser.add_argument("ml_features_csv", help="Path to the ml_features.csv file.")
    parser.add_argument("--output_dir", default="output_plots",
                        help="Directory to save the generated plots (default: output_plots).")
    parser.add_argument("--run_tsne", action='store_true',
                        help="Also run and plot t-SNE (can be slow and requires careful interpretation).")
    parser.add_argument("--perplexity_tsne", type=float, default=30.0,
                        help="Perplexity for t-SNE (default: 30.0).")
    parser.add_argument("--n_iter_tsne", type=int, default=1000,
                        help="Number of iterations for t-SNE (default: 1000).")
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

    feature_cols_all = [col for col in df_all_channels.columns if df_all_channels[col].dtype in ['int64', 'float64']]
    if not feature_cols_all:
        print("No numerical features found for dimensionality reduction.")
        return
    
    unique_channels = df_all_channels['channel'].unique()
    print(f"Found channels: {unique_channels}")

    for channel_name in unique_channels:
        print(f"\nProcessing Channel: {channel_name}")
        df_channel = df_all_channels[df_all_channels['channel'] == channel_name].copy()

        if df_channel.empty:
            print(f"No data found for channel {channel_name}. Skipping.")
            continue
        
        # Create a subdirectory for each channel's plots
        channel_output_dir = os.path.join(args.output_dir, f"channel_{channel_name}")
        if not os.path.exists(channel_output_dir):
            os.makedirs(channel_output_dir)
            print(f"Created output directory for channel {channel_name}: {channel_output_dir}")

        # Re-select feature columns based on df_channel, in case some are all NaN for this channel
        feature_cols = [col for col in feature_cols_all if col in df_channel.columns and df_channel[col].dtype in ['int64', 'float64']]
        
        X = df_channel[feature_cols].copy()
        for col in X.columns: # Impute NaNs
            if X[col].isnull().any():
                print(f"  Warning: Channel {channel_name}, Feature '{col}' contains NaNs. Filling with mean for this channel.")
                X[col].fillna(X[col].mean(), inplace=True)
        
        if X.empty or X.shape[1] == 0:
            print(f"  No valid features remaining for channel {channel_name} after pre-processing. Skipping.")
            continue

        cols_to_drop_low_variance = [col for col in X.columns if X[col].nunique() <= 1]
        if cols_to_drop_low_variance:
            print(f"  Warning: Channel {channel_name}, Dropping constant/low-variance features: {cols_to_drop_low_variance}")
            X.drop(columns=cols_to_drop_low_variance, inplace=True)

        if X.empty or X.shape[1] == 0:
            print(f"  No valid features remaining for channel {channel_name} after dropping low-variance columns. Skipping.")
            continue
            
        y_labels = df_channel['fault_type'] 

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- PCA Visualization ---
        print(f"  Performing PCA for Channel {channel_name}...")
        try:
            n_components_pca = min(X_scaled.shape[0], X_scaled.shape[1], 10)
            if n_components_pca < 1:
                print(f"    Not enough samples/features ({X_scaled.shape[0]}/{X_scaled.shape[1]}) for PCA. Skipping PCA for channel {channel_name}.")
                continue # Skip to next channel or t-SNE if applicable
            
            pca_full = PCA(n_components=n_components_pca)
            pca_full.fit(X_scaled)

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_, marker='o', linestyle='--')
            plt.title(f'Channel {channel_name} - PCA Scree Plot')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Proportion of Variance Explained')
            plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))
            plt.grid(True); plt.tight_layout()
            scree_plot_filename = os.path.join(channel_output_dir, f'{channel_name}_pca_scree_plot.png')
            plt.savefig(scree_plot_filename); plt.close()
            print(f"    Saved PCA Scree plot: {scree_plot_filename}")

            if n_components_pca < 2:
                print(f"    Not enough components ({n_components_pca}) for 2D PCA plot. Skipping 2D plot for channel {channel_name}.")
            else:
                pca_2d = PCA(n_components=2) # Re-fit for exactly 2 components if n_components_pca was > 2
                if X_scaled.shape[1] < 2: # If original num features < 2
                     pca_2d = PCA(n_components=X_scaled.shape[1])
                
                X_pca = pca_2d.fit_transform(X_scaled)
                
                plt.figure(figsize=(10, 7))
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="viridis", s=50, alpha=0.7)
                plt.title(f'Channel {channel_name} - 2D PCA of Features (Focus on Overlap)')
                plt.xlabel(f'PC 1 (Explains {pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
                if X_pca.shape[1] > 1:
                    plt.ylabel(f'PC 2 (Explains {pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
                else:
                    plt.ylabel(' (Only 1 PC available)')

                plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True); plt.tight_layout()
                pca_plot_filename = os.path.join(channel_output_dir, f'{channel_name}_pca_2d_visualization.png')
                plt.savefig(pca_plot_filename); plt.close()
                print(f"    Saved 2D PCA plot: {pca_plot_filename}")
                
                cumulative_variance_2_components = np.sum(pca_2d.explained_variance_ratio_) * 100
                print(f"    Cumulative variance explained by first {pca_2d.n_components_} PCs for channel {channel_name}: {cumulative_variance_2_components:.2f}%")

        except Exception as e:
            print(f"    Error during PCA for channel {channel_name}: {e}")

        # --- t-SNE Visualization (Optional) ---
        if args.run_tsne:
            print(f"\n  --- t-SNE Visualization for Channel {channel_name} ---")
            print("  WARNING: t-SNE interpretation caveats apply (see script header).")
            
            X_for_tsne = X_scaled # Start with scaled features for this channel
            if X_for_tsne.shape[0] <= X_for_tsne.shape[1]: # n_samples <= n_features
                if X_for_tsne.shape[1] > 50: 
                    pca_tsne_dims = min(50, X_for_tsne.shape[0]-1)
                    if pca_tsne_dims <=1: X_for_tsne = None
                    else:
                        pca_for_tsne = PCA(n_components=pca_tsne_dims); X_for_tsne = pca_for_tsne.fit_transform(X_for_tsne)
                elif X_for_tsne.shape[0]-1 <=1: X_for_tsne = None
            
            if X_for_tsne is not None and X_for_tsne.shape[0] > 1 :
                current_perplexity = min(args.perplexity_tsne, X_for_tsne.shape[0] - 1)
                if current_perplexity <= 0: print(f"    Perplexity ({current_perplexity}) must be positive. Skipping t-SNE for channel {channel_name}.")
                else:
                    if current_perplexity < 5: print(f"    Warning: Perplexity ({current_perplexity}) is very low for channel {channel_name}. t-SNE results might be suboptimal.")
                    print(f"    Performing t-SNE for channel {channel_name} (this may take a while)...")
                    try:
                        tsne = TSNE(n_components=2, perplexity=current_perplexity, n_iter=args.n_iter_tsne, random_state=42, learning_rate='auto', init='pca')
                        X_tsne = tsne.fit_transform(X_for_tsne)
                        
                        plt.figure(figsize=(10, 7))
                        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, palette="viridis", s=50, alpha=0.7)
                        plt.title(f'Channel {channel_name} - 2D t-SNE (Interpret Cautiously!) - Perp: {current_perplexity}')
                        plt.xlabel('t-SNE Component 1'); plt.ylabel('t-SNE Component 2')
                        plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(True); plt.tight_layout()
                        tsne_plot_filename = os.path.join(channel_output_dir, f'{channel_name}_tsne_2d_visualization.png')
                        plt.savefig(tsne_plot_filename); plt.close()
                        print(f"    Saved t-SNE plot: {tsne_plot_filename}")
                    except Exception as e:
                        print(f"    Error during t-SNE for channel {channel_name}: {e}")
            else:
                print(f"    Skipping t-SNE for channel {channel_name} due to insufficient samples or pre-processing issues.")
        else:
            print(f"\n  Skipping t-SNE analysis for channel {channel_name} as per user request.")

    print(f"\nAll per-channel dimensionality reduction analyses complete. Plots saved to subdirectories in {args.output_dir}")

if __name__ == "__main__":
    main()