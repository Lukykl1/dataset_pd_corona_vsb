#!/usr/bin/env python3
"""
dimensionality_reduction_analyzer.py

Usage:
    python dimensionality_reduction_analyzer.py <ml_features_csv> [--output_dir <dir>]

Example:
    python dimensionality_reduction_analyzer.py ml_features.csv --output_dir ./output_plots

This script applies PCA and t-SNE to the features in ml_features.csv
and generates 2D scatter plots, colored by fault_type.
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
    parser = argparse.ArgumentParser(description="Perform and visualize dimensionality reduction.")
    parser.add_argument("ml_features_csv", help="Path to the ml_features.csv file.")
    parser.add_argument("--output_dir", default="output_plots",
                        help="Directory to save the generated plots (default: output_plots).")
    parser.add_argument("--perplexity_tsne", type=float, default=30.0,
                        help="Perplexity for t-SNE (default: 30.0).")
    parser.add_argument("--n_iter_tsne", type=int, default=1000,
                        help="Number of iterations for t-SNE (default: 1000).")
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

    # Select numerical features for dimensionality reduction
    # Exclude identifiers and non-numeric columns
    feature_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    if not feature_cols:
        print("No numerical features found for dimensionality reduction.")
        return
    
    X = df[feature_cols].copy()
    # Handle NaNs by filling with mean. More sophisticated imputation could be used.
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Warning: Feature '{col}' contains NaNs. Filling with mean.")
            X[col].fillna(X[col].mean(), inplace=True)
            
    # Handle cases where a feature might be constant after NaN filling or by nature
    if X.empty or X.shape[1] == 0:
        print("No valid features remaining after pre-processing.")
        return

    # Check for constant features which cause issues with StandardScaler
    cols_to_drop_due_to_low_variance = [col for col in X.columns if X[col].nunique() <= 1]
    if cols_to_drop_due_to_low_variance:
        print(f"Warning: Dropping constant/low-variance features: {cols_to_drop_due_to_low_variance}")
        X.drop(columns=cols_to_drop_due_to_low_variance, inplace=True)

    if X.empty or X.shape[1] == 0:
        print("No valid features remaining after dropping low-variance columns.")
        return
        
    y_labels = df['fault_type'] # Target labels for coloring

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA Visualization ---
    print("Performing PCA...")
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="viridis", s=50, alpha=0.7)
        plt.title('2D PCA of Features by Fault Type')
        plt.xlabel(f'Principal Component 1 (Explains {pca.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'Principal Component 2 (Explains {pca.explained_variance_ratio_[1]*100:.2f}%)')
        plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        pca_plot_filename = os.path.join(args.output_dir, 'pca_2d_visualization.png')
        plt.savefig(pca_plot_filename)
        print(f"Saved PCA plot: {pca_plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error during PCA: {e}")


    # --- t-SNE Visualization ---
    # t-SNE can be slow on large datasets. Consider subsampling if necessary.
    # Number of features for t-SNE should be less than n_samples - 1
    if X_scaled.shape[0] <= X_scaled.shape[1]:
        print(f"Warning: Number of samples ({X_scaled.shape[0]}) is not greater than number of features ({X_scaled.shape[1]}). Skipping t-SNE or using PCA pre-reduction.")
        # Option: Reduce dimensionality with PCA first if many features
        if X_scaled.shape[1] > 50: # Arbitrary threshold
            print("Applying PCA to reduce features before t-SNE.")
            pca_tsne = PCA(n_components=min(50, X_scaled.shape[0]-1)) # Ensure n_components < n_samples
            X_for_tsne = pca_tsne.fit_transform(X_scaled)
        else:
            X_for_tsne = X_scaled
    else:
        X_for_tsne = X_scaled

    if X_for_tsne.shape[0] > 1 : # t-SNE requires at least 2 samples
        print("Performing t-SNE (this may take a while)...")
        # Adjust perplexity if number of samples is small
        current_perplexity = min(args.perplexity_tsne, X_for_tsne.shape[0] - 1)
        if current_perplexity < 5: # t-SNE might not work well with very low perplexity
             print(f"Warning: Perplexity ({current_perplexity}) is very low. t-SNE results might be suboptimal.")
        
        if current_perplexity > 0:
            try:
                tsne = TSNE(n_components=2, perplexity=current_perplexity, n_iter=args.n_iter_tsne, random_state=42)
                X_tsne = tsne.fit_transform(X_for_tsne)
                
                plt.figure(figsize=(10, 7))
                sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, palette="viridis", s=50, alpha=0.7)
                plt.title(f'2D t-SNE of Features by Fault Type (Perplexity: {current_perplexity})')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                tsne_plot_filename = os.path.join(args.output_dir, 'tsne_2d_visualization.png')
                plt.savefig(tsne_plot_filename)
                print(f"Saved t-SNE plot: {tsne_plot_filename}")
                plt.close()
            except Exception as e:
                print(f"Error during t-SNE: {e}")
        else:
            print("Skipping t-SNE due to insufficient samples for perplexity calculation.")
    else:
        print("Skipping t-SNE due to insufficient samples.")


    print(f"\nDimensionality reduction analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()