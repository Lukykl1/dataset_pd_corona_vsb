#!/usr/bin/env python3
"""
correlation_analysis.py

Usage:
    python correlation_analysis.py ml_features.csv

This script reads the CSV of ML features, computes the correlation matrix,
and prints (or saves) highly correlated pairs.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Perform correlation analysis on ML features.")
    parser.add_argument("csv_file", help="Path to the CSV file containing features (e.g., ml_features.csv)")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)
    # Remove non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    corr_matrix = df_numeric.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    # Identify highly correlated pairs (absolute correlation > 0.9)
    threshold = 0.9
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    if correlated_pairs:
        print("Highly correlated feature pairs (|corr| > 0.9):")
        for pair in correlated_pairs:
            print(pair)
    else:
        print("No highly correlated feature pairs found.")

if __name__ == "__main__":
    main()
