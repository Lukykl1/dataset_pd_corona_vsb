#!/usr/bin/env python3
"""
quantitative_validation_report.py

Usage:
    python quantitative_validation_report.py <ml_features_csv> [--output_file <file>]

Example:
    python quantitative_validation_report.py ml_features.csv --output_file ./output_reports/validation_report.txt

This script computes quantitative metrics from ml_features.csv to assess
dataset quality, such as intra-class consistency and inter-class separability.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import os

def fisher_discriminant_ratio(series1, series2):
    """Calculates Fisher's discriminant ratio between two series."""
    if len(series1) < 2 or len(series2) < 2: # Need at least 2 samples for variance
        return np.nan
    mean1, mean2 = np.mean(series1), np.mean(series2)
    std1, std2 = np.std(series1), np.std(series2)
    if std1**2 + std2**2 == 0: # Avoid division by zero if variances are zero
        return np.inf if (mean1 - mean2)**2 > 0 else 0
    return (mean1 - mean2)**2 / (std1**2 + std2**2)

def main():
    parser = argparse.ArgumentParser(description="Generate a quantitative validation report.")
    parser.add_argument("ml_features_csv", help="Path to the ml_features.csv file.")
    parser.add_argument("--output_file", default=None,
                        help="Optional path to save the report (default: print to console).")
    args = parser.parse_args()

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory for report: {output_dir}")

    try:
        df = pd.read_csv(args.ml_features_csv)
    except FileNotFoundError:
        print(f"Error: ML features file not found at {args.ml_features_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.ml_features_csv}: {e}")
        return

    report_lines = ["Quantitative Validation Report\n", "="*30 + "\n"]

    # Identify numerical features
    feature_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    if not feature_cols:
        report_lines.append("No numerical features found for analysis.\n")
        print("".join(report_lines))
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write("".join(report_lines))
        return
        
    # --- Intra-class Consistency (Mean and Std Dev per class) ---
    report_lines.append("1. Intra-class Consistency (Mean ± Std Dev):\n")
    grouped_by_fault = df.groupby('fault_type')
    for feature in feature_cols:
        report_lines.append(f"\n  Feature: {feature}\n")
        for fault_type, group_data in grouped_by_fault:
            # Drop NaNs for mean/std calculation for this specific group/feature
            feature_values = group_data[feature].dropna()
            if not feature_values.empty:
                mean_val = feature_values.mean()
                std_val = feature_values.std()
                report_lines.append(f"    {fault_type}: {mean_val:.4f} ± {std_val:.4f} (N={len(feature_values)})\n")
            else:
                report_lines.append(f"    {fault_type}: No valid data\n")
    report_lines.append("\n")

    # --- Inter-class Separability ---
    report_lines.append("2. Inter-class Separability:\n")

    # ANOVA F-statistic for each feature across all classes
    report_lines.append("  a) ANOVA F-statistic (higher F indicates better separability across all classes):\n")
    for feature in feature_cols:
        # Prepare groups for ANOVA, dropping NaNs within each group
        groups_for_anova = [group[feature].dropna().values for name, group in grouped_by_fault if not group[feature].dropna().empty]
        
        if len(groups_for_anova) >= 2 and all(len(g) > 0 for g in groups_for_anova) : # Need at least two groups with data
             # Check if all groups have variance. f_oneway fails if a group is constant.
            if any(np.std(g) == 0 for g in groups_for_anova if len(g) > 1):
                report_lines.append(f"    Feature: {feature} - ANOVA skipped (one or more groups have zero variance).\n")
            else:
                try:
                    f_val, p_val = f_oneway(*groups_for_anova)
                    report_lines.append(f"    Feature: {feature} - F-statistic = {f_val:.4f}, p-value = {p_val:.4g}\n")
                except ValueError as e: # Catches issues like not enough samples or all same values
                     report_lines.append(f"    Feature: {feature} - ANOVA error: {e}\n")
        else:
            report_lines.append(f"    Feature: {feature} - ANOVA skipped (insufficient groups or data for comparison).\n")
    report_lines.append("\n")

    # Fisher's Discriminant Ratio for key class pairs (e.g., 'corona' vs 'pd')
    report_lines.append("  b) Fisher's Discriminant Ratio (higher FDR indicates better separability for the pair):\n")
    key_classes = ['corona', 'pd'] # Define key classes for pairwise comparison
    # Check if these classes exist in the data
    actual_key_classes = [cls for cls in key_classes if cls in df['fault_type'].unique()]
    
    if len(actual_key_classes) == 2:
        class1_data = df[df['fault_type'] == actual_key_classes[0]]
        class2_data = df[df['fault_type'] == actual_key_classes[1]]
        report_lines.append(f"    Comparison: {actual_key_classes[0]} vs {actual_key_classes[1]}\n")
        for feature in feature_cols:
            series1 = class1_data[feature].dropna()
            series2 = class2_data[feature].dropna()
            if not series1.empty and not series2.empty:
                fdr = fisher_discriminant_ratio(series1, series2)
                report_lines.append(f"      Feature: {feature} - FDR = {fdr:.4f}\n")
            else:
                report_lines.append(f"      Feature: {feature} - FDR skipped (insufficient data for one or both classes).\n")
    else:
        report_lines.append("    Skipping Fisher's Discriminant Ratio for 'corona' vs 'pd' as one or both classes not found or not enough distinct key classes.\n")
    report_lines.append("\n")
    
    # --- Overall Dataset Summary ---
    report_lines.append("3. Overall Dataset Summary:\n")
    report_lines.append(f"  Total number of samples: {len(df)}\n")
    report_lines.append(f"  Number of unique fault types: {df['fault_type'].nunique()}\n")
    report_lines.append("  Samples per fault type:\n")
    report_lines.append(str(df['fault_type'].value_counts()))
    report_lines.append("\n\n")


    # Output the report
    final_report = "".join(report_lines)
    print(final_report)

    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                f.write(final_report)
            print(f"Report saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving report to {args.output_file}: {e}")

if __name__ == "__main__":
    main()