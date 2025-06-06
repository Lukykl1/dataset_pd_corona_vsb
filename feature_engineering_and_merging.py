#!/usr/bin/env python3
"""
feature_engineering_and_merging.py

Usage:
    python feature_engineering_and_merging.py <spectral_summary_csv> <class_and_channel_summary_csv> <output_ml_features_csv>

Example:
    python feature_engineering_and_merging.py spectral_summary.csv class_and_channel_summary.csv ml_features.csv

This script merges features from spectral_summary.csv and class_and_channel_summary.csv
to create a consolidated ml_features.csv file.
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Merge spectral and class summary features.")
    parser.add_argument("spectral_csv", help="Path to the spectral_summary.csv file.")
    parser.add_argument("class_channel_csv", help="Path to the class_and_channel_summary.csv file.")
    parser.add_argument("output_csv", help="Path to save the merged ml_features.csv file.")
    args = parser.parse_args()

    try:
        spectral_df = pd.read_csv(args.spectral_csv)
    except FileNotFoundError:
        print(f"Error: Spectral summary file not found at {args.spectral_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.spectral_csv}: {e}")
        return

    try:
        class_channel_df = pd.read_csv(args.class_channel_csv)
    except FileNotFoundError:
        print(f"Error: Class and channel summary file not found at {args.class_channel_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.class_channel_csv}: {e}")
        return

    # spectral_df columns: file, fault_type, dominant_freq, num_peaks, rms, skewness, kurtosis
    # class_channel_df columns: mean, median, min, max, std, file, channel, fault_type, sample

    # Merge based on 'file'.
    # We expect 'fault_type' to be consistent for the same 'file' in both CSVs.
    # Let's verify and handle this.
    merged_df = pd.merge(spectral_df, class_channel_df, on="file", suffixes=('_spectral', '_class'))

    # Check consistency of 'fault_type'
    if 'fault_type_spectral' in merged_df.columns and 'fault_type_class' in merged_df.columns:
        if not merged_df['fault_type_spectral'].equals(merged_df['fault_type_class']):
            print("Warning: Discrepancy found between 'fault_type_spectral' and 'fault_type_class'.")
            # Prioritize one, or investigate. For now, let's use _class and report.
            # Count mismatches
            mismatches = merged_df[merged_df['fault_type_spectral'] != merged_df['fault_type_class']]
            if not mismatches.empty:
                print(f"Number of mismatches: {len(mismatches)}")
                print("Example mismatches:")
                print(mismatches[['file', 'fault_type_spectral', 'fault_type_class']].head())
            # We will use fault_type_class as the primary fault_type
            merged_df['fault_type'] = merged_df['fault_type_class']
            merged_df.drop(columns=['fault_type_spectral', 'fault_type_class'], inplace=True)
        else:
            # They are identical, use one and drop the other
            merged_df['fault_type'] = merged_df['fault_type_class'] # or _spectral
            merged_df.drop(columns=['fault_type_spectral', 'fault_type_class'], inplace=True)
    elif 'fault_type_spectral' in merged_df.columns: # Only spectral has fault_type (unlikely given class_df spec)
         merged_df.rename(columns={'fault_type_spectral': 'fault_type'}, inplace=True)
         if 'fault_type_class' in merged_df.columns: # Should not happen if logic above is correct
            merged_df.drop(columns=['fault_type_class'], inplace=True)
    elif 'fault_type_class' in merged_df.columns: # Only class_df has fault_type
        merged_df.rename(columns={'fault_type_class': 'fault_type'}, inplace=True)
        if 'fault_type_spectral' in merged_df.columns:
             merged_df.drop(columns=['fault_type_spectral'], inplace=True)
    # If only one 'fault_type' column exists after merge (e.g. if one df didn't have it), it's fine.


    # Reorder columns for clarity, putting identifiers first
    id_cols = ['file', 'fault_type', 'channel', 'sample']
    feature_cols = [col for col in merged_df.columns if col not in id_cols]
    final_cols = id_cols + feature_cols
    # Ensure all expected id_cols are present before reordering
    final_cols = [col for col in final_cols if col in merged_df.columns]
    
    merged_df = merged_df[final_cols]

    try:
        merged_df.to_csv(args.output_csv, index=False)
        print(f"Successfully merged features saved to {args.output_csv}")
        print("Columns in merged_df:", merged_df.columns.tolist())
    except Exception as e:
        print(f"Error writing merged CSV to {args.output_csv}: {e}")

if __name__ == "__main__":
    main()