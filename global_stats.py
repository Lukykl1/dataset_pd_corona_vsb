#!/usr/bin/env python3
"""
global_stats.py

Usage:
    python global_stats.py path/to/directory [--output summary_statistics.csv]
       [--data_offset 0x1470] [--ch_volt_div_val 5000] [--code_per_div 25] [--ch_vert_offset -7.7]

This script computes global summary statistics for each binary file and saves the results to a CSV file.
"""

import os
import argparse
import numpy as np
import pandas as pd

def extract_voltage(file_path, data_offset=0x1470, ch_volt_div_val=5000, code_per_div=25, ch_vert_offset=-7.7):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    return voltages

def compute_stats(voltages):
    return {
        "mean": np.mean(voltages),
        "median": np.median(voltages),
        "min": np.min(voltages),
        "max": np.max(voltages),
        "std": np.std(voltages)
    }

def process_directory(directory, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset):
    records = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                voltages = extract_voltage(file_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset)
                stats = compute_stats(voltages)
                stats["file"] = file
                records.append(stats)
    return records

def main():
    parser = argparse.ArgumentParser(description="Compute global summary statistics for binary files.")
    parser.add_argument("directory", help="Directory containing .bin files")
    parser.add_argument("--output", default="summary_statistics.csv", help="Output CSV filename (default: summary_statistics.csv)")
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470)
    parser.add_argument("--ch_volt_div_val", type=float, default=5000)
    parser.add_argument("--code_per_div", type=float, default=25)
    parser.add_argument("--ch_vert_offset", type=float, default=-7.7)
    args = parser.parse_args()

    records = process_directory(args.directory, args.data_offset, args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"Summary statistics saved to {args.output}")

if __name__ == "__main__":
    main()
