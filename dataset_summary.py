#!/usr/bin/env python3
"""
dataset_summary.py

Usage:
    python dataset_summary.py path/to/directory

This script summarizes dataset statistics (e.g., amplitude distributions, duration, frequency content)
and produces a report.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_voltage(file_path, data_offset=0x1470, ch_volt_div_val=5000, code_per_div=25, ch_vert_offset=-7.7):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    return (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset

def generate_summary(directory, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset):
    records = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                voltages = extract_voltage(file_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset)
                record = {
                    "file": file,
                    "mean": np.mean(voltages),
                    "std": np.std(voltages),
                    "min": np.min(voltages),
                    "max": np.max(voltages)
                }
                records.append(record)
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Generate a summary report of the dataset.")
    parser.add_argument("directory", help="Directory containing .bin files")
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470)
    parser.add_argument("--ch_volt_div_val", type=float, default=5000)
    parser.add_argument("--code_per_div", type=float, default=25)
    parser.add_argument("--ch_vert_offset", type=float, default=-7.7)
    args = parser.parse_args()
    
    df = generate_summary(args.directory, args.data_offset, args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)
    print(df.head())
    
    # Plot histograms of mean voltage
    plt.figure(figsize=(8, 6))
    plt.hist(df["mean"], bins=20, edgecolor='black')
    plt.xlabel("Mean Voltage (V)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Mean Voltage")
    plt.grid(True)
    plt.show()
    
    # Save summary report as CSV
    df.to_csv("dataset_summary.csv", index=False)
    print("Dataset summary saved to dataset_summary.csv")
    # For a full HTML/PDF report, one could integrate with libraries such as ReportLab or Jinja2.

if __name__ == "__main__":
    main()
