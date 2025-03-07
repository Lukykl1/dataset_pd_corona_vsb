#!/usr/bin/env python3
"""
class_and_channel_summary.py

Usage:
    python class_and_channel_summary.py path/to/directory [--data_offset 0x1470]
       [--ch_volt_div_val 5000] [--code_per_div 25] [--ch_vert_offset -7.7]

This script computes summary statistics for each binary file, categorizes the results by antenna channel,
and produces plots for comparison.
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

def parse_filename(file_path):
    """
    Expected filename format:
        C{antenna}_{fault_type}_{sample}.bin
    Example: C2_pd_HI_99.bin or C1_corona_77.bin
    """
    base = os.path.basename(file_path).replace('.bin', '')
    tokens = base.split('_')
    channel = tokens[0]            # e.g., C1 or C2
    sample = tokens[-1]            # sample number
    fault_type = "_".join(tokens[1:-1])
    return channel, fault_type, sample

def process_directory(directory, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset):
    records = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                channel, fault_type, sample = parse_filename(file_path)
                voltages = extract_voltage(file_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset)
                stats = compute_stats(voltages)
                stats.update({"file": file, "channel": channel, "fault_type": fault_type, "sample": sample})
                records.append(stats)
    return records

def main():
    parser = argparse.ArgumentParser(description="Compute and plot summary statistics grouped by antenna channel.")
    parser.add_argument("directory", help="Directory containing .bin files")
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470)
    parser.add_argument("--ch_volt_div_val", type=float, default=5000)
    parser.add_argument("--code_per_div", type=float, default=25)
    parser.add_argument("--ch_vert_offset", type=float, default=-7.7)
    args = parser.parse_args()

    records = process_directory(args.directory, args.data_offset, args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)
    df = pd.DataFrame(records)
    
    # Plot mean voltage per channel
    plt.figure(figsize=(8, 6))
    for channel, group in df.groupby("channel"):
        plt.plot(group["sample"], group["mean"], 'o-', label=f'Channel {channel}')
    plt.xlabel("Sample Number")
    plt.ylabel("Mean Voltage (V)")
    plt.title("Mean Voltage Grouped by Antenna Channel")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Optionally, save the dataframe
    df.to_csv("class_and_channel_summary.csv", index=False)
    print("Class and channel summary saved to class_and_channel_summary.csv")

if __name__ == "__main__":
    main()
