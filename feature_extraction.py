#!/usr/bin/env python3
"""
feature_extraction.py

Usage:
    python feature_extraction.py path/to/directory [--sample_rate 50000000]

This script converts time-series data into numerical features (statistical and spectral)
and saves them in a CSV file.
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal

def extract_voltage(file_path, data_offset=0x1470, ch_volt_div_val=5000, code_per_div=25, ch_vert_offset=-7.7):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    return (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset

def compute_time_features(voltages):
    return {
        "mean": np.mean(voltages),
        "median": np.median(voltages),
        "std": np.std(voltages),
        "min": np.min(voltages),
        "max": np.max(voltages),
        "rms": np.sqrt(np.mean(voltages**2)),
        "ptp": np.ptp(voltages)  # peak-to-peak
    }

def compute_spectral_features(voltages, sample_rate):
    N = len(voltages)
    fft_vals = np.fft.fft(voltages)
    fft_freq = np.fft.fftfreq(N, d=1/sample_rate)
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    fft_magnitude = np.abs(fft_vals[pos_mask])
    
    dominant_freq = freqs[np.argmax(fft_magnitude[1:])+1] if len(fft_magnitude) > 1 else 0
    peaks, _ = scipy.signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)
    
    features = {
        "dominant_freq": dominant_freq,
        "num_peaks": len(peaks)
    }
    return features

def parse_fault_type(file_path):
    base = os.path.basename(file_path).replace('.bin', '')
    tokens = base.split('_')
    return "_".join(tokens[1:-1])

def process_directory(directory, sample_rate, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset):
    records = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                voltages = extract_voltage(file_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset)
                time_feats = compute_time_features(voltages)
                spec_feats = compute_spectral_features(voltages, sample_rate)
                fault_type = parse_fault_type(file_path)
                record = {"file": file, "fault_type": fault_type}
                record.update(time_feats)
                record.update(spec_feats)
                records.append(record)
    return records

def main():
    parser = argparse.ArgumentParser(description="Extract features for machine learning from binary files.")
    parser.add_argument("directory", help="Directory containing .bin files")
    parser.add_argument("--sample_rate", type=float, default=50000000)
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470)
    parser.add_argument("--ch_volt_div_val", type=float, default=5000)
    parser.add_argument("--code_per_div", type=float, default=25)
    parser.add_argument("--ch_vert_offset", type=float, default=-7.7)
    args = parser.parse_args()
    
    records = process_directory(args.directory, args.sample_rate, args.data_offset,
                                args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)
    df = pd.DataFrame(records)
    df.to_csv("ml_features.csv", index=False)
    print("Feature extraction complete. Results saved to ml_features.csv")

if __name__ == "__main__":
    main()
