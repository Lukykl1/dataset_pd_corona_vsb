#!/usr/bin/env python3
"""
spectral_summary.py

Usage:
    python spectral_summary.py path/to/directory [--sample_rate 500000000]
       [--data_offset 0x1470] [--ch_volt_div_val 5000] [--code_per_div 25] [--ch_vert_offset -7.7]

This script computes spectral features (via FFT) for each binary file and groups the results by fault type.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy.signal

def extract_voltage(file_path, data_offset=0x1470, ch_volt_div_val=5000, code_per_div=25, ch_vert_offset=-7.7):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    return voltages

def compute_spectral_features(voltages, sample_rate):
    N = len(voltages)
    fft_vals = np.fft.fft(voltages)
    fft_freq = np.fft.fftfreq(N, d=1/sample_rate)
    # Consider only positive frequencies
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    fft_magnitude = np.abs(fft_vals[pos_mask])
    
    # Dominant frequency (ignoring the zero-frequency/DC component)
    if len(fft_magnitude) > 1:
        idx = np.argmax(fft_magnitude[1:]) + 1
        dominant_freq = freqs[idx]
    else:
        dominant_freq = 0
    
    # Find number of peaks (using 10% of max amplitude as threshold)
    peaks, _ = scipy.signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)
    num_peaks = len(peaks)
    
    # Additional time-domain features
    rms = np.sqrt(np.mean(voltages**2))
    skewness = scipy.stats.skew(voltages)
    kurtosis = scipy.stats.kurtosis(voltages)
    
    return dominant_freq, num_peaks, rms, skewness, kurtosis

def parse_fault_type(file_path):
    """
    Expected filename format:
        C{antenna}_{fault_type}_{sample}.bin
    """
    base = os.path.basename(file_path).replace('.bin', '')
    tokens = base.split('_')
    fault_type = "_".join(tokens[1:-1])
    return fault_type

def process_directory(directory, sample_rate, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset):
    records = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                fault_type = parse_fault_type(file_path)
                voltages = extract_voltage(file_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset)
                dominant_freq, num_peaks, rms, skewness, kurtosis = compute_spectral_features(voltages, sample_rate)
                record = {
                    "file": file,
                    "fault_type": fault_type,
                    "dominant_freq": dominant_freq,
                    "num_peaks": num_peaks,
                    "rms": rms,
                    "skewness": skewness,
                    "kurtosis": kurtosis
                }
                records.append(record)
    return records

def main():
    parser = argparse.ArgumentParser(description="Perform spectral analysis on binary files.")
    parser.add_argument("directory", help="Directory containing .bin files")
    parser.add_argument("--sample_rate", type=float, default=500000000,
                        help="Sampling rate in Hz (default: 500000000 for 10^7 points over 20 ms)")
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470)
    parser.add_argument("--ch_volt_div_val", type=float, default=5000)
    parser.add_argument("--code_per_div", type=float, default=25)
    parser.add_argument("--ch_vert_offset", type=float, default=-7.7)
    args = parser.parse_args()

    records = process_directory(args.directory, args.sample_rate, args.data_offset,
                                args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)
    df = pd.DataFrame(records)
    
    # Example plot: dominant frequency by fault type
    plt.figure(figsize=(10, 6))
    for fault_type, group in df.groupby("fault_type"):
        plt.plot(group.index, group["dominant_freq"], 'o', label=fault_type)
    plt.xlabel("File Index")
    plt.ylabel("Dominant Frequency (Hz)")
    plt.title("Dominant Frequency by Fault Type")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    df.to_csv("spectral_summary.csv", index=False)
    print("Spectral summary saved to spectral_summary.csv")

if __name__ == "__main__":
    main()
