#!/usr/bin/env python3
"""
downsampling.py

Usage:
    python downsampling.py path/to/npy_file.npy --target_rate 1000000 --method averaging

This script downsamples the given numpy array file to a target number of samples.
Supported methods: averaging, decimation, max_pooling.
"""

import argparse
import numpy as np
import os

def downsample(data, factor, method="averaging"):
    if method == "averaging":
        # Reshape data to (n_groups, factor) and take the mean along axis 1
        n = len(data) // factor
        data = data[:n*factor]
        return data.reshape(-1, factor).mean(axis=1)
    elif method == "decimation":
        return data[::factor]
    elif method == "max_pooling":
        n = len(data) // factor
        data = data[:n*factor]
        return data.reshape(-1, factor).max(axis=1)
    else:
        raise ValueError("Unsupported downsampling method.")

def main():
    parser = argparse.ArgumentParser(description="Downsample a numpy array file.")
    parser.add_argument("npy_file", help="Path to the .npy file")
    parser.add_argument("--target_rate", type=int, required=True, help="Target number of samples")
    parser.add_argument("--method", choices=["averaging", "decimation", "max_pooling"], default="averaging")
    args = parser.parse_args()
    
    data = np.load(args.npy_file)
    factor = len(data) // args.target_rate
    if factor < 1:
        print("Target rate is higher than the original sample count. No downsampling applied.")
        downsampled = data
    else:
        downsampled = downsample(data, factor, method=args.method)
    
    out_file = args.npy_file.replace('.npy', f'_downsampled.npy')
    np.save(out_file, downsampled)
    print(f"Downsampled data saved to {out_file}")

if __name__ == "__main__":
    main()
