#!/usr/bin/env python3
"""
waveform_and_spectrogram_analyzer.py

Usage:
    python waveform_and_spectrogram_analyzer.py <npy_directory> <ml_features_csv> 
                                                [--sample_rate <rate>] [--num_samples <n>] 
                                                [--output_dir <dir>]

Example:
    python waveform_and_spectrogram_analyzer.py ./npy_data ml_features.csv --sample_rate 500000000 --num_samples 2 --output_dir ./output_waveforms

This script loads representative .npy waveform files for different fault types,
plots their time-domain signals and spectrograms, and saves them.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import random

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot waveforms and spectrograms.")
    parser.add_argument("npy_directory", help="Directory containing .npy waveform files.")
    parser.add_argument("ml_features_csv", help="Path to ml_features.csv for fault type labels.")
    parser.add_argument("--sample_rate", type=float, default=500_000_000,  # 500 MHz
                        help="Sample rate of the signals in Hz (default: 500000000).")
    parser.add_argument("--num_samples_per_class", type=int, default=2,
                        help="Number of random samples to plot per fault class (default: 2).")
    parser.add_argument("--output_dir", default="output_waveforms_spectrograms",
                        help="Directory to save the generated plots (default: output_waveforms_spectrograms).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    try:
        features_df = pd.read_csv(args.ml_features_csv)
    except FileNotFoundError:
        print(f"Error: ML features file not found at {args.ml_features_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.ml_features_csv}: {e}")
        return
    
    if 'file' not in features_df.columns or 'fault_type' not in features_df.columns:
        print("Error: 'file' or 'fault_type' column not found in ml_features.csv.")
        return

    fault_types = features_df['fault_type'].unique()

    for fault_type in fault_types:
        print(f"\nProcessing fault type: {fault_type}")
        class_files_df = features_df[features_df['fault_type'] == fault_type]
        
        if class_files_df.empty:
            print(f"No files found for fault type: {fault_type}")
            continue

        num_to_sample = min(args.num_samples_per_class, len(class_files_df))
        if len(class_files_df) < args.num_samples_per_class:
             print(f"Warning: Requested {args.num_samples_per_class} samples, but only {len(class_files_df)} available for {fault_type}.")
        
        # Sample files (get their original .bin filenames)
        # Ensure 'file' column contains .bin filenames as per original scripts
        sampled_bin_filenames = random.sample(list(class_files_df['file']), num_to_sample)

        for bin_filename in sampled_bin_filenames:
            npy_filename = bin_filename.replace('.bin', '.npy')
            npy_filepath = os.path.join(args.npy_directory, npy_filename)

            if not os.path.exists(npy_filepath):
                print(f"  Skipping: .npy file not found at {npy_filepath}")
                continue
            
            print(f"  Plotting: {npy_filename}")
            try:
                voltages = np.load(npy_filepath)
            except Exception as e:
                print(f"  Error loading {npy_filepath}: {e}")
                continue

            time_vector = np.arange(len(voltages)) / args.sample_rate

            plt.figure(figsize=(15, 10))

            # Time-domain plot
            plt.subplot(2, 1, 1)
            plt.plot(time_vector * 1000, voltages) # Time in ms
            plt.title(f'Waveform: {npy_filename} (Fault: {fault_type})')
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage (V)')
            plt.grid(True)

            # Spectrogram plot
            plt.subplot(2, 1, 2)
            # Adjust nperseg for resolution. Power of 2 is common.
            # For 10M points, nperseg=2048 or 4096 could be reasonable.
            # If sample rate is 500MHz, N_points = 10^7, duration = 0.02s
            # nperseg default is 256. Let's try something larger for better freq resolution.
            nperseg_val = min(len(voltages), 2048) 
            noverlap_val = nperseg_val // 2

            # Handle potential issues with very short signals for spectrogram
            if len(voltages) < nperseg_val:
                 print(f"    Signal too short for spectrogram with nperseg={nperseg_val}. Skipping spectrogram for {npy_filename}.")
                 plt.text(0.5, 0.5, "Signal too short for spectrogram", horizontalalignment='center', verticalalignment='center')

            else:
                f, t_spec, Sxx = signal.spectrogram(voltages, fs=args.sample_rate, nperseg=nperseg_val, noverlap=noverlap_val)
                # Plot in dB for better dynamic range visualization
                Sxx_db = 10 * np.log10(Sxx + 1e-9) # Add small epsilon to avoid log(0)
                plt.pcolormesh(t_spec * 1000, f / 1e6, Sxx_db, shading='gouraud', cmap='viridis') # Time in ms, Freq in MHz
                plt.ylabel('Frequency (MHz)')
                plt.xlabel('Time (ms)')
                plt.title(f'Spectrogram: {npy_filename} (Fault: {fault_type})')
                plt.colorbar(label='Intensity (dB)')
            
            plt.tight_layout()
            
            base_plot_filename = os.path.splitext(npy_filename)[0]
            plot_filename = os.path.join(args.output_dir, f'{base_plot_filename}_analysis.png')
            try:
                plt.savefig(plot_filename)
            except Exception as e:
                print(f"  Error saving plot {plot_filename}: {e}")
            plt.close()
            
    print(f"\nWaveform and spectrogram analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()