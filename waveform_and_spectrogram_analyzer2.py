#!/usr/bin/env python3
"""
waveform_and_spectrogram_analyzer.py

Usage:
    python waveform_and_spectrogram_analyzer.py <npy_directory> <ml_features_csv> 
                                                [--sample_rate <rate>] [--num_samples_per_class <n>] 
                                                [--output_dir <dir>]

Example:
    python waveform_and_spectrogram_analyzer.py ./npy_data ml_features.csv --sample_rate 500000000 --num_samples_per_class 2 --output_dir ./output_waveforms

This script loads representative .npy waveform files for different fault types,
plots their time-domain signals and spectrograms, and saves them,
separately for each antenna channel.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import random

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot waveforms and spectrograms per channel.")
    parser.add_argument("npy_directory", help="Directory containing .npy waveform files.")
    parser.add_argument("ml_features_csv", help="Path to ml_features.csv for fault type and channel labels.")
    parser.add_argument("--sample_rate", type=float, default=500_000_000,
                        help="Sample rate of the signals in Hz (default: 500000000).")
    parser.add_argument("--num_samples_per_class", type=int, default=2,
                        help="Number of random samples to plot per fault class per channel (default: 2).")
    parser.add_argument("--output_dir", default="output_waveforms_spectrograms",
                        help="Directory to save the generated plots (default: output_waveforms_spectrograms).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    try:
        features_df_all_channels = pd.read_csv(args.ml_features_csv)
    except FileNotFoundError:
        print(f"Error: ML features file not found at {args.ml_features_csv}")
        return
    except Exception as e:
        print(f"Error reading {args.ml_features_csv}: {e}")
        return
    
    if 'file' not in features_df_all_channels.columns or \
       'fault_type' not in features_df_all_channels.columns or \
       'channel' not in features_df_all_channels.columns:
        print("Error: 'file', 'fault_type', or 'channel' column not found in ml_features.csv.")
        return

    unique_channels = features_df_all_channels['channel'].unique()
    print(f"Found channels: {unique_channels}")

    for channel_name in unique_channels:
        print(f"\nProcessing Channel: {channel_name}")
        df_channel = features_df_all_channels[features_df_all_channels['channel'] == channel_name]
        
        if df_channel.empty:
            print(f"No data found for channel {channel_name}. Skipping.")
            continue

        # Create a subdirectory for each channel's plots
        channel_output_dir = os.path.join(args.output_dir, f"channel_{channel_name}")
        if not os.path.exists(channel_output_dir):
            os.makedirs(channel_output_dir)
            print(f"Created output directory for channel {channel_name}: {channel_output_dir}")

        fault_types_in_channel = df_channel['fault_type'].unique()

        for fault_type in fault_types_in_channel:
            print(f"  Processing Fault Type: {fault_type} for Channel {channel_name}")
            class_files_df = df_channel[df_channel['fault_type'] == fault_type]
            
            if class_files_df.empty:
                print(f"    No files found for fault type: {fault_type} in channel {channel_name}")
                continue

            num_to_sample = min(args.num_samples_per_class, len(class_files_df))
            if len(class_files_df) < args.num_samples_per_class:
                 print(f"    Warning: Requested {args.num_samples_per_class} samples, but only {len(class_files_df)} available for {fault_type} in channel {channel_name}.")
            
            sampled_bin_filenames = random.sample(list(class_files_df['file']), num_to_sample)

            for bin_filename in sampled_bin_filenames:
                npy_filename = bin_filename.replace('.bin', '.npy')
                npy_filepath = os.path.join(args.npy_directory, npy_filename) # Assuming npy files are flat in npy_directory

                if not os.path.exists(npy_filepath):
                    print(f"    Skipping: .npy file not found at {npy_filepath}")
                    continue
                
                print(f"    Plotting: {npy_filename} (Channel {channel_name})")
                try:
                    voltages = np.load(npy_filepath)
                except Exception as e:
                    print(f"    Error loading {npy_filepath}: {e}")
                    continue

                time_vector = np.arange(len(voltages)) / args.sample_rate

                plt.figure(figsize=(15, 10))

                plt.subplot(2, 1, 1)
                plt.plot(time_vector * 1000, voltages) # Time in ms
                plt.title(f'Waveform: {npy_filename} (Ch: {channel_name}, Fault: {fault_type})')
                plt.xlabel('Time (ms)')
                plt.ylabel('Voltage (V)')
                plt.grid(True)

                plt.subplot(2, 1, 2)
                nperseg_val = min(len(voltages), 2048) 
                noverlap_val = nperseg_val // 2

                if len(voltages) < nperseg_val:
                     print(f"      Signal too short for spectrogram with nperseg={nperseg_val}. Skipping spectrogram for {npy_filename}.")
                     plt.text(0.5, 0.5, "Signal too short for spectrogram", horizontalalignment='center', verticalalignment='center')
                else:
                    f, t_spec, Sxx = signal.spectrogram(voltages, fs=args.sample_rate, nperseg=nperseg_val, noverlap=noverlap_val)
                    Sxx_db = 10 * np.log10(Sxx + 1e-9) 
                    plt.pcolormesh(t_spec * 1000, f / 1e6, Sxx_db, shading='gouraud', cmap='viridis') # Time in ms, Freq in MHz
                    plt.ylabel('Frequency (MHz)')
                    plt.xlabel('Time (ms)')
                    plt.title(f'Spectrogram: {npy_filename} (Ch: {channel_name}, Fault: {fault_type})')
                    plt.colorbar(label='Intensity (dB)')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                base_plot_filename = os.path.splitext(npy_filename)[0]
                # Ensure filename uniqueness if multiple channels share npy filenames (should not happen with C1/C2 prefix in original .bin)
                plot_filename = os.path.join(channel_output_dir, f'{channel_name}_{base_plot_filename}_analysis.png')
                try:
                    plt.savefig(plot_filename)
                except Exception as e:
                    print(f"    Error saving plot {plot_filename}: {e}")
                plt.close()
            
    print(f"\nPer-channel waveform and spectrogram analysis complete. Plots saved to subdirectories in {args.output_dir}")

if __name__ == "__main__":
    main()