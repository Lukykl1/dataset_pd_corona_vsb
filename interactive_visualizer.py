#!/usr/bin/env python3
"""
interactive_visualizer.py

Usage:
    jupyter notebook interactive_visualizer.py

This script creates an interactive widget for exploring signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, IntSlider
import os

# Define a simple voltage extraction function (adjust parameters as needed)
def extract_voltage(file_path, data_offset=0x1470, ch_volt_div_val=5000, code_per_div=25, ch_vert_offset=-7.7):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    return voltages

def plot_signal(file_path):
    voltages = extract_voltage(file_path)
    plt.figure(figsize=(10, 4))
    plt.plot(voltages, label='Raw Signal')
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.title(f"Signal from {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_fft(file_path, sample_rate=50000000):
    voltages = extract_voltage(file_path)
    N = len(voltages)
    fft_vals = np.fft.fft(voltages)
    fft_freq = np.fft.fftfreq(N, d=1/sample_rate)
    pos_mask = fft_freq >= 0
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq[pos_mask], np.abs(fft_vals[pos_mask]), label='FFT Magnitude')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.show()

# List .bin files in the current directory (or adjust path as needed)
def get_bin_files(directory="."):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bin')]

def interactive_visualizer():
    files = get_bin_files()
    if not files:
        print("No .bin files found in the current directory.")
        return
    file_selector = Dropdown(options=files, description="File:")
    view_mode = Dropdown(options=["Raw Signal", "FFT"], description="View Mode:")

    def update(file, mode):
        if mode == "Raw Signal":
            plot_signal(file)
        else:
            plot_fft(file)
    
    interact(update, file=file_selector, mode=view_mode)

if __name__ == "__main__":
    interactive_visualizer()
