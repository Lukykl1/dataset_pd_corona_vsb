#!/usr/bin/env python3
"""
extractor.py

Usage:
    python extractor.py path/to/your_file.bin [--data_offset 0x1470]
       [--ch_volt_div_val 1000000] [--code_per_div 256] [--ch_vert_offset 0]

This script extracts analog channel data from a binary file, converts it to voltage values,
and displays a plot of the voltage signal.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def extract_voltage(file_path, data_offset=0x1470, ch_volt_div_val=1000000, code_per_div=256, ch_vert_offset=0):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    return voltages

def main():
    parser = argparse.ArgumentParser(description="Extract and plot analog data from a binary file.")
    parser.add_argument("file_path", help="Path to the binary file (e.g., C1_corona_77.bin)")
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470,
                        help="Byte offset where analog data begins (default: 0x1470)")
    parser.add_argument("--ch_volt_div_val", type=float, default=1000000,
                        help="Channel voltage division value in mV (default: 1000000)")
    parser.add_argument("--code_per_div", type=float, default=256,
                        help="Data code per horizontal division (default: 256)")
    parser.add_argument("--ch_vert_offset", type=float, default=0,
                        help="Channel vertical offset in V (default: 0)")
    args = parser.parse_args()

    voltages = extract_voltage(args.file_path, args.data_offset, args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)

    plt.figure(figsize=(10, 6))
    plt.plot(voltages, label='Analog Channel Data')
    plt.title(f'Voltage Signal from {args.file_path}')
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
