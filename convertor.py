#!/usr/bin/env python3
"""
convertor.py

Usage:
    python convertor.py path/to/directory [--data_offset 0x1470]
       [--ch_volt_div_val 5000] [--code_per_div 25] [--ch_vert_offset -7.7]

This script converts all binary files (.bin) in a directory to numpy arrays (.npy) using
the specified conversion parameters.
"""

import os
import argparse
import numpy as np

def convert_bin_to_npy(file_path, data_offset=0x1470, ch_volt_div_val=5000, code_per_div=25, ch_vert_offset=-7.7):
    with open(file_path, 'rb') as f:
        data = f.read()
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8).astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    return voltages

def process_directory(directory_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                print(f'Processing file: {file_path}')
                voltages = convert_bin_to_npy(file_path, data_offset, ch_volt_div_val, code_per_div, ch_vert_offset)
                npy_file_path = file_path.replace('.bin', '.npy')
                np.save(npy_file_path, voltages)
                print(f'Saved numpy array to: {npy_file_path}')

def main():
    parser = argparse.ArgumentParser(description="Convert binary files in a directory to numpy arrays.")
    parser.add_argument("directory", help="Path to the directory containing .bin files")
    parser.add_argument("--data_offset", type=lambda x: int(x, 0), default=0x1470,
                        help="Byte offset where analog data begins (default: 0x1470)")
    parser.add_argument("--ch_volt_div_val", type=float, default=5000,
                        help="Channel voltage division value in mV (default: 5000)")
    parser.add_argument("--code_per_div", type=float, default=25,
                        help="Data code per horizontal division (default: 25)")
    parser.add_argument("--ch_vert_offset", type=float, default=-7.7,
                        help="Channel vertical offset in V (default: -7.7)")
    args = parser.parse_args()

    process_directory(args.directory, args.data_offset, args.ch_volt_div_val, args.code_per_div, args.ch_vert_offset)

if __name__ == "__main__":
    main()
