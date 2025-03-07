#!/usr/bin/env python3
"""
batch_preprocess.py

Usage:
    python batch_preprocess.py path/to/directory

This script runs convertor.py, global_stats.py, class_summary.py, and spectral_summary.py sequentially
to process the entire dataset automatically.
"""

import subprocess
import sys
import os

def run_script(script_name, args):
    command = [sys.executable, script_name] + args
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_preprocess.py path/to/directory")
        sys.exit(1)
    directory = sys.argv[1]
    
    # Convert binary files to numpy arrays
    run_script("convertor.py", [directory])
    
    # Compute global statistics
    run_script("global_stats.py", [directory])
    
    # Compute class summary
    run_script("class_summary.py", [directory])
    
    # Perform spectral analysis
    run_script("spectral_summary.py", [directory])
    
    print("Batch preprocessing complete.")

if __name__ == "__main__":
    main()
