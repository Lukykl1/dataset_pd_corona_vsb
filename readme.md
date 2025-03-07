# XLPE Partial Discharge and Corona Discharge Analysis Tools

## Overview
This repository provides a suite of Python tools designed for processing, analyzing, and extracting features from a dataset collected for distinguishing between partial discharges and corona discharges in XLPE-covered conductors used in medium-voltage overhead power distribution lines.

Using a contactless antenna-based approach, we collected signals with a sampling rate of \(10^7\) data points over a 20 ms window. The dataset includes measurements from two different antennas to capture variations in signal characteristics, collected over two days with 100 samples per discharge type across five fault classes and two background types. This dataset aims to support the development of machine learning models for accurate and non-invasive discharge classification, enhancing the reliability of power distribution networks.

## Tools Overview
The repository includes scripts for preprocessing, feature extraction, visualization, and statistical analysis:

### 1. **Preprocessing and Data Conversion**
- `batch_preprocess.py`: Automates the execution of all preprocessing steps in sequence.
- `convertor.py`: Converts raw binary measurement files into numpy arrays (.npy) for further processing.
- `downsampling.py`: Downsamples high-resolution signals using various methods such as averaging, decimation, and max pooling.

### 2. **Statistical and Feature Analysis**
- `global_stats.py`: Computes global statistical summaries (mean, median, standard deviation, min/max values) for each signal.
- `class_summary.py`: Aggregates and summarizes measurements by fault type.
- `spectral_summary.py`: Performs spectral analysis using FFT and extracts dominant frequency and peak characteristics.
- `feature_extraction.py`: Extracts numerical features for machine learning, including time-domain and spectral characteristics.
- `correlation_analysis.py`: Computes feature correlations to identify redundancy in extracted attributes.

### 3. **Visualization and Interactive Tools**
- `dataset_summary.py`: Generates summary statistics and visualizations of amplitude distributions and frequency content.
- `extractor.py`: Extracts analog channel data from a binary file and plots the voltage signal.
- `interactive_visualizer.py`: Provides an interactive Jupyter Notebook interface for exploring signals and their FFT representations.

### 4. **Classification and ML Pipeline (Upcoming)**
- Planned integration with machine learning models for classification and anomaly detection.

## Usage
Most scripts can be executed via the command line, with parameters allowing customization of data processing steps. For example, to convert binary files to numpy arrays:
```bash
python convertor.py path/to/dataset --data_offset 0x1470 --ch_volt_div_val 5000
```
For batch processing of an entire dataset:
```bash
python batch_preprocess.py path/to/dataset
```
To visualize a specific signal interactively, launch Jupyter Notebook and run:
```bash
jupyter notebook interactive_visualizer.py
```

## Dataset Significance
The dataset is structured to aid in the development and validation of non-invasive discharge classification techniques. The ability to accurately distinguish between partial discharges and corona discharges can significantly improve the maintenance and safety of overhead power lines, reducing the risk of insulation failure and associated hazards.


## License
This repository is released under the [MIT License](LICENSE).