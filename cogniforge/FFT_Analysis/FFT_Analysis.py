# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:07:40 2025

@author: Johann
"""
from io import BytesIO
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cogniforge.utils.furthr import FURTHRmind, download_item_bytes
import streamlit as st
###############################################################################
# Functions
def calculate_resistance(voltage, current):
    """Calculate resistance from voltage and current arrays."""
    return np.where(current != 0, voltage / current, np.nan)

# Function for sampling rate
def calculate_Fs(timely):
    """Calculate sampling frequency for FFT power spectrum"""
    time_diff = np.diff(timely.ravel())  # Get differences between consecutive timestamps
    avg_time_interval = np.mean(time_diff)  # Average time interval between samples
    Fs = 1 / avg_time_interval  # Sampling frequency
    return Fs

# Function for FFT and subsequent transformations
def fftit(df_array, Fs, lbFreq, ubFreq):
    """Conduct Fast-Fourier Transform on vector"""
    meanLess = df_array.ravel() - np.mean(df_array.ravel())  # Subtract mean value
    Y = np.fft.fft(meanLess)  # Calculate FFT
    L = len(df_array)  # Sample length
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] *= 2  # Only one-sided FFT             
    f = Fs * np.arange((L / 2 + 1)) / L  # Frequency bins in Hz
    # Filter based on frequency bounds
    P1 = P1[(f >= lbFreq) & (f < ubFreq)]
    f = f[(f >= lbFreq) & (f < ubFreq)]
    maxP1 = np.max(P1)  # Find maximum power in spectrum
    FundFreq = f[maxP1 == P1] if maxP1.size > 0 else None  # Get frequency with highest power
    
    return P1, f, FundFreq

# Define Gaussian function for fitting
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def normalize_array(arr):
    """Normalize a NumPy array to the range [0, 1]."""
    min_value = np.min(arr)
    max_value = np.max(arr)

    if max_value > min_value:
        return (arr - min_value) / (max_value - min_value)
    else:
        return np.zeros_like(arr)  # Return an array of zeros if all values are identical

# Select the experiment using the FURTHRmind API
def select_experiment():
    csv_widget = FURTHRmind(id="experiment")
    csv_widget.file_extension = "csv"
    csv_widget.container_category = "experiment"
    
    # Let the user select the experiment
    csv_widget.select_container()
    
    # Check if an experiment is selected
    if st.button("Perform"):
        st.session_state.perform_clicked = True
    if csv_widget.selected and st.session_state.perform_clicked:
        return csv_widget.selected.get()
    else:
        print("No experiment selected, exiting.")
        exit(1)
###############################################################################
# Main Execution

# Select the experiment and retrieve the files
if "perform_clicked" not in st.session_state:
    st.session_state.perform_clicked = False
experiment = select_experiment()
if st.session_state.perform_clicked:
    output_folder = "C:\\FFT_Results"  # Prompt for output folder
    os.makedirs(output_folder, exist_ok=True)
    allData = {}  # Dictionary to store all DataFrames

    # Fetch all CSV files associated with the experiment
    csv_files = [file for file in experiment.files if file.name.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the selected experiment.")
    else:
        for file in csv_files:
            try:
                # Get the CSV bytes using download_item_bytes
                csv_bytes, _ = download_item_bytes(file)
                
                # Read CSV data into pandas dataframe
                from io import StringIO
                csvdata = pd.read_csv(csv_bytes, delimiter=';', decimal=',', header=2)
                csv_text = csv_bytes.getvalue().decode('utf-8')
                # Process the CSV data (same logic as before)
                headers_and_units = [line.strip().split(';') for line in csv_text.splitlines()[:2]]
                headers = [f"{item1}_{item2}" for item1, item2 in zip(headers_and_units[0], headers_and_units[1])]
                csvdata.columns = headers
                
                voltage_col_name = csvdata.columns[csvdata.columns.str.contains('volt|spannung', case=False)]
                current_col_name = csvdata.columns[csvdata.columns.str.contains('current|ampere|amperage|strom', case=False)]
                time_col_name = csvdata.columns[csvdata.columns.str.contains('time|zeit', case=False)]
                
                # Check if the necessary columns exist
                if not voltage_col_name.empty and not current_col_name.empty:
                    resistance = calculate_resistance(csvdata[voltage_col_name].values, csvdata[current_col_name].values)
                    
                    if not time_col_name.empty:
                        # Calculate FS
                        Fs = calculate_Fs(csvdata[time_col_name].values)
                        print(f"Calculated Sampling Frequency (Fs): {Fs} Hz")
                        
                        # Calculate FFT power spectrum of resistance
                        power_values, fft_bins, fundamental_freq = fftit(resistance, Fs, lbFreq=100, ubFreq=10000)
                        
                        # Normalize power values between 0 and 1
                        norm_power_values = normalize_array(power_values)
                        
                        # Find the index of the maximum value in the filtered magnitude array
                        peak_index = np.argmax(power_values)
                        
                        if peak_index > 0:
                            peak_frequency = fft_bins[peak_index]
                            peak_power_value = power_values[peak_index]
                            peak_norm_power_value = norm_power_values[peak_index]
                            
                            print(f"Peak Frequency: {peak_frequency} Hz with Power: {peak_power_value} Normalized Power: {peak_norm_power_value * 100} %")
                            
                            fit_range = peak_frequency * 0.4
                            fit_range_indices = (fft_bins >= peak_frequency - fit_range) & (fft_bins <= peak_frequency + fit_range)
                            fit_frequencies = fft_bins[fit_range_indices]
                            fit_amplitudes = norm_power_values[fit_range_indices]
                            
                            initial_guess = [peak_norm_power_value, peak_frequency, 1]
                            
                            try:
                                popt, _ = curve_fit(gaussian, fit_frequencies, fit_amplitudes, p0=initial_guess)
                                gauss_mean, gauss_stddev = popt[1], popt[2]
                                
                                # Plot FFT and Gaussian Fit
                                plt.figure(figsize=(10,6))
                                plt.plot(fft_bins, norm_power_values[:len(fft_bins)], label='FFT Magnitude')
                                
                                x_fit_range = np.linspace(0, peak_frequency + 1000, num=500)
                                y_fit_range = gaussian(x_fit_range, *popt)
                                
                                short_filename = file.name[:-4]
                                plt.plot(x_fit_range, y_fit_range, label='Gaussian Fit', color='red')
                                plt.axvline(x=gauss_mean, color='green', linestyle='--', label='Mean')
                                plt.axvline(x=gauss_mean + gauss_stddev, color='orange', linestyle=':', label='Mean + Std Dev')
                                plt.axvline(x=gauss_mean - gauss_stddev, color='orange', linestyle=':', label='Mean - Std Dev')
                                
                                plt.title(f'FFT Analysis for {short_filename}')
                                plt.xlabel('Frequency (Hz)')
                                plt.ylabel('Power')
                                plt.legend()
                                plt.grid()
                                
                                # Save plot to output directory
                                output_plot_filepath = os.path.join(output_folder, f"{short_filename}_FFT_Analysis.png")
                                plt.savefig(output_plot_filepath)
                                
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png')
                                buffer.seek(0)
                                plt.close()

                            except RuntimeError as e:
                                print(f"Gaussian fitting failed for {file.name}: {e}")
                
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
