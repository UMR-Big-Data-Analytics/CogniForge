# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:07:40 2025
@author: Johann
"""
from io import BytesIO, StringIO
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cogniforge.utils.furthr import FURTHRmind, download_item_bytes
import streamlit as st
from furthrmind.file_loader import FileLoader
from furthrmind import Furthrmind as API
from furthrmind.collection import Experiment


###############################################################################
# Utility Functions

def calculate_resistance(voltage, current):
    return np.where(current != 0, voltage / current, np.nan)

def calculate_Fs(timely):
    time_diff = np.diff(timely.ravel())
    avg_time_interval = np.mean(time_diff)
    Fs = 1 / avg_time_interval
    return Fs

def fftit(df_array, Fs, lbFreq, ubFreq):
    meanLess = df_array.ravel() - np.mean(df_array.ravel())
    Y = np.fft.fft(meanLess)
    L = len(df_array)
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] *= 2
    f = Fs * np.arange((L / 2 + 1)) / L
    P1 = P1[(f >= lbFreq) & (f < ubFreq)]
    f = f[(f >= lbFreq) & (f < ubFreq)]
    maxP1 = np.max(P1)
    FundFreq = f[maxP1 == P1] if maxP1.size > 0 else None
    return P1, f, FundFreq

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def normalize_array(arr):
    min_value = np.min(arr)
    max_value = np.max(arr)
    return (arr - min_value) / (max_value - min_value) if max_value > min_value else np.zeros_like(arr)

###############################################################################
# Step Control Initialization

if "step" not in st.session_state:
    st.session_state.step = "select_experiment"
if "output_folder" not in st.session_state:
    st.session_state.output_folder = "C:\\FFT_Results"

###############################################################################
# STEP 1: Select Experiment and Analyze

if st.session_state.step == "select_experiment":
    st.title("Step 1: Select Experiment for FFT Analysis")
    csv_widget = FURTHRmind(id="experiment")
    csv_widget.container_category = "experiment"
    csv_widget.file_extension = "csv"
    csv_widget.select_container()

    if st.button("Perform FFT Analysis"):
        if csv_widget.selected:
            st.session_state.selected_experiment = csv_widget.selected.get()
            st.session_state.step = "analyze"
            st.rerun()
        else:
            st.warning("Please select an experiment folder.")

###############################################################################
# STEP 2: Run FFT Analysis

elif st.session_state.step == "analyze":
    experiment = st.session_state.selected_experiment
    output_folder = st.session_state.output_folder
    os.makedirs(output_folder, exist_ok=True)
    csv_files = [file for file in experiment.files if file.name.endswith('.csv')]

    if not csv_files:
        st.warning("No CSV files found in the selected experiment.")
        st.stop()

    for file in csv_files:
        try:
            csv_bytes, _ = download_item_bytes(file)
            csv_text = csv_bytes.getvalue().decode('utf-8')
            headers_and_units = [line.strip().split(';') for line in csv_text.splitlines()[:2]]
            headers = [f"{item1}_{item2}" for item1, item2 in zip(headers_and_units[0], headers_and_units[1])]

            csvdata = pd.read_csv(StringIO(csv_text), delimiter=';', decimal=',', header=2)
            csvdata.columns = headers

            voltage_col = csvdata.columns[csvdata.columns.str.contains('volt|spannung', case=False)]
            current_col = csvdata.columns[csvdata.columns.str.contains('current|ampere|amperage|strom', case=False)]
            time_col = csvdata.columns[csvdata.columns.str.contains('time|zeit', case=False)]

            if not voltage_col.empty and not current_col.empty and not time_col.empty:
                resistance = calculate_resistance(csvdata[voltage_col].values, csvdata[current_col].values)
                Fs = calculate_Fs(csvdata[time_col].values)
                power_values, fft_bins, _ = fftit(resistance, Fs, lbFreq=100, ubFreq=10000)
                norm_power = normalize_array(power_values)
                peak_index = np.argmax(power_values)

                if peak_index > 0:
                    peak_freq = fft_bins[peak_index]
                    peak_power = power_values[peak_index]
                    peak_norm = norm_power[peak_index]

                    fit_range = peak_freq * 0.4
                    fit_indices = (fft_bins >= peak_freq - fit_range) & (fft_bins <= peak_freq + fit_range)
                    fit_freqs = fft_bins[fit_indices]
                    fit_amps = norm_power[fit_indices]

                    initial_guess = [peak_norm, peak_freq, 1]
                    popt, _ = curve_fit(gaussian, fit_freqs, fit_amps, p0=initial_guess)

                    plt.figure(figsize=(10, 6))
                    plt.plot(fft_bins, norm_power[:len(fft_bins)], label='FFT Magnitude')
                    plt.plot(np.linspace(0, peak_freq + 1000, 500), gaussian(np.linspace(0, peak_freq + 1000, 500), *popt), color='red', label='Gaussian Fit')
                    plt.axvline(popt[1], color='green', linestyle='--', label='Mean')
                    plt.axvline(popt[1] + popt[2], color='orange', linestyle=':', label='Mean + Std Dev')
                    plt.axvline(popt[1] - popt[2], color='orange', linestyle=':', label='Mean - Std Dev')
                    plt.title(f'FFT Analysis: {file.name[:-4]}')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Normalized Power')
                    plt.legend()
                    plt.grid()

                    plot_path = os.path.join(output_folder, f"{file.name[:-4]}_FFT_Analysis.png")
                    plt.savefig(plot_path)
                    plt.close()

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    st.success("FFT analysis completed and plots saved.")
    st.session_state.step = "upload"
    st.rerun()

###############################################################################
# STEP 3: Upload to Cloud

elif st.session_state.step == "upload":
    st.title("Step 2: Upload FFT Results to FURTHRmind")

    upload_widget = FURTHRmind(id="experiment_upload")
    upload_widget.container_category = "experiment"
    upload_widget.file_extension = "png"
    upload_widget.select_container()

    project_details = upload_widget.setup_project()
    file_loader = FileLoader(project_details.host, project_details.api_key)

    # Button to trigger the upload process
    if st.button("Upload Results"):
        if upload_widget.selected:
            uploaded_count = 0
            experiment = upload_widget.selected.get()  # Retrieve the full Experiment object

            for file_name in os.listdir(st.session_state.output_folder):
                print(file_name)
                if file_name.endswith("_FFT_Analysis.png"):
                    local_path = os.path.join(st.session_state.output_folder, file_name)
                    print(local_path)
                    try:
                        experiment.add_file(local_path)  # ✅ pass path directly!
                        uploaded_count += 1
                    except Exception as e:
                        st.error(f"Upload failed for {file_name}: {e}")
                    
            if uploaded_count:
                st.success(f"Uploaded {uploaded_count} FFT result images to the experiment.")

                # Clean up the local result folder
                for file_name in os.listdir(st.session_state.output_folder):
                    os.remove(os.path.join(st.session_state.output_folder, file_name))
                try:
                    os.rmdir(st.session_state.output_folder)
                    st.info("Local result folder cleaned up.")
                except Exception as e:
                    st.warning(f"Cleanup issue: {e}")
            else:
                st.warning("No PNG images were uploaded.")
        else:
            st.warning("Please select a destination experiment.")
