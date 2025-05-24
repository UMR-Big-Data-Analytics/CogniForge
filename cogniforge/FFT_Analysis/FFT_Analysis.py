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

# Ensure latest_fft is initialized in session state
if "latest_fft" not in st.session_state:
    st.session_state["latest_fft"] = {
        "experiment_name": None,
        "output_folder": None
    }

st.title("FFT Analysis")

csv_widget = FURTHRmind(id="experiment")
csv_widget.container_category = "experiment"
csv_widget.file_extension = "csv"
csv_widget.select_container()

if st.button("Perform Analysis"):
    if csv_widget.selected:
        experiment = csv_widget.selected.get()
        experiment_name = experiment.name
        output_folder = os.path.join("C:\\FFT_Results", experiment_name)
        os.makedirs(output_folder, exist_ok=True)

        csv_files = [file for file in experiment.files if file.name.endswith('.csv')]
        if not csv_files:
            st.warning(f"No CSV files in {experiment_name}")
        else:
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

            # Update session state after FFT analysis
            st.session_state["latest_fft"].update({
                "experiment_name" : experiment_name,
                "output_folder" : output_folder
            })
            st.success(f"FFT done for: {experiment_name}")
    else:
        st.warning("Please select an experiment")

# Show upload section only if a recent FFT analysis exists
if st.session_state["latest_fft"]["experiment_name"]:
    with st.expander("🔼 Upload FFT Results to FURTHRmind", expanded=True):
        st.subheader(f"Upload Result for: `{st.session_state['latest_fft']['experiment_name']}`")

        upload_widget = FURTHRmind(id="experiment_upload")
        upload_widget.container_category = "experiment"
        upload_widget.file_extension = "png"
        upload_widget.select_container()

        if st.button("Upload Results"):
            if upload_widget.selected:
                upload_exp = upload_widget.selected.get()
                uploaded_count = 0
                output_folder = st.session_state["latest_fft"]["output_folder"]

                for fname in os.listdir(output_folder):
                    if fname.endswith("_FFT_Analysis.png"):
                        fpath = os.path.join(output_folder, fname)
                        try:
                            upload_exp.add_file(fpath)
                            uploaded_count += 1
                        except Exception as e:
                            st.error(f"Failed to upload {fname}: {e}")

                if uploaded_count:
                    st.success(f"Uploaded {uploaded_count} FFT result(s).")

                    # Clean up
                    for f in os.listdir(output_folder):
                        os.remove(os.path.join(output_folder, f))
                    os.rmdir(output_folder)

                    # Reset session
                    st.session_state["latest_fft"] = {
                        "experiment_name": None,
                        "output_folder": None
                    }
                    st.info("Local results cleaned after upload.")
            else:
                st.warning("Please select a destination experiment.")
