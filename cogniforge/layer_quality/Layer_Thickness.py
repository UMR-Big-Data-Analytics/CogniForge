import datetime as dt
import io
import os
import shutil
import time
import sys

import lttb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from cogniforge.utils.furthr import FURTHRmind, download_item_bytes

sys.path.append("../cogniforge")
from cogniforge.algorithms.anomaly_detection import (
    AnomalyDetector,
    AutoTSADAnomalyDetector,
    IsolationForestAnomalyDetector,
    KMeansAnomalyDetector,
)
from cogniforge.utils.dataloader import DataLoader
from cogniforge.utils.furthr import FURTHRmind
from cogniforge.utils.object_select_box import selectbox
from cogniforge.utils.plotting import plot_sampled, plot_xy_chart
from cogniforge.utils.state_button import button
import config

def save_univariate_time_series(df: pd.DataFrame, selected_column: str, output_path: str):
    if df.index.name is not None:
        df.reset_index(inplace=True)
    zeit_column = next(
    (col for col in df.columns if col.lower().startswith(("Zeit", "Time"))), 
    df.columns[0]  # Default to the first column if no match is found
    )
    univariate_df = pd.DataFrame({
        "timestamp": df[zeit_column],  # Generate sequential timestamps
        f"value-0": df[selected_column]
    })

    # Save to CSV
    univariate_df.to_csv(output_path, index=False, header=True)

def plot_anomaly_detection(anomaly_score: np.ndarray,threshold: float = 0.6) -> np.ndarray:
    zeit_column = next(
    (col for col in df.columns if col.lower().startswith(("Zeit", "Time"))), 
    df.columns[0]  # Default to the first column if no match is found
    )

    # Extract the Zeit column values
    zeit_values = df[zeit_column].values.astype(np.float32)[: len(anomaly_score)]
    if len(anomaly_score) > 1000:
        anomaly_score_downsampled = lttb.downsample(
            np.c_[
                zeit_values,
                anomaly_score,
            ],
            1000,
        )
    else: 
        anomaly_score_downsampled = np.c_[zeit_values, anomaly_score]
    df_plot = pd.DataFrame(
        {
            "Zeit": anomaly_score_downsampled[:, 0],
            "Anomaly Score": anomaly_score_downsampled[:, 1],
        }
    ).set_index("Zeit")

    # Categorize points based on threshold
    df_plot["Color"] = np.where(df_plot["Anomaly Score"] > threshold,f"Above {threshold}", f"Below {threshold}")

    # Plot with Plotly
    fig = px.scatter(df_plot.reset_index(), x="Zeit", y="Anomaly Score", color="Color",
                     color_discrete_map={f"Above {threshold}": "red", f"Below {threshold}": "blue"},
                     title="Anomaly Score Visualization")

    # Display in Streamlit
    st.plotly_chart(fig)

    return anomaly_score

def run_anomaly_analysis(df, filtered_columns, filename, mode="Single File"):
    if mode == "Single File":
        available_algorithms = [KMeansAnomalyDetector(), AutoTSADAnomalyDetector(), IsolationForestAnomalyDetector()]
    else:  # Batch Mode
        available_algorithms = [KMeansAnomalyDetector(), IsolationForestAnomalyDetector()]

    algorithm: AnomalyDetector = selectbox(
        available_algorithms,
        format_name=lambda a: a.name(),
        label=f"Choose an anomaly detector for {filename}",
        key=f"algorithm_{filename}"
    )

    if isinstance(algorithm, AutoTSADAnomalyDetector) and mode == "Batch Folder":
        st.warning("AutoTSAD is not available in batch mode.")
        return

    if isinstance(algorithm, AutoTSADAnomalyDetector):
        column_name = st.selectbox("Select a column for AutoTSAD", filtered_columns, key=f"col_{filename}")
        if button(f"Run {algorithm.name()} on {filename}", f"run_autotsad_{filename}", True):
            temp_csv_path = f"{filename}_temp_univariate.csv"
            save_univariate_time_series(df, column_name, temp_csv_path)
            scores = algorithm.detect(temp_csv_path)
            show_and_upload_anomaly(scores, filename, algorithm.name())
    else:
        column_names = st.multiselect(
            f"Choose columns for {filename}", filtered_columns, default=filtered_columns, key=f"cols_{filename}"
        )
        if button(f"Run {algorithm.name()} on {filename}", f"run_algorithm_{filename}", True):
            try:
                scores = algorithm.detect(df[column_names].values)
                show_and_upload_anomaly(scores, filename, algorithm.name())
            except Exception as e:
                st.error(f"Error running {algorithm.name()} on {filename}: {e}")


def show_and_upload_anomaly(scores, filename, algorithm_name):
    threshold = st.slider("Set anomaly threshold", 0.0, 1.0, 0.6, key=f"threshold_{filename}")
    plot_anomaly_detection(scores, threshold)
    with st.expander("Upload Anomaly Score", expanded=True):
        try:
            FURTHRmind("upload").upload_csv(
                pd.DataFrame({"score": scores, "is_anomaly": scores > threshold}),
                f"{filename}-anomaly_score-{algorithm_name}-{dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
            )
        except Exception as upload_error:
            st.error(f"Upload error: {str(upload_error)}")
            st.exception(upload_error)


#st.set_page_config(page_title="CogniForge | Layer Thickness", page_icon="💦")
st.write("# Layer Thickness")
st.write(
    "Welcome to the Layer Thickness tool. Here you can analyze and visualize the thickness of your layer"
)
st.write(
    f"Upload your data to the [FURTHRmind]({config.furthr['host']}) database. Then, here, you can choose the correct dataset and let our algorithms tell you the quality of your layer."
)

if "latest_results" not in st.session_state:
    st.session_state["latest_results"] = {
        "experiment_name" : None,
        "output_folder" : None
    }

if "fileName" not in st.session_state:
    st.session_state.fileName = None
if "data" not in st.session_state:
    st.session_state.data = None
if "files_loaded" not in st.session_state:
    st.session_state.files_loaded = False 
with st.status("Download Data from FURTHRmind", expanded=True):
    mode = st.radio("Select Mode", ["Single File", "Batch Folder"],horizontal=True)
    if mode == "Single File":
        downloader = FURTHRmind("download")
        downloader.file_extension = "csv"
        downloader.select_file()
        st.session_state.data = None
        data, filename = downloader.download_string_button() or (None, None)
        print(data)
        st.session_state.data = data
        st.session_state.fileName = filename
    elif mode == "Batch Folder":
        st.session_state.data = None
        st.session_state.fileName = None
        folder_widget = FURTHRmind(id="experiment")
        folder_widget.container_category = "experiment"
        folder_widget.file_extension = "csv"
        folder_widget.select_container()

        experiment = folder_widget.selected.get()
        if experiment:
            csv_files = [file for file in experiment.files if file.name.endswith('.csv')]
            if csv_files:
                st.markdown(f"**📂{len(experiment.files)} CSV files found in the {experiment.name} folder.**")
                st.session_state.files_loaded = experiment
            else:
                st.warning(f"No CSV files in {experiment.name}")

        

if "anomalyNameChanged" not in st.session_state:
    st.session_state.anomalyNameChanged = False
if "anomaly_score" not in st.session_state:
    st.session_state.anomaly_score = None
    st.session_state.anomaly_algorithm = None
if mode=="Single File" and st.session_state.data is not None:
    st.success("Data downloaded!")
    st.session_state.fileName = filename
    print(st.session_state.data)
    tabs = st.tabs(["Data Preview", "Plot Data", "Analysis"])
    with tabs[0]:
        dl = DataLoader(csv=st.session_state.data)
        df = dl.get_processedDataFrame()

    if st.session_state.data is not None and df is not None:
        filtered_columns = [col for col in df.columns if not col.lower().startswith(("zeit", "time"))]
        with tabs[1]:
                 plot_xy_chart(df)

        with tabs[2]:
            run_anomaly_analysis(df, filtered_columns, st.session_state.fileName, mode="Single File")

elif mode == "Batch Folder" and st.session_state.files_loaded is not None:
    
    experiment = st.session_state.files_loaded
    experiment_name = experiment.name
    # Prepare once
    batch_dataframes = {}
    filtered_columns_dict = {}

    # Select algorithm
    available_algorithms = [KMeansAnomalyDetector(), IsolationForestAnomalyDetector()]
    algorithm: AnomalyDetector = selectbox(
        available_algorithms,
        format_name=lambda a: a.name(),
        label="Choose an anomaly detector for all files",
        key="batch_algorithm_selector"
    )

    threshold = st.slider("Set anomaly threshold", 0.0, 1.0, 0.6, key=f"threshold_")

    # Load all data first
    for file in experiment.files:
        try:
            csv_bytes, _ = download_item_bytes(file)
            csv_text = csv_bytes.getvalue().decode("utf-8")
            df = pd.read_csv(io.StringIO(csv_text), delimiter=';',decimal=',',header=2)

            # dl = DataLoader(csv=csv_io)
            # df = dl.get_processedDataFrame()

            if df is not None:
                batch_dataframes[file.name] = df
                filtered_columns = [col for col in df.columns if not col.lower().startswith(("zeit", "time"))]
                filtered_columns_dict[file.name] = filtered_columns
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    # Only show analysis options after loading
    st.subheader("Batch Analysis ")
    
    if st.button("Run Batch Anomaly Detection"):
        # Define output folder for saving results
        output_folder = os.path.join("C:\\Anomaly_Results", experiment.name)
        os.makedirs(output_folder, exist_ok=True)

        # 🧹 Clean up any old files in the output folder before starting a new run
        for old_file in os.listdir(output_folder):
            old_path = os.path.join(output_folder, old_file)
            try:
                os.remove(old_path)
            except Exception as e:
                st.warning(f"Failed to delete old file {old_file}: {e}")
        
        progress_bar = st.progress(0)
        total_files = len(experiment.files)
        
        for i, file in enumerate(experiment.files):
            try:
                df = batch_dataframes.get(file.name)
                if df is None:
                    continue
                    
                filtered_columns = filtered_columns_dict.get(file.name, [])
                
                # Calculate the anomaly scores
                scores = algorithm.detect(df[filtered_columns].values)
                
                # Create a DataFrame with scores and anomalies
                result_df = pd.DataFrame({
                    "score": scores,
                    "is_anomaly": scores > threshold
                })

                # Define file path to save the anomaly results
                save_path = os.path.join(
                    output_folder,
                    f"{experiment.name}-anomaly_score-{algorithm.name()}-{file.name}-{dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv"
                )
                
                # Save the result to a CSV file
                result_df.to_csv(save_path, index=False)
                
                st.success(f"Processed {file.name} -> {save_path}")
                progress_bar.progress((i + 1) / total_files)

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
        
        st.session_state["latest_results"].update({
            "experiment_name" : experiment_name,
            "output_folder" : output_folder
        })

    if st.session_state["latest_results"]["experiment_name"]:
        with st.expander("🔼 Upload Anomaly results to FURTHRmind", expanded=True):
            st.subheader(f"Upload Result for: `{st.session_state['latest_results']['experiment_name']}`")

            upload_widget = FURTHRmind(id="experiment_upload")
            upload_widget.container_category = "experiment"
            upload_widget.file_extension = "csv"
            upload_widget.select_container()

            if st.button("Upload Results"):
                if upload_widget.selected:
                    upload_exp = upload_widget.selected.get()
                    uploaded_count = 0
                    output_folder = st.session_state["latest_results"]["output_folder"]

                    for fname in os.listdir(output_folder):
                        if fname.startswith(f"{experiment_name}-anomaly_score"):
                            fpath = os.path.join(output_folder, fname)
                            try:
                                upload_exp.add_file(fpath)
                                uploaded_count += 1
                            except Exception as e:
                                st.error(f"Failed to upload {fname}: {e}")

                    if uploaded_count:
                        st.success(f"Uploaded {uploaded_count} Anomaly result(s).")

                        # Clean up
                        for f in os.listdir(output_folder):
                            os.remove(os.path.join(output_folder, f))
                        os.rmdir(output_folder)

                        # Reset session
                        st.session_state["latest_results"] = {
                            "experiment_name": None,
                            "output_folder": None
                        }
                        st.info("Local results cleaned after upload.")
                    else:
                        st.warning("Please select a destination experiment.")

