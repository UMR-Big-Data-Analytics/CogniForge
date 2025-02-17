import datetime as dt
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

def plot_anomaly_detection(anomaly_score: np.ndarray) -> np.ndarray:
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
    df_plot["Color"] = np.where(df_plot["Anomaly Score"] > 0.6, "Above 60%", "Below 60%")

    # Plot with Plotly
    fig = px.scatter(df_plot.reset_index(), x="Zeit", y="Anomaly Score", color="Color",
                     color_discrete_map={"Above 60%": "red", "Below 60%": "blue"},
                     title="Anomaly Score Visualization")

    # Display in Streamlit
    st.plotly_chart(fig)

    return anomaly_score


#st.set_page_config(page_title="CogniForge | Layer Thickness", page_icon="💦")
st.write("# Layer Thickness")
st.write(
    "Welcome to the Layer Thickness tool. Here you can analyze and visualize the thickness of your layer"
)
st.write(
    "Upload your data to the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database. Then, here, you can choose the correct dataset and let our algorithms tell you the quality of your layer."
)

if "fileName" not in st.session_state:
    st.session_state.fileName = None
if "data" not in st.session_state:
    st.session_state.data = None
with st.status("Download Data from FURTHRmind", expanded=True):
    data,filename = FURTHRmind("download").download_csv() or (None, None)
if "anomalyNameChanged" not in st.session_state:
    st.session_state.anomalyNameChanged = False
if "anomaly_score" not in st.session_state:
    st.session_state.anomaly_score = None
    st.session_state.anomaly_algorithm = None
if st.session_state.data is not None:
    st.session_state.data.name = filename
    tabs = st.tabs(["Data Preview", "Plot Data", "Analysis"])
    with tabs[0]:
        dl = DataLoader(csv=st.session_state.data)
        df = dl.get_processedDataFrame()

    if st.session_state.data is not None and df is not None:
        filtered_columns = [col for col in df.columns if not col.lower().startswith(("zeit", "time"))]
        with tabs[1]:
                 plot_xy_chart(df)

        with tabs[2]:
            # Allow column selection dynamically
            algorithm: AnomalyDetector = selectbox(
                [KMeansAnomalyDetector(), AutoTSADAnomalyDetector() , IsolationForestAnomalyDetector()],
                format_name=lambda a: a.name(),
                label="Choose an anomaly detector",
            )
            if "prev_algorithm" not in st.session_state:
                st.session_state.prev_algorithm = None 
            if algorithm.name() != st.session_state.prev_algorithm:
                st.session_state.anomalyNameChanged = True
                st.session_state.anomaly_score = None
                st.session_state.anomaly_algorithm = None
                st.session_state.prev_algorithm = algorithm.name()
            if algorithm is not None:
                algorithm.parameters()
                if "anomaly_score" not in st.session_state:
                    st.session_state.anomaly_score = None
                    st.session_state.anomaly_algorithm = None

                if isinstance(algorithm, AutoTSADAnomalyDetector):
                    column_name = st.selectbox("Select a column for AutoTSAD", filtered_columns)

                    if column_name:
                        st.write(f"Selected column: {column_name}")

                        # ✅ Run AutoTSAD when Button is Clicked
                        if button(f"Run {algorithm.name()}", "run_autotsad", True):
                            temp_csv_path = "temp_univariate_data.csv"
                            save_univariate_time_series(df, column_name, temp_csv_path)
                            st.session_state.anomaly_score = algorithm.detect(temp_csv_path) # Store anomaly score 
                            st.session_state.clicked["run_autotsad"] = False  
                            st.session_state.anomaly_algorithm = algorithm.name()
                            st.session_state.anomalyNameChanged = False
                            if(os.path.isdir("results-autotsad")):
                                shutil.rmtree("results-autotsad")
                else:
                    # For other detectors: Allow multiselect for multiple columns
                    column_names = st.multiselect(
                        "Choose columns", list(filtered_columns), default=list(filtered_columns)
                    )
                    if button(f"Run {algorithm.name()}", "run_algorithm", True):
                        anomaly_score = algorithm.detect(df[column_names].values)
                        st.session_state.clicked["run_algorithm"] = False
                        st.session_state.anomaly_score = anomaly_score  # Save anomaly score
                        st.session_state.anomaly_algorithm = algorithm.name()
                        st.session_state.anomalyNameChanged = False

if st.session_state.anomaly_score is not None and st.session_state.anomaly_algorithm is not None and not st.session_state.anomalyNameChanged:
    plot_anomaly_detection(st.session_state.anomaly_score)
    with st.expander("Upload Anomaly Score",expanded=True):
            FURTHRmind("upload").upload_csv(
                pd.DataFrame({"score": st.session_state.anomaly_score, "is_anomaly": st.session_state.anomaly_score > 0.6}),
                f"{st.session_state.fileName}-anomaly_score-{st.session_state.anomaly_algorithm}-{dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}",
            )