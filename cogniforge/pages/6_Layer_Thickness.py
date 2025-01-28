import datetime as dt
import sys

import lttb
import numpy as np
import pandas as pd
import streamlit as st

if __name__ == '__main__':
    sys.path.append("../cogniforge")
    from cogniforge.algorithms.anomaly_detection import (
        AnomalyDetector,
        AutoTSADAnomalyDetector,
        KMeansAnomalyDetector,
    )
    from cogniforge.utils.dataloader import DataLoader
    from cogniforge.utils.furthr import FURTHRmind
    from cogniforge.utils.object_select_box import selectbox
    from cogniforge.utils.plotting import plot_sampled
    from cogniforge.utils.state_button import button

    def save_univariate_time_series(df: pd.DataFrame, selected_column: str, output_path: str):
        if df.index.name is not None:
            df.reset_index(inplace=True)

        univariate_df = pd.DataFrame({
            "timestamp": range(len(df)),  # Generate sequential timestamps
            f"value-0": df[selected_column]
        })

        # Save to CSV
        univariate_df.to_csv(output_path, index=False, header=True)

    def run_anomaly_detection(algorithm: AnomalyDetector, data: np.ndarray) -> np.ndarray:
        anomaly_score = algorithm.detect(data)
        anomaly_score_downsampled = lttb.downsample(
            np.c_[
                np.arange(len(anomaly_score)).astype(np.float32),
                anomaly_score,
            ],
            1000,
        )
        df_plot = pd.DataFrame(
            {
                "Index": anomaly_score_downsampled[:, 0],
                "Anomaly Score": anomaly_score_downsampled[:, 1],
            }
        ).set_index("Index")
        st.line_chart(df_plot)
        return anomaly_score


    st.set_page_config(page_title="CogniForge | Layer Thickness", page_icon="💦")
    st.write("# Layer Thickness")
    st.write(
        "Welcome to the Layer Thickness tool. Here you can analyze and visualize the thickness of your layer"
    )
    st.write(
        "Upload your data to the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database. Then, here, you can choose the correct dataset and let our algorithms tell you the quality of your layer."
    )

    with st.expander("Download Data from FURTHRmind"):
        data = FURTHRmind("download").download_csv()

    if data is not None:
        with st.expander("Parse Data"):
            dl = DataLoader(csv=data)
            df = dl.get_dataframe()

    if data is not None and df is not None:
        with st.expander("Plot Data"):
            if button("Plot data", "plot_data_layer", True):
                plot_sampled(df)

        with st.expander("Analysis"):
            # Allow column selection dynamically
            algorithm: AnomalyDetector = selectbox(
                [KMeansAnomalyDetector(), AutoTSADAnomalyDetector()],
                format_name=lambda a: a.name(),
                label="Choose an anomaly detector",
            )
            if algorithm is not None:
                algorithm.parameters()

                if isinstance(algorithm, AutoTSADAnomalyDetector):
                    # For AutoTSAD: Select a single column for univariate time series
                    column_name = st.selectbox("Select a column for AutoTSAD", df.columns)
                    if column_name:
                        st.write(f"Selected column: {column_name}")
                        if button(f"Run {algorithm.name()}", "run_autotsad", True):
                            temp_csv_path = "temp_univariate_data.csv"
                            save_univariate_time_series(df, column_name, temp_csv_path)

                            anomaly_score = algorithm.detect(temp_csv_path)
                            anomaly_score_downsampled = lttb.downsample(
                                                        np.c_[np.arange(len(anomaly_score)).astype(np.float32),anomaly_score,],1000,
                                                            )
                            df_plot = pd.DataFrame({
                                                    "Index": anomaly_score_downsampled[:, 0],
                                                    "Anomaly Score": anomaly_score_downsampled[:, 1],
                                                    }
                                                    ).set_index("Index")
                            st.line_chart(df_plot)
                            st.write("## Upload Score")
                            FURTHRmind("upload").upload_csv(
                                    pd.DataFrame({"score": anomaly_score}),
                                    f"anomaly_score-{algorithm.name()}-{dt.datetime.now().isoformat()}",
                    )
                else:
                    # For other detectors: Allow multiselect for multiple columns
                    column_names = st.multiselect(
                        "Choose columns", list(df.columns), default=list(df.columns)
                    )
                    if button(f"Run {algorithm.name()}", "run_algorithm", True):
                        anomaly_score = run_anomaly_detection(
                            algorithm, df[column_names].values
                        )
                        st.write("## Upload Score")
                        FURTHRmind("upload").upload_csv(
                            pd.DataFrame({"score": anomaly_score}),
                            f"anomaly_score-{algorithm.name()}-{dt.datetime.now().isoformat()}",
                        )

