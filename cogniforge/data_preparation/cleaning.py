# cogniforge/data_preparation/cleaning.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional  # Add this line to import typing tools
import streamlit as st
from cogniforge.algorithms.autotsad_wrapper import AutoTSADWrapper

class DataCleaner:
    """
    Enhanced data cleaning system that uses AutoTSAD for sophisticated anomaly detection
    in LDS process data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaned_df = None
        # Initialize our AutoTSAD wrapper
        self.anomaly_detector = AutoTSADWrapper()

    def clean_data(self) -> Optional[pd.DataFrame]:
        """
        Main cleaning method that combines AutoTSAD's anomaly detection with
        traditional cleaning techniques.
        """
        st.write("### Configure Cleaning Parameters")

        # Create an intuitive interface for parameter selection
        col1, col2 = st.columns(2)

        with col1:
            columns_to_clean = st.multiselect(
                "Select process parameters to clean",
                options=self.df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
                default=self.df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
                help="Choose which process parameters need cleaning"
            )

            interpolation_method = st.selectbox(
                "Gap filling method",
                options=['linear', 'polynomial', 'spline'],
                help="How to fill gaps after removing anomalies"
            )

        with col2:
            anomaly_threshold = st.slider(
                "Anomaly detection sensitivity",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                help="Higher values mean stricter anomaly detection"
            )

            smooth_window = st.slider(
                "Smoothing window size",
                min_value=3,
                max_value=21,
                value=5,
                step=2,
                help="Larger windows give smoother results"
            )

        if st.button("Clean Data"):
            cleaned_df = self.df.copy()
            cleaning_stats = {}

            for column in columns_to_clean:
                # Get the process parameter data
                series = cleaned_df[column]

                # Detect anomalies using AutoTSAD's ensemble approach
                anomalies = self.anomaly_detector.detect_anomalies(
                    series,
                    threshold=anomaly_threshold
                )

                # Get individual detector contributions for analysis
                detector_scores = self.anomaly_detector.get_detector_contributions(series)

                # Replace anomalies with NaN for interpolation
                series[anomalies] = np.nan

                # Fill the gaps left by removed anomalies
                series = series.interpolate(method=interpolation_method)

                # Apply final smoothing to reduce noise
                series = series.rolling(
                    window=smooth_window,
                    center=True,
                    min_periods=1
                ).mean()

                # Update the data
                cleaned_df[column] = series

                # Record cleaning statistics
                cleaning_stats[column] = {
                    'anomalies_removed': np.sum(anomalies),
                    'percent_anomalous': (np.sum(anomalies) / len(anomalies)) * 100
                }

            # Show cleaning results
            self._display_cleaning_results(cleaning_stats, detector_scores)

            return cleaned_df

        return None

    def _display_cleaning_results(self, stats: Dict, detector_scores: Dict):
        """
        Presents the cleaning results in an informative way.
        """
        st.write("### Cleaning Results")

        for column, column_stats in stats.items():
            st.write(f"Parameter: {column}")
            st.write(f"- Anomalies removed: {column_stats['anomalies_removed']}")
            st.write(f"- Percent anomalous: {column_stats['percent_anomalous']:.2f}%")

            # Show how different detectors contributed
            st.write("Detector Contributions:")
            for detector, scores in detector_scores.items():
                st.write(f"- {detector}: {np.mean(scores):.3f} average anomaly score")