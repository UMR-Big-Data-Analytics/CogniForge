import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from typing import Dict, Any
from datetime import datetime


def initialize_session_state():
    """Initialize required session state variables if they don't exist."""
    if 'detrend_steps' not in st.session_state:
        st.session_state.detrend_steps = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'detrend_stats' not in st.session_state:
        st.session_state.detrend_stats = {}
    if 'column_params' not in st.session_state:
        st.session_state.column_params = {}
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'detrending_active' not in st.session_state:
        st.session_state.detrending_active = False


def generate_detrend_column_name(original_column: str) -> str:
    """Append '_detrended' to the original column name."""
    return f"{original_column}_det"

def detect_trend(data: np.ndarray, timestamps: np.ndarray) -> dict:
    """Detect linear trend using linear regression."""
    slope, _, r_value, p_value, _ = stats.linregress(timestamps, data)
    return {'slope': slope, 'r_squared': r_value**2, 'p_value': p_value}

def record_detrend_step(column: str, method: str, params: dict):
    """Track detrending"""
    step = {
        "timestamp": datetime.now(),
        "column": column,
        "method": method,
        "params": params,
        "result_column": generate_detrend_column_name(column),
    }
    st.session_state.detrend_steps.append(step)
    history_message = f"Detrending applied to {column} )"
    if not any(history_message in existing_message for existing_message in st.session_state.analysis_history):
        st.session_state.analysis_history.append(history_message)

def detrend_moving_average(data, window):
    """Remove trend using centered moving average."""
    trend = pd.Series(data).rolling(window=window, center=True).mean()
    trend = trend.fillna(method='bfill').fillna(method='ffill').values
    detrended = data - trend
    return detrended, trend


def detrend_linear(data, time):
    """Remove linear trend using polynomial fitting of degree 1"""
    line_coefficients = np.polyfit(time, data, 1)
    trend_line = np.polyval(line_coefficients, time)
    detrended_data = data - trend_line
    return detrended_data, trend_line


def detrend_polynomial(data, time, degree):
    """Remove polynomial trend of specified degree."""
    curve_coefficients = np.polyfit(time, data, degree)
    trend_curve = np.polyval(curve_coefficients, time)
    detrended_data = data - trend_curve
    return detrended_data, trend_curve

def analyze_detrend(df: pd.DataFrame = None) -> pd.DataFrame:
    initialize_session_state()

    if df is not None:
        st.session_state.current_df = df.copy()

    if st.session_state.current_df is None:
        st.error("Please load data first using the Load Data Page.")
        return None

     # Display current dataset info
    df = st.session_state.current_df
    dataset_name = st.session_state.get('current_dataset_name', 'Unnamed Dataset')
    st.markdown(f"**Dataset Name:** {dataset_name}")
    actual_rows = len(df)
    st.write(f"Using a dataset with {actual_rows:,} rows")

    df = st.session_state.current_df
    chosen_columns = st.multiselect("Choose columns to analyze", options=df.columns[1:])

    if not chosen_columns:
        st.info("Please choose at least one column to analyze.")
        return df

    detrend_method = st.selectbox(
        "Choose detrending method", options=["Linear", "Polynomial", "Moving Average"]
    )

    params = {}
    if detrend_method == "Polynomial":
        params['degree'] = st.slider("Polynomial degree", 2, 10, 2)
    elif detrend_method == "Moving Average":
        params['window'] = st.slider("Window size", 3, 101, 7, 2)

    timestamps = np.arange(len(df))
    needs_detrending = {}

    tabs = st.tabs([f"Analysis for {col}" for col in chosen_columns])

    for tab, chosen_column in zip(tabs, chosen_columns):
        with tab:
            st.write(f"### Analysis for {chosen_column}")
            data = df[chosen_column].astype(float).values
            trend_stats = detect_trend(data, timestamps)

            if trend_stats['r_squared'] > 0.1 and trend_stats['p_value'] < 0.05:
                st.warning("⚠️ Trend detected - Detrending recommended")
                needs_detrending[chosen_column] = True
            else:
                st.success("✓ No significant trend detected - No detrending required")
                needs_detrending[chosen_column] = False

            st.write("#### Trend Statistics")
            st.dataframe(pd.DataFrame({
                'Metric': ['R-squared', 'P-value', 'Slope'],
                'Value': [f"{trend_stats['r_squared']:.4f}",
                          f"{trend_stats['p_value']:.4f}",
                          f"{trend_stats['slope']:.6f}"]
            }), use_container_width=True)

            if needs_detrending[chosen_column] and st.button(f"Apply Detrending to {chosen_column}", key=f"btn_{chosen_column}", type="primary"):
                data = df[chosen_column].astype(float).values
                detrended, trend = (
                    detrend_linear(data, timestamps) if detrend_method == "Linear" else
                    detrend_polynomial(data, timestamps, params['degree']) if detrend_method == "Polynomial" else
                    detrend_moving_average(data, params['window'])
                )

                result_column = generate_detrend_column_name(chosen_column)
                df[result_column] = detrended
                record_detrend_step(chosen_column, detrend_method, params)

                st.success(f"✅ Detrending applied to {chosen_column}")
                st.session_state.current_df = df.copy()

    st.write("### Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)
    return df