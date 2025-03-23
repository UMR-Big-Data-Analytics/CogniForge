import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from typing import Dict, Any
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress
from utils.session_state_management import update_session_state

# DETRENDING ANALYSIS FEATURES
# ====================================
# The code detects trends in time series data and offers methods to remove them using different techniques.
# TREND DETECTION METHODS:
# - Linear regression (R-squared, slope, p-value)
# - Augmented Dickey-Fuller test for stationarity
# - Autocorrelation
# DETRENDING METHODS:
# - Linear detrending: Removes first-order polynomial trend using division and normalization
# - Moving average detrending: Removes local trends using centered rolling window
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
    if 'downsample_steps' not in st.session_state:
        st.session_state.downsample_steps = []
    if 'show_plots' not in st.session_state:
        st.session_state.show_plots = False


def detect_trend(data: np.ndarray, timestamps: np.ndarray) -> dict:
    """Detect trends using statistical tests."""
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, data)
    r_squared = r_value ** 2
    direction = "⬆ Increasing" if slope > 0 else "⬇ Decreasing"
    # Stationarity test
    adf_p_value = adfuller(data)[1]
    r2_threshold = 0.2 # If R2 > 0.2, trend is considered significant. Change if needed 0.2 is pretty generous
    stats_dict = {
        'r_squared': r_squared,
        'slope': slope,
        'p_value': p_value,
        'direction': direction,
        'adf_p_value': adf_p_value,
        'needs_detrending': r_squared > r2_threshold and adf_p_value > 0.05
    }
    return stats_dict


def recommend_detrend_method(data, timestamps):
    # Autocorrelation: check how similar a time series at time t is similar to it at t-1
    # It tells basically if past value can predict future values.
    autocorrelation = np.corrcoef(data[:-1], data[1:])[0, 1]
    trend_stats = detect_trend(data, timestamps)
    if trend_stats['r_squared'] > 0.8:
        return {"method": "Linear"}
    else:
        # Current threshold = 0.5. Change if needed
        if abs(autocorrelation) > 0.5:
            return {"method": "Moving Average"}
        else:
            return {"method": "Linear"}

def create_trend_plots(data, df, trend=None, detrended=None, title="", method=None, params=None):
    """ before and after detreding"""
    time_col = [col for col in df.columns if 'Zeit' in col][0]
    timestamps = df[time_col].values
    time_unit = "seconds" if "Zeit[(s)]" in time_col else "milliseconds"
    # Title
    if method:
        if method == "Moving Average":
            title = f"{title} (Window Size: {params.get('window', 'N/A')})"
        elif method == "Linear":
            title = f"{title} (Linear Detrending)"

    if detrended is not None:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Data with Trend', 'Detrended Data'),
            vertical_spacing=0.25
        )
        # set legend position!
        legend_settings = dict(orientation="h",y=0.475,yanchor="middle",xanchor="center",x=0.5)
    else:
        fig = make_subplots(rows=1, cols=1)
        legend_settings = dict(orientation="h", y=-0.25, yanchor="top", xanchor="center", x=0.5)
    fig.add_trace(
        go.Scatter(x=timestamps, y=data, name='Original Data', mode='lines'),
        row=1, col=1
    )
    # add trend
    if trend is not None:
        fig.add_trace(
            go.Scatter(x=timestamps, y=trend, name='Trend', mode='lines'),
            row=1, col=1
        )

    # Add detrended data
    if detrended is not None:
        fig.add_trace(
            go.Scatter(x=timestamps, y=detrended, name='Detrended Data',
                       mode='lines', line=dict(color='orange')),
            row=2, col=1
        )
    fig.update_xaxes(title_text=f"Time ({time_unit})")
    fig.update_yaxes(title_text="Value")

    # Configure plot
    fig.update_layout(
        height=600 if detrended is not None else 400,
        title_text=title,
        showlegend=True,
        legend=legend_settings,
    )
    return fig

def record_detrend_step(column: str, method: str, params: dict):
    """Track detrending steps."""
    timestamp = datetime.now()
    step = {
        "timestamp": timestamp,
        "column": column,
        "method": method,
        "params": params,
        "result_column": f"{column}_detrended",
    }
    st.session_state.detrend_steps.append(step)
    # Create history message with timestamp
    history_message = f"[{timestamp.strftime('%Y-%m-%d %H:%M')}] Detrending applied to {column}, method = {method}"
    if not any(history_message in existing_message for existing_message in st.session_state.analysis_history):
        st.session_state.analysis_history.append(history_message)


def recommend_moving_average_window(data):
    """Recommend a moving average window based on the data length and variability."""
    data_length = len(data)
    variability = np.std(data) / np.mean(data)  # Measure of variability compared to the mean

    if variability > 0.2:
        window = min(max(7, int(data_length * 0.1)), 101)
    else:
        window = 7
    return window

def detrend_moving_average(data, window):
    """Remove trend using centered moving average with optional window size."""
    if window is None:
        window = recommend_moving_average_window(data)  # Automatically recommend window if not provided
    trend = pd.Series(data).rolling(window=window, center=True).mean()
    trend = trend.fillna(method='bfill').fillna(method='ffill').values
    detrended = data - trend
    return detrended, trend


def detrend_linear(data, time):
    """Remove linear trend using polynomial fitting of degree 1"""
    line_coefficients = np.polyfit(time, data, 1)
    trend_line = np.polyval(line_coefficients, time)
    detrended_data = (data / trend_line) - 1
    return detrended_data, trend_line


def analyze_detrend(df: pd.DataFrame = None) -> pd.DataFrame:
    initialize_session_state()
    if df is None:
        df = st.session_state.current_df.copy() if 'current_df' in st.session_state and st.session_state.current_df is not None else None
        if df is None:
            st.error("Please load data first using the Load Data Page.")
            return None

    # ***Dataset Information - Moved and updated***
    st.write("#### Current Dataset Information")
    dataset_name = st.session_state.get('current_dataset_name', 'Unnamed Dataset')
    st.markdown(f"**Dataset Name:** {dataset_name}")
    st.write(f"Using a dataset with {len(st.session_state.current_df):,} rows")

    if st.session_state.analysis_history:
        with st.expander("📝 **Analysis History**", expanded=False):
            for message in st.session_state.analysis_history:
                st.write(message)
    chosen_columns = st.multiselect("Choose columns to analyze", options=df.columns[1:])

    if not chosen_columns:
        st.info("Please choose at least one column to analyze.")
        return df

    time_col = [col for col in df.columns if 'Zeit' in col][0]
    timestamps = df[time_col].values
    needs_detrending = {}

    # tabs for each chosen column
    tabs = st.tabs([f"Analysis for {col}" for col in chosen_columns])
    for tab, chosen_column in zip(tabs, chosen_columns):
        with tab:
            st.write(f"### Analysis for {chosen_column}")
            data = df[chosen_column].astype(float).values
            trend_stats = detect_trend(data, timestamps)
            if trend_stats['needs_detrending']:
                st.warning(f"⚠️ Trend detected - {trend_stats['direction']} - Detrending recommended")
                needs_detrending[chosen_column] = True
                # Recommendation
                recommendation = recommend_detrend_method(data, timestamps)
                detrend_method = st.selectbox(
                    "Choose detrending method",
                    options=["Linear", "Moving Average"],
                    key=f"detrend_method_{chosen_column}",
                    index=["Linear", "Moving Average"].index(recommendation['method'])
                )

            else:
                st.success("✓ No significant trend detected - No detrending required")
                needs_detrending[chosen_column] = False
                detrend_method = None

            # Parameters for detrending methods
            params = {}
            if needs_detrending.get(chosen_column, False):
                if detrend_method == "Moving Average":
                    default_window = recommend_moving_average_window(data)
                    st.info(f"""
                    **Recommended: Moving Average Detrending**:
                    - Cyclical patterns detected
                    - Recommended Window Size: {default_window}
                    """)
                    params['window'] = st.slider(
                        "Window size",
                        3, 101,
                        default_window,
                        key=f"moving_avg_window_{chosen_column}"
                    )
            with st.expander("🔍 Trend Statistics"):
                # Compute trend statistics for original data
                original_stats_df = pd.DataFrame({
                    'Metric': ['R-squared', 'P-value', 'ADF P-value'],
                    'Value': [f"{trend_stats['r_squared']:.4f}",
                              f"{trend_stats['p_value']:.4f}",
                              f"{trend_stats['adf_p_value']:.4f}"]
                })
                # Explain stats
                stat_info = """
                Trend Statistics Interpretation:
                - **R-squared**: Indicates trend strength (0-1)
                  - Close to 0: Weak trend
                  - Close to 1: Strong trend
                - **P-value**: Trend significance
                  - <0.05: Statistically significant trend
                  - \>0.05: No significant trend
                - **ADF P-value**: Stationarity test
                  - \>0.05: Non-stationary (trend present)
                  - <0.05: Stationary (no trend)
                """
                st.info(stat_info)
                #  detrended statistics
                if needs_detrending.get(chosen_column, False):
                    if detrend_method == "Linear":
                        detrended, trend = detrend_linear(data, timestamps)
                    elif detrend_method == "Moving Average":
                        detrended, trend = detrend_moving_average(data, params.get('window', 7))
                    # Trend statistics for detrended data
                    detrended_stats = detect_trend(detrended, timestamps)
                    detrended_stats_df = pd.DataFrame({
                        'Metric': ['R-squared', 'P-value', 'ADF P-value'],
                        'Value': [f"{detrended_stats['r_squared']:.4f}",
                                  f"{detrended_stats['p_value']:.4f}",
                                  f"{detrended_stats['adf_p_value']:.4f}"]
                    })

                    # Before and After Original vs Detrended
                    st.markdown("##### Trend Analysis: Before and After Detrending")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Original Data")
                        st.dataframe(original_stats_df, use_container_width=True, hide_index=True)
                    with col2:
                        st.markdown("##### Detrended Data")
                        st.dataframe(detrended_stats_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(original_stats_df, use_container_width=True, hide_index=True)

            # Visualization Button
            st.caption("Warning: Plotting a large dataset may take time or affect UI performance.")
            if st.button("Plot Data", key=f"plot_btn_{chosen_column}", type="primary"):
                with st.expander("📊 Visualization", expanded=True):
                    data = df[chosen_column].astype(float).values
                    detrended = None
                    trend = None
                    if needs_detrending.get(chosen_column, False):
                        if detrend_method == "Linear":
                            detrended, trend = detrend_linear(data, timestamps)
                        elif detrend_method == "Moving Average":
                            detrended, trend = detrend_moving_average(data, params.get('window', 7))

                        plot_title = f"Trend Analysis Preview for {chosen_column}"
                    else:
                        plot_title = f"Original Data for {chosen_column}"

                    fig = create_trend_plots(data, df, trend, detrended, plot_title,
                                             method=detrend_method, params=params)
                    st.plotly_chart(fig, use_container_width=True)


            # Apply Detrending Button
            if needs_detrending.get(chosen_column, False):
                if st.button(f"Apply Detrending to {chosen_column}", key=f"btn_{chosen_column}", type="primary"):
                    with st.spinner('Applying detrending...'):
                        st.session_state.current_df = df.copy()
                        data = df[chosen_column].astype(float).values
                        detrended, trend = (
                            detrend_linear(data, timestamps) if detrend_method == "Linear" else
                            detrend_moving_average(data, params['window'])
                        )
                        result_column = f"{chosen_column}_detrended"
                        df[result_column] = detrended
                        record_detrend_step(chosen_column, detrend_method, params)
                        update_session_state(df, analysis_type='detrend')
                        st.success(f"✅ Detrending applied to {chosen_column}")

    # Dataset Preview
    st.write("### Dataset Preview")
    preview_df = st.session_state.current_df.copy()
    display_df = preview_df.copy().head(15)
    numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notnull(x) else x)
        display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True, height=350)
    return st.session_state.current_df