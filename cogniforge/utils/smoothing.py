import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
from typing import Dict, Any
from datetime import datetime
from utils.session_state_management import update_session_state
"""
SMOOTHING ANALYSIS FEATURE
==================================
Perform smoothing analysis on selected columns using exponential smoothing
Functions:
- Select columns to undergo smoothing
- Code check whether smoothing is needed
- Code suggests an initial smoothing factor based on variable's variability (calculated by calculate_column_volatility())
- User can adjust smoothing factor using a slider
- Enable visualization of before and after smoothing
- Add smoothing to analysis history if smoothing is performed
"""
def initialize_session_state():
    if 'downsample_steps' not in st.session_state:
        st.session_state.downsample_steps = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'smoothing_steps' not in st.session_state:
        st.session_state.smoothing_steps = []
    if 'smoothing_stats' not in st.session_state:
        st.session_state.smoothing_stats = {}


def calculate_column_volatility(column_data: np.ndarray) -> float:
    """Calculate volatility for a given column with more aggressive smoothing."""
    changes_between_points = np.diff(column_data)
    spread_of_changes = np.std(changes_between_points)
    spread_of_data = np.std(column_data)
    volatility = spread_of_changes / spread_of_data
    base_alpha = max(0.02, min(0.2, 0.5 - volatility))
    # Adjust based on outliers
    z_scores = np.abs(stats.zscore(column_data))
    outlier_ratio = np.mean(z_scores > 2.5)
    # more smoothing
    if outlier_ratio > 0.02:
        base_alpha *= 0.6

    return base_alpha

def check_smoothing_need(data: np.ndarray) -> bool:
        """Determine if smoothing is need
        Process:
        - Calualate rolling standard deviation over a small windoww &
        compare it to overall std   OR
        - Check for outlier > 5% (i.e. it creates noises)  OR
        - AVG of local change is higher than 10% mean
        """
        window = min(10, len(data) // 4)
        rolling_std = pd.Series(data).rolling(window=window).std()
        overall_std = np.std(data)
        z_scores = np.abs(stats.zscore(data))
        outlier_ratio = np.mean(z_scores >3.5)
        local_changes = np.abs(np.diff(data))
        signal_mean = np.mean(np.abs(data))
        test_alpha = 0.5
        # Extra check
        test_smoothed = exponential_smoothing(data, test_alpha)
        test_stats = calculate_smoothing_statistics(data, test_smoothed)
        noise_indicators = [
            rolling_std.mean() > 0.03 ** overall_std,
            outlier_ratio > 0.02,
            local_changes.mean() > 0.05 * signal_mean
        ]
        return any(noise_indicators) and calculate_smoothing_statistics(
            data, exponential_smoothing(data, 0.5)
        )['noise_reduction'] > 0.4


# Weighted moving average
def exponential_smoothing(data: np.ndarray, alpha: float, iterations: int = 1) -> np.ndarray:
    """Apply exponential smoothing with optional multiple passes."""
    smoothed = np.zeros(len(data))
    smoothed[0] = data[0]
    mask = np.isnan(data)
    current_data = np.where(mask, smoothed[0], data)
    for _ in range(iterations):
        for t in range(1, len(data)):
            if mask[t]:
                smoothed[t] = smoothed[t - 1]
            else:
                smoothed[t] = alpha * current_data[t] + (1 - alpha) * smoothed[t - 1]
        current_data = smoothed.copy()
    return smoothed

def calculate_smoothing_statistics(original_data: np.ndarray, smoothed_data: np.ndarray) -> Dict[str, float]:
        """Calculate statistics to evaluate the smoothing process."""
        stdev_original = np.std(original_data)
        stdev_smoothed = np.std(smoothed_data)
        noise_reduction_percent = (stdev_original - stdev_smoothed) / stdev_original * 100
        max_diff = np.max(np.abs(original_data - smoothed_data))
        mean_abs_error = np.mean(np.abs(original_data - smoothed_data))
        return {
            'original_std': stdev_original,
            'smoothed_std': stdev_smoothed,
            'noise_reduction': noise_reduction_percent,
            'max_deviation': max_diff,
            'mean_absolute_error': mean_abs_error
        }

# Before and after visualization
def visualize_smoothing(original_data, smoothed_data, column_name, alpha, timestamps):
    """
    Visualize original and smoothed data using time values for x-axis."""
    fig = go.Figure()

    # Original data
    fig.add_trace(
        go.Scatter(x=timestamps, y=original_data,
                   name="Original data",
                   line=dict(color='#1f77b4', width=1),
                   opacity=0.8)
    )

    # Smoothed data
    fig.add_trace(
        go.Scatter(x=timestamps, y=smoothed_data,
                   name="Smoothed data",
                   line=dict(color='#ff7f0e', width=2))
    )

    # Configure plot
    fig.update_layout(
        height=400,
        title=f"Smoothing: {column_name} (α = {alpha:.2f})",
        showlegend=True,
        legend=dict(orientation="h", y=-0.3, yanchor="middle", xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Value")
    st.plotly_chart(fig, use_container_width=True)


def visualize_original_data(data, column_name, timestamps):
    """
    Visualize original data using time values for x-axis.
    """
    fig = go.Figure()

    # Original data
    fig.add_trace(
        go.Scatter(x=timestamps, y=data,
                   name="Original data",
                   line=dict(color='#1f77b4', width=1))
    )

    # Configure plot
    fig.update_layout(
        height=400,
        title=f"Time Series: {column_name}",
        showlegend=True,
        legend=dict(orientation="h", y=-0.3, yanchor="middle", xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Value")
    st.plotly_chart(fig, use_container_width=True)


def record_smoothing_step(column: str, alpha: float, stats: dict):
    """Record a smoothing operation in the session state history."""
    timestamp = datetime.now()
    step = {
        'timestamp': timestamp,
        'column': column,
        'alpha': alpha,
        'result_column': f"{column}_smoothed"
    }
    st.session_state.smoothing_steps.append(step)
    history_message = f"[{timestamp.strftime('%Y-%m-%d %H:%M')}] Smoothing applied to {column} (α={alpha:.2f})"
    if not any(history_message in existing_message for existing_message in st.session_state.analysis_history):
        st.session_state.analysis_history.append(history_message)


def analyze_smooth(df: pd.DataFrame = None) -> pd.DataFrame:
    initialize_session_state()
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []

    if df is not None:
        pass
    elif 'current_df' in st.session_state and st.session_state.current_df is not None:
        df = st.session_state.current_df.copy()
    else:
        st.error("Please load data first using the Data Loader.")
        return None

    st.write("##### Current Dataset Information")
    dataset_name = st.session_state.get('current_dataset_name', 'Unnamed Dataset')
    st.markdown(f"**Dataset Name:** {dataset_name}")
    actual_rows = len(st.session_state.current_df)
    st.write(f"Using a dataset with {actual_rows:,} rows")

    if st.session_state.analysis_history:
        with st.expander("📝 **Analysis History**", expanded=False):
            for message in st.session_state.analysis_history:
                st.write(message)

    time_col = [col for col in df.columns if 'Zeit' in col][0]
    timestamps = df[time_col].values

    chosen_columns = st.multiselect("Choose columns to analyze", options=df.columns[1:])
    if not chosen_columns:
        st.info("Please choose at least one column to smooth.")
        return df

    # Create tabs for each column
    tabs = st.tabs([f"Analysis for {' '.join(col) if isinstance(col, tuple) else str(col)}"
                    for col in chosen_columns])

    processed_columns = []  # number of variables chosen
    attempted_columns = []  # number of actual smoothed data

    # Process each column
    for tab, chosen_column in zip(tabs, chosen_columns):
        with tab:
            st.write(
                f"### Analysis for {' '.join(chosen_column) if isinstance(chosen_column, tuple) else str(chosen_column)}")
            # Prepare data
            data = df[chosen_column].astype(float).values
            attempted_columns.append(chosen_column)
            needs_smoothing = check_smoothing_need(data)
            if needs_smoothing:
                st.warning("⚠️ Noise detected - Smoothing recommended")

                # Only calculate and show suggested alpha if smoothing is needed
                suggested_alpha = calculate_column_volatility(data)
                st.info(f"""
                **Suggested Smoothing Strength (α):** {suggested_alpha:.2f}
                Interpretation:
                - α = {suggested_alpha:.2f} suggests moderate smoothing for this column
                - Closer to 0: Very smooth, slow to show changes
                - Closer to 1: Keeps more original data patterns
                """)

                # Allow user to adjust slider
                slider_key = f"alpha_slider_{str(chosen_column).replace(' ', '_')}"
                alpha = st.slider(
                    "Adjust Smoothing Strength (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(suggested_alpha),
                    step=0.1,
                    key=slider_key
                )

                # Apply smoothing with current alpha value
                smoothed_data = exponential_smoothing(data, alpha)
                stats = calculate_smoothing_statistics(data, smoothed_data)
                # Put stats in a collapsible expander
                with st.expander("Smoothing Statistics"):
                    stat_info = """
                                   - **Noise Reduction**: Percentage of noise removed by smoothing  
                                   - **Original Std Dev**: Standard deviation of original data  
                                   - **Smoothed Std Dev**: Standard deviation after smoothing  
                                   """
                    st.info(stat_info)
                    metrics_df = pd.DataFrame({
                        'Metric': ['Noise Reduction (%)', 'Original Std Dev', 'Smoothed Std Dev'],
                        'Value': [f"{stats['noise_reduction']:.2f}%",
                                  f"{stats['original_std']:.6f}",
                                  f"{stats['smoothed_std']:.6f}"]
                    })
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                result_column = f"{chosen_column}_smoothed"

                # Visualization button and plot
                st.caption("Warning: Plotting a large dataset may take time or affect UI performance.")
                if st.button("Plot Data", key=f"plot_btn_{chosen_column}", type="primary"):
                    with st.expander("📊 Visualization", expanded=True):
                        visualize_smoothing(data, smoothed_data, str(chosen_column), alpha, timestamps)

                # Add apply button for this specific column
                if st.button(f"Apply Smoothing to {chosen_column}", key=f"apply_{chosen_column}", type="primary"):
                    df[result_column] = smoothed_data
                    st.session_state.smoothing_stats[result_column] = stats
                    record_smoothing_step(chosen_column, alpha, stats)
                    processed_columns.append(chosen_column)
                    update_session_state(df, analysis_type='smooth')
                    st.success(f"✅ Smoothing applied to {chosen_column}")
            else:
                st.success("✓ No significant noise detected - No smoothing required")
                visualize_original_data(data, str(chosen_column), timestamps)

    # Show final status
    if processed_columns:
        st.session_state.current_df = df.copy()

    # Preview of dataframe
    st.write("### Dataset Preview")
    preview_df = df.head(15)
    display_df = preview_df.copy()
    numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notnull(x) else x)
        display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True, height=350)
    return st.session_state.current_df
