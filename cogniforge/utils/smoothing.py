import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
from typing import Dict, Any
from datetime import datetime

"""
Perform smoothing analysis on selected columns using exponential smoothing
Functions:
- Select columns to undergo smoothing
- Code check
- Code suggests an initial smoothing factor based on variable's variability (calculated by calculate_column_volatility())
- User can adjust smoothing factor using a slider
- Enable visualization of before and after smoothing
- Add smoothing to analysis history if smoothing is performed
"""
def initialize_session_state():
    if 'smoothing_steps' not in st.session_state:
        st.session_state.smoothing_steps = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'smoothing_stats' not in st.session_state:
        st.session_state.smoothing_stats = {}
    if 'column_alphas' not in st.session_state:
        st.session_state.column_alphas = {}
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def analyze_smooth(df: pd.DataFrame = None) -> pd.DataFrame:
    """Main function for smoothing"""
    initialize_session_state()

    if df is not None:
        st.session_state.current_df = df.copy()
    if st.session_state.current_df is None:
        st.error("Please load data first using the Data Loader.")
        return None

    # Display current dataset info
    df = st.session_state.current_df
    dataset_name = st.session_state.get('current_dataset_name', 'Unnamed Dataset')
    st.markdown(f"**Dataset Name:** {dataset_name}")
    actual_rows = len(df)
    st.write(f"Using a dataset with {actual_rows:,} rows")

    # Select variables
    chosen_columns = st.multiselect(
        "Choose columns to smooth",
        options=list(df.columns[1:]),
        format_func=lambda x: ' '.join(x) if isinstance(x, tuple) else str(x)
    )

    if not chosen_columns:
        st.info("Please choose at least one column to smooth.")
        return df

    # Compute initial suggested alphas for each column
    column_volatilities = {col: calculate_column_volatility(df[col].astype(float).values) for col in chosen_columns}

    # Create tabs for each column
    tabs = st.tabs([f"Analysis for {' '.join(col) if isinstance(col, tuple) else str(col)}"
                    for col in chosen_columns])

    processed_columns = []
    attempted_columns = []

    # Process each column
    for idx, (tab, chosen_column) in enumerate(zip(tabs, chosen_columns)):
        with tab:
            st.write(
                f"### Analysis for {' '.join(chosen_column) if isinstance(chosen_column, tuple) else str(chosen_column)}")

            suggested_alpha = column_volatilities[chosen_column]
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

            # Prepare data
            data = df[chosen_column].astype(float).values
            attempted_columns.append(chosen_column)
            needs_smoothing = check_smoothing_need(data)

            if needs_smoothing:
                st.warning("⚠️ Noise detected - Smoothing recommended")
                # Apply smoothing with current alpha value
                smoothed_data = exponential_smoothing(data, alpha)
                result_column = f"{chosen_column}_smoothed"

                # Calculate and display statistics for current alpha
                stats = calculate_smoothing_statistics(data, smoothed_data)
                st.write("#### Smoothing Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Noise Reduction (%)', 'Original Std Dev', 'Smoothed Std Dev'],
                    'Value': [f"{stats['noise_reduction']:.2f}%",
                              f"{stats['original_std']:.6f}",
                              f"{stats['smoothed_std']:.6f}"]
                })
                st.dataframe(metrics_df, use_container_width=True)

                # Visualize current smoothing
                visualize_smoothing(data, smoothed_data, str(chosen_column), alpha)

                # Add apply button for this specific column
                if st.button(f"Apply Smoothing to {chosen_column}", key=f"apply_{chosen_column}", type="primary"):
                    df[result_column] = smoothed_data
                    st.session_state.smoothing_stats[result_column] = stats
                    record_smoothing_step(chosen_column, alpha, stats)
                    processed_columns.append(chosen_column)
                    st.success(f"✅ Smoothing applied to {chosen_column}")
            else:
                st.success("✓ No significant noise detected - No smoothing applied")

    # Show final status
    if processed_columns:
        if len(attempted_columns) > len(processed_columns):
            st.success(
                f"✅ Smoothing complete:\n"
                f"- Columns checked: {len(attempted_columns)}\n"
                f"- Columns actually smoothed: {len(processed_columns)}"
            )
        else:
            st.success(f"✅ Smoothing complete - {len(processed_columns)} columns processed")
        st.session_state.current_df = df.copy()

    # Preview of dataframe
    st.write("### Dataset Preview")
    preview_df = df.head(15)
    display_df = preview_df.copy()
    numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notnull(x) else x)

    st.dataframe(display_df, use_container_width=True, height=350)
    return df


def calculate_column_volatility(column_data: np.ndarray) -> float:
    """Calculate volatility for a given column."""
    changes_between_points = np.diff(column_data)
    spread_of_changes = np.std(changes_between_points)
    spread_of_data = np.std(column_data)
    volatility = spread_of_changes / spread_of_data
    return max(0.1, min(0.9, 1 - volatility))


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
    outlier_ratio = np.mean(z_scores > 3)
    local_changes = np.abs(np.diff(data))
    signal_mean = np.mean(np.abs(data))
    test_alpha = 0.5
    # Extra check
    test_smoothed = exponential_smoothing(data, test_alpha)
    test_stats = calculate_smoothing_statistics(data, test_smoothed)
    needs_smoothing = (
            ((rolling_std.mean() > 0.1 * overall_std) or
             (outlier_ratio > 0.05) or
             (local_changes.mean() > 0.1 * signal_mean)) and
            (test_stats['noise_reduction'] > 0.5)
    )
    return needs_smoothing

# Weighted moving average
def exponential_smoothing(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential smoothing to the input data. """
    smoothed = np.zeros(len(data))
    smoothed[0] = data[0]
    mask = np.isnan(data)
    clean_data = np.where(mask, smoothed[0], data)

    for t in range(1, len(data)):
        if mask[t]:
            smoothed[t] = smoothed[t - 1]
        else:
            smoothed[t] = alpha * clean_data[t] + (1 - alpha) * smoothed[t - 1]
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
def visualize_smoothing(original_data, smoothed_data, column_name, alpha):
    time_idx = np.arange(len(original_data))

    fig = go.Figure()
    # original data
    fig.add_trace(go.Scatter(x=time_idx, y=original_data, name="Original", line=dict(color='#1f77b4', width=1), opacity=0.8))
    # smoothed data
    fig.add_trace(go.Scatter(x=time_idx, y=smoothed_data, name="Smoothed", line=dict(color='#ff7f0e', width=2)))
    # Configure plot
    fig.update_layout(
        height=400, title=f"Smoothing: {column_name} (α = {alpha:.2f})", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    fig.update_xaxes(title_text="Time Index")
    fig.update_yaxes(title_text="Value")
    st.plotly_chart(fig, use_container_width=True)


def record_smoothing_step(column: str, alpha: float, stats: dict):
    """Record a smoothing operation in the session state history."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    step = {
        'timestamp': timestamp,
        'column': column,
        'alpha': alpha,
        'noise_reduction': stats['noise_reduction'],
        'result_column': f"{column}_smoothed"
    }
    st.session_state.smoothing_steps.append(step)
    history_message = f"Smoothing applied to {column} (α={alpha:.2f}, noise reduction={stats['noise_reduction']:.1f}%)"
    if not any(history_message in existing_message for existing_message in st.session_state.analysis_history):
        st.session_state.analysis_history.append(history_message)


