import numpy as np
import pandas as pd
import lttb
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from utils.session_state_management import update_session_state

@st.cache_data
def initialize_session_state():
    if 'downsample_steps' not in st.session_state:
        st.session_state.downsample_steps = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'is_downsampled' not in st.session_state:
        st.session_state.is_downsampled = False
    if 'downsampled_df' not in st.session_state:
        st.session_state.downsampled_df = None


def downsample_dataframe(
        df: pd.DataFrame,
        time_column: str = None,
        max_points: int = 1000,
        display_mode: bool = False
) -> pd.DataFrame:
    """Downsample a DataFrame using LTTB"""
    if df is None or df.empty:
        st.error("DataFrame is None or empty")
        return None

    # Time column
    if time_column is None:
        time_column = next(
            (col for col in df.columns if col.lower().startswith(("zeit", "time"))),
            df.columns[0]
        )

    # Select numeric columns (excluding time)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != time_column]
    # Validate numeric columns
    if not numeric_columns:
        st.error("No numeric columns available for downsampling")
        return None
    # Ensure time column is numeric
    try:
        time_values = df[time_column].values.astype(np.float64)
    except Exception as e:
        st.error(f"Could not convert time column to numeric: {e}")
        return None
    if len(df) <= max_points:
        return df.copy()

    # Perform LTTB downsamplig
    try:
        downsampled_data = {}
        # Downsample time column first
        downsampled_data[time_column] = lttb.downsample(
            np.column_stack([time_values, time_values]),
            max_points
        )[:, 0]
        # Downsample each numeric column
        for col in numeric_columns:
            column_values = df[col].values
            downsampled_data[col] = lttb.downsample(
                np.column_stack([time_values, column_values]),
                max_points
            )[:, 1]
        downsampled_df = pd.DataFrame(downsampled_data)
    except Exception as e:
        st.error(f"Downsampling failed: {e}")
        st.error(f"Input data shapes: {[df[col].shape for col in [time_column] + numeric_columns]}")
        return None

    if display_mode:
        display_df = downsampled_df.copy()
        numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(
                lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notnull(x) else x
            )
        return display_df

    return downsampled_df

def record_downsampling_step(original_size: int, new_size: int):
    """Record a downsampling operation in the session state history."""
    timestamp = datetime.now()
    step = {
        "timestamp": timestamp,
        "original_size": original_size,
        "new_size": new_size,
    }
    if 'downsample_steps' not in st.session_state:
        st.session_state.downsample_steps = []
    st.session_state.downsample_steps.append(step)
    history_message = f"[{timestamp.strftime('%Y-%m-%d %H:%M')}] Data downsampled from {original_size:,} to {new_size:,} observations"
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if not any(history_message in msg for msg in st.session_state.analysis_history):
        st.session_state.analysis_history.append(history_message)


def create_downsampling_plots(original_df, column, downsampled_df=None, title="Downsampling Analysis"):
    """
    plot before and after
    """
    # Get timestamps and data
    original_timestamps = original_df['Zeit[(s)]'].values
    if downsampled_df is None:
        return None
    downsampled_timestamps = downsampled_df['Zeit[(s)]'].values
    # Get the data values
    original_values = original_df[column].values
    downsampled_values = downsampled_df[column].values

    # Create the figure
    fig = go.Figure()

    # Add original data
    fig.add_trace(
        go.Scatter(
            x=original_timestamps,
            y=original_values,
            name='Original Data',
            mode='lines',
            line=dict(color='royalblue', width=1.5),
            hovertemplate="Time: %{x:.2f}s<br>Value: %{y:.4f}<br><extra>Original data</extra>"
        )
    )
    # Downsampled data
    fig.add_trace(
        go.Scatter(
            x=downsampled_timestamps,
            y=downsampled_values,
            name='Downsampled Points',
            mode='markers+lines',
            line={'color': 'orange', 'width': 1},
            marker=dict(color='darkorange', size=2, symbol='circle'),
            hovertemplate="Time: %{x:.2f}s<br>Value: %{y:.4f}<br><extra>Downsampled data</extra>"
        )
    )

    # Zoom feature
    fig.update_layout(
        height=500,
        title=f"{title} - {column}<br><sup>Comparing {len(original_df):,} original points with {len(downsampled_df):,} downsampled points</sup>",
        dragmode='zoom',
        clickmode='event+select',
        selectdirection='h',

        # X-axis configurations
        xaxis=dict(title="Time (seconds)", rangeslider=dict(visible=True), showspikes=True, spikecolor='gray'),
        yaxis=dict(title="Value", showspikes=True, spikecolor='gray'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5))
    fig.update_layout(hovermode='x unified')
    return fig


def downsampling_page(df: pd.DataFrame = None) -> pd.DataFrame:
    """Main page for data downsampling analysis and visualization."""
    initialize_session_state()
    if st.session_state.current_df is None:
        st.error("Please load data first using the Load Data section.")
        return None

    df = st.session_state.get('current_df', df).copy()
    # Display dataset information
    st.write("#### Current Dataset Information")
    dataset_name = st.session_state.get('current_dataset_name', 'Unnamed Dataset')
    st.markdown(f"**Dataset Name:** {dataset_name}")
    st.write(f"Using a dataset with {len(df):,} rows")

    # analysis history
    if st.session_state.analysis_history:
        with st.expander("📝 **Analysis History**", expanded=False):
            for message in st.session_state.analysis_history:
                st.write(message)

    # Configure downsampling parameters
    max_points = st.slider(
        "Sample size",
        min_value=10,
        max_value=min(5000, len(df)),
        value=min(1000, len(df)),
        step=100
    )
    # Main visualization control
    st.caption("Warning: Plotting a large dataset may take time or affect UI performance.")
    plot_button = st.button("Plot Data", type="primary")
    if plot_button:
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                           if col != 'Zeit[(s)]']

        if not numeric_columns:
            st.info("No numeric columns available for downsampling.")
            return df

        with st.spinner("Generating visualization..."):
            preview_downsampled_df = downsample_dataframe(df, max_points=max_points)

            if preview_downsampled_df is not None:
                tabs = st.tabs([f"Analysis for {col}" for col in numeric_columns])

                for tab, column in zip(tabs, numeric_columns):
                    with tab:
                        # Display analysis information
                        if len(df) > max_points:
                            st.info(f"""
                            Current data points: {len(df):,}
                            Target data points: {max_points:,}
                            """)
                        with st.expander("📊 Visualization", expanded=True):
                            fig = create_downsampling_plots(
                                df, column, preview_downsampled_df)
                            st.plotly_chart(fig, use_container_width=True)


    # Apply Downsampling button
    apply_downsample_button = st.button("Apply Downsampling", type="primary")
    if apply_downsample_button:
        with st.spinner('Applying downsampling to entire dataset...'):
            downsampled_df = downsample_dataframe(df, max_points=max_points)
            if downsampled_df is not None:
                update_session_state(downsampled_df, analysis_type='downsample')
                record_downsampling_step(len(df), len(downsampled_df))
                st.success(
                    f"✅ Downsampling applied successfully! Reduced from {len(df):,} to {len(downsampled_df):,} observations."
                )
                st.write("#### Updated Dataset Information")
                st.markdown(f"**Dataset Name:** {st.session_state.get('current_dataset_name', 'Unnamed Dataset')}")
                st.write(f"Using a dataset with {len(downsampled_df):,} rows")
                preview_df = downsampled_df
                display_df = preview_df.copy()
                numeric_cols = display_df.select_dtypes(
                    include=['float64', 'float32']).columns
                for col in numeric_cols:
                    display_df[col] = display_df[col].apply(
                        lambda x: f'{float(x):.6f}'.replace(',', '.')
                        if pd.notnull(x) else x)

                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True, height=350)
                return downsampled_df

    return st.session_state.current_df