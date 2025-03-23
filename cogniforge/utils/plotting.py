import lttb
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np


def plot_sampled(df: pd.DataFrame):
    if 'current_df' not in st.session_state:
        st.session_state.current_df = df
    if st.session_state.get("current_df") is not None:
        df = st.session_state.current_df
    if st.session_state.get("is_downsampled", False):
        df = st.session_state.downsampled_df
    if st.session_state.get("detrending_active", False):
        df = st.session_state.full_analyzed_df
    if st.session_state.get("smoothing_active", False):
        df = st.session_state.full_analyzed_df

    if df is None:
        st.error("Please load data first using the Load Data Page.")
        return None

    # ***Dataset Information - Moved and updated***
    st.write("#### Current Dataset Information")
    dataset_name = st.session_state.get('current_dataset_name', 'Unnamed Dataset')
    st.markdown(f"**Dataset Name:** {dataset_name}")
    st.write(f"Using a dataset with {len(st.session_state.current_df):,} rows")

    # Find the time column (contains 'Zeit')
    time_column = None
    for col in df.columns:
        if 'Zeit' in col:
            time_column = col
            break

    if time_column is None:
        st.error("No column containing 'Zeit' found for time series data.")
        return

    columns = [col for col in df.columns if col != time_column]
    chosen_columns = st.multiselect("Choose columns", columns)

    if len(chosen_columns) == 0:
        st.error("Please choose at least one column to plot.")
        return

    normalize = st.checkbox("Normalize Data")

    plot_df = df.copy()
    fig = make_subplots(rows=1, cols=1)
    for col in chosen_columns:
        y_data = plot_df[col]
        if normalize:
            y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())

        fig.add_trace(
            go.Scatter(
                x=plot_df[time_column],
                y=y_data,
                name=f"{col} ({'Normalized' if normalize else 'Original'})",
                mode='lines'
            )
        )

    fig.update_layout(
        height=500,
        title_text=f"Time Series Plot ({'Normalized' if normalize else 'Original'})",
        xaxis_title="Time (seconds)" if 'Zeit[(s)]' in time_column else "Time (milliseconds)",
        yaxis_title="Normalized Values" if normalize else "Original Values",
        showlegend=True,
        dragmode='zoom',
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(
            orientation="h",
            y=1.0,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_xy_chart(df: pd.DataFrame, sampling_size: int = 1000) -> None:
    columns = df.columns

    x_axis = st.selectbox("Select X-axis", columns, index=0)
    y_axis_options = [col for col in columns if col != x_axis]
    y_axis = st.multiselect("Select Y-axis", y_axis_options, default=[y_axis_options[0]] if y_axis_options else [])

    if not y_axis:
        st.error("Please select at least one column for Y-axis.")
        return

    df_filtered = df[[x_axis] + y_axis].dropna()
    df_filtered[x_axis] = pd.to_numeric(df_filtered[x_axis], errors="coerce")
    df_filtered = df_filtered.sort_values(by=x_axis).drop_duplicates(subset=[x_axis])

    # Downsample only if necessary
    if len(df_filtered) > sampling_size:
        sampled_data = {}
        for col in y_axis:
            col_with_x = np.c_[df_filtered[x_axis].values, df_filtered[col].values]
            sampled = lttb.downsample(col_with_x, sampling_size)
            sampled_data[col] = sampled[:, 1]
        sampled_data[x_axis] = sampled[:, 0]
        df_plot = pd.DataFrame(sampled_data)
    else:
        df_plot = df_filtered  # Use full dataset if no downsampling is needed

    fig = px.line(df_plot, x=x_axis, y=y_axis, title="")
    fig.update_layout(
        legend=dict(
            orientation="v",  # vertical layout
            yanchor="bottom",
            y=-1,  # Adjust position
            xanchor="center",
            x=0.5,
        )
    )
    st.plotly_chart(fig, use_container_width=True)
