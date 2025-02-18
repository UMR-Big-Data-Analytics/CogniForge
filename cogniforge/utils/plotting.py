import lttb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_sampled(df: pd.DataFrame, sampling_size: int = 1000) -> None:
    columns = df.columns
    idx = df.index.values

    chosen_columns = st.multiselect("Choose columns", columns, default=columns[0])
    normalize = st.checkbox("Normalize data", value=False)
    index_range = st.slider(
        "Index range",
        min_value=idx[0],
        max_value=idx[-1],
        value=(idx[0], idx[-1]),
    )
    idx_sampled = None
    plotting_dict = {}
    if len(chosen_columns) == 0:
        st.error("Please choose at least one column to plot.")
        return
    for col in chosen_columns:
        df_range = df[(df.index >= index_range[0]) & (df.index <= index_range[1])]
        col_with_idx = np.c_[df_range.index.values, df_range[col].values]
        if sampling_size >= len(col_with_idx):
            sampled = col_with_idx
        else:
            sampled = lttb.downsample(col_with_idx, sampling_size)
        if idx_sampled is None:
            idx_sampled = sampled[:, 0]
        plotting_dict[col] = sampled[:, 1]
    plotting_dict["Index"] = idx_sampled

    df_plot = pd.DataFrame(plotting_dict).set_index("Index")[chosen_columns]
    if len(chosen_columns) == 1:
        # rename columns, remove special characters (https://github.com/streamlit/streamlit/issues/7714)
        df_plot.columns = df_plot.columns.str.replace("[^a-zA-Z0-9]", "_", regex=True)
    if normalize:
        df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    st.line_chart(df_plot)

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