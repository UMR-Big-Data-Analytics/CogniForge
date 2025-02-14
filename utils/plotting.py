import lttb
import numpy as np
import pandas as pd
import streamlit as st


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