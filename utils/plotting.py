import lttb
import numpy as np
import pandas as pd
import streamlit as st


def plot_sampled(df: pd.DataFrame, sampling_size: int = 1000) -> None:
    # start at column index 1 as the first include row number
    plottable_columns = list(df.columns[1:])

    chosen_columns = st.multiselect(
        "Choose columns",
        options=plottable_columns
    )
    # add message when nothing is chosen
    if not chosen_columns:
        st.error("Please choose at least one column to plot.")
        return

    normalize = st.checkbox("Normalize data", value=False)
    row_numbers = df[df.columns[0]].values
    index_range = st.slider(
        "Index range",
        min_value=int(row_numbers[0]),
        max_value=int(row_numbers[-1]),
        value=(int(row_numbers[0]), int(row_numbers[-1])),
    )

    idx_sampled = None
    plotting_dict = {}
    for col in chosen_columns:
        mask = (df[df.columns[0]] >= index_range[0]) & (df[df.columns[0]] <= index_range[1])
        df_range = df[mask]
        col_with_idx = np.c_[df_range[df.columns[0]].values, df_range[col].values]
        if sampling_size >= len(col_with_idx):
            sampled = col_with_idx
        else:
            sampled = lttb.downsample(col_with_idx, sampling_size)

        # Store sampled data
        if idx_sampled is None:
            idx_sampled = sampled[:, 0]
        plotting_dict[col] = sampled[:, 1]

    # Prep sample
    plotting_dict["Index"] = idx_sampled
    df_plot = pd.DataFrame(plotting_dict).set_index("Index")[chosen_columns]

    if len(chosen_columns) == 1:
        # rename columns, remove special characters (https://github.com/streamlit/streamlit/issues/7714)
        df_plot.columns = df_plot.columns.str.replace("[^a-zA-Z0-9]", "_", regex=True)
    if normalize:
        df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    st.line_chart(df_plot)

