import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

def plot_sampled(df: pd.DataFrame):
    columns = [col for col in df.columns if col != 'Zeit[(s)]']
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
                x=plot_df['Zeit[(s)]'],
                y=y_data,
                name=f"{col} ({'Normalized' if normalize else 'Original'})",
                mode='lines'
            )
        )

    fig.update_layout(
        height=500,
        title_text=f"Time Series Plot ({'Normalized' if normalize else 'Original'})",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized Values" if normalize else "Original Values",
        showlegend=True,
        dragmode='zoom',
        xaxis=dict(rangeslider=dict(visible=True)),
        legend = dict(
            orientation="h",
            y=1.0,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)

