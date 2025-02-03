import sys

import streamlit as st

sys.path.append("../cogniforge")
from cogniforge.algorithms.anomaly_detection import SpuckerCounter
from cogniforge.utils.dataloader import DataLoader
from cogniforge.utils.furthr import FURTHRmind
from cogniforge.utils.plotting import plot_sampled
from cogniforge.utils.state_button import button

st.set_page_config(
    page_title="CogniForge | Wire Quality",
    page_icon="🔌",
)

st.write("# Wire Quality")
st.write(
    "Welcome to the Wire Quality tool. Here you can analyze and visualize the quality of your wire."
)
st.write(
    "Upload your data to the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database. Then, here, you can choose the correct dataset and let our algorithms tell you the quality of your wire."
)

with st.expander("Download Data from FURTHRmind"):
    downloader = FURTHRmind()
    downloader.file_extension = "csv"
    downloader.select_file()
    data = downloader.download_string_button()

if data is not None:
    with st.expander("Parse Data"):
        dl = DataLoader(csv=data)
        df = dl.get_dataframe()

if data is not None and df is not None:
    with st.expander("Plot Data"):
        if button("Plot data", "plot_data_wire", True):
            plot_sampled(df)

    with st.expander("Analysis"):
        if button("Count Spucker", "spucker", True):
            st.write("## Spucker Count")
            column = st.selectbox("Select column", df.columns)
            column_idx = df.columns.get_loc(column)
            threshold = st.slider("Threshold", 0.0, df.iloc[:, column_idx].max(), 70.0)
            index_range = st.slider(
                "Select range for Spucker count",
                0.0,
                df.index[-1],
                (1.0, df.index[-1] - 1.0),
            )
            context_window = st.slider("Context window", 0, int(df.index[-1]), (1, 5))
            counter = SpuckerCounter(
                threshold=threshold,
                ignore_begin_seconds=index_range[0],
                ignore_end_seconds=df.index[-1] - index_range[1],
                context_window=context_window,
            )
            if button("Start counting", "count_spucker", True):
                st.write(f"Counting Spucker in {len(df)} lines...")
                count = counter.count(df.iloc[:, column_idx])
                st.write(f"Spucker count: {count}")
