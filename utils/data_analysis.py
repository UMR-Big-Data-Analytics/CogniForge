import streamlit as st
from .state_button import button
from cogniforge.algorithms.anomaly_detection import SpuckerCounter

# basic stats: maybe not needed
def analyze_wire_data():
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("### Statistical Summary")
        st.dataframe(df.describe())
    else:
        st.warning("Load data first")


def analyze_spucker(df):
    with st.expander("Spucker Analysis"):
        if not button("Count Spucker", "spucker", True):
            return
        st.write("## Spucker Count")

        column = st.selectbox("Select column", df.columns)
        column_idx = df.columns.get_loc(column)
        threshold = st.slider(
            "Threshold",
            0.0,
            df.iloc[:, column_idx].max(),
            70.0
        )

        index_range = st.slider(
            "Select range for Spucker count",
            0.0,
            float(df.index[-1]),
            (1.0, df.index[-1] - 1.0),
        )

        context_window = st.slider("Context window", 0,
                                   int(df.index[-1]),
                                   (1, 5)
                                   )

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