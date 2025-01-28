import sys
import streamlit as st

# Add the Cogniforge directory to the Python path
sys.path.append("../cogniforge")

# Import necessary modules
from cogniforge.algorithms.anomaly_detection import SpuckerCounter
from cogniforge.utils.dataloader import DataLoader
from cogniforge.utils.furthr import FURTHRmind
from cogniforge.utils.plotting import plot_sampled
from cogniforge.utils.state_button import button

# Page configuration
st.set_page_config(
    page_title="CogniForge | Wire Quality",
    page_icon="🔌",
)

# Header and introduction
st.title("🔌 Wire Quality Analysis")
st.markdown(
    """
    Welcome to the **Wire Quality Analysis Tool**!  
    - Upload and analyze wire quality data.
    - Visualize trends and anomalies using advanced algorithms.
    - Get insights on the wire quality through Spucker analysis.

    👉 **Start by downloading your data from the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database!**
    """
)

# Download Data from FURTHRmind
with st.expander("📥 Download Data from FURTHRmind"):
    st.info("Use the FURTHRmind integration to download your data.")
    data = FURTHRmind().download_csv()

if data is not None:
    # Parse the downloaded data
    with st.expander("📊 Parse Data"):
        st.info("Parsing your CSV data...")
        dl = DataLoader(csv=data)
        df = dl.get_dataframe()
        st.success("Data successfully parsed!")
        st.dataframe(df)

if data is not None and df is not None:
    # Plot the data
    with st.expander("📈 Visualize Data"):
        if button("Plot Data", "plot_data_wire", True):
            st.info("Generating a sampled plot...")
            plot_sampled(df)

    # Analyze data using Spucker Counter
    with st.expander("🔍 Perform Spucker Analysis"):
        if button("Count Spucker", "spucker", True):
            st.subheader("🧮 Spucker Count Analysis")
            st.write("Select the required parameters for analysis:")

            # Column selection
            column = st.selectbox("Select Column for Analysis", df.columns)

            # Threshold selection
            column_idx = df.columns.get_loc(column)
            threshold = st.slider(
                "Threshold", 0.0, float(df.iloc[:, column_idx].max()), 70.0
            )

            # Index range selection
            index_range = st.slider(
                "Select Range for Analysis",
                0.0,
                float(df.index[-1]),
                (1.0, float(df.index[-1]) - 1.0),
            )

            # Context window
            context_window = st.slider(
                "Context Window (seconds)", 0, int(df.index[-1]), (1, 5)
            )

            # Spucker Counter initialization
            counter = SpuckerCounter(
                threshold=threshold,
                ignore_begin_seconds=index_range[0],
                ignore_end_seconds=df.index[-1] - index_range[1],
                context_window=context_window,
            )

            if button("Start Spucker Count", "count_spucker", True):
                st.info(f"Analyzing {len(df)} rows...")
                count = counter.count(df.iloc[:, column_idx])
                st.success(f"Spucker Count: **{count}**")

# Footer
st.markdown("---")
st.caption("CogniForge | Wire Quality Analysis Tool")
