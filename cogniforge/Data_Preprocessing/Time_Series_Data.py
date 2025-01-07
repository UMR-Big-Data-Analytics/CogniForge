import sys
import streamlit as st

# Initialize subpage state if not exists
if 'ts_subpage' not in st.session_state:
    st.session_state.ts_subpage = "Overview"

# Initialize dataframe state if not exists
if 'df' not in st.session_state:
    st.session_state.df = None

# Subnavigation in sidebar
with st.sidebar:
    st.subheader("Time Series Tools")

    if st.button("📋 Overview", use_container_width=True, key="ts_overview"):
        st.session_state.ts_subpage = "Overview"

    if st.button("📥 Load Data", use_container_width=True, key="ts_load"):
        st.session_state.ts_subpage = "Load Data"

    if st.button("📊 Plot Data", use_container_width=True, key="ts_plot"):
        st.session_state.ts_subpage = "Plot Data"

    if st.button("📉 Detrend Data", use_container_width=True, key="ts_detrend"):
        st.session_state.ts_subpage = "Detrend Data"

    if st.button("✂️ Cut Data", use_container_width=True, key="ts_cut"):
        st.session_state.ts_subpage = "Cut Data"

    if st.button("📉 Smooth Data", use_container_width=True, key="ts_smooth"):
        st.session_state.ts_subpage = "Smooth Data"

# Subpages
if st.session_state.ts_subpage == "Overview":
    st.title("⏳ Time Series Analysis")
    st.write("Time Series Analysis section.")
    st.markdown("""
           ### Available Tools
           - **Load Data**: Download and process data from FURTHRmind database
           - **Plot Data**: Visualize data
           - **Time Trend**: Analyze time trends in your data
           - **Detrend Data**: Perform detrending on the data
           - **TO ADD:** cut/smooth data
           """)


elif st.session_state.ts_subpage == "Load Data":
    st.title("📥 Load Data")
    st.write("Download your data from the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database.")

    try:
        from cogniforge.utils.dataloader import DataLoader
        from cogniforge.utils.furthr import FURTHRmind

        data = None
        with st.expander("Step 1: Download Data", expanded=True):
            with st.spinner('Downloading data...'):
                data = FURTHRmind().download_csv()
                if data is not None:
                    st.success("Data downloaded!")

        if data is not None:
            with st.expander("Step 2: Process Data", expanded=False):
                with st.spinner('Processing...'):
                    dl = DataLoader(csv=data)
                    df = dl.get_dataframe()
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.success("Data processed successfully")

                        st.write("### Data Preview")
                        preview_df = df.head(10).copy()
                        st.dataframe(
                            preview_df,
                            use_container_width=True,
                            hide_index=False
                        )
    except ImportError as e:
        st.error(f"Import error: {e}. Please ensure that the `cogniforge` package is correctly installed.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

elif st.session_state.ts_subpage == "Plot Data":
    from cogniforge.utils.plotting import plot_sampled

    st.title("📊 Plot Data")

    if st.session_state.df is not None:
        tab1 = st.selectbox("Select a Plot", ["Data Visualization", "Time Trend Analysis"])

        if tab1 == "Data Visualization":
            plot_sampled(st.session_state.df)

        elif tab1 == "Time Trend Analysis":
            st.warning("Please use the 'Detrend Data' tab to perform time trend analysis.")

    else:
        st.warning("Please load and process data first in the Load Data section.")

else:
        st.warning("Please load and process data first in the Load Data section.")