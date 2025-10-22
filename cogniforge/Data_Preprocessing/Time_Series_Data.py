import streamlit as st
from datetime import datetime
import pandas as pd
import time
from furthrmind import Furthrmind as API
from furthrmind.file_loader import FileLoader
from utils.downsampling import downsampling_page
from utils.session_state_management import update_session_state

# TIME SERIES ANALYSIS
# =========================================
# This code implements the main application structure for a time series analysis tool.
# MAIN FEATURES:
# - Data loading from the FURTHRmind database
# - Interactive visualizations of time series data
# - Trend detection and detrending
# - Noise reduction through smoothing
# - Data size reduction through downsampling
# - Results uploading back to the FURTHRmind database
if 'ts_subpage' not in st.session_state:
    st.session_state.ts_subpage = "Overview"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'use_full_dataset' not in st.session_state:
    st.session_state.use_full_dataset = False
if 'detrend_steps' not in st.session_state:
    st.session_state.detrend_steps = []
if 'smoothing_steps' not in st.session_state:
    st.session_state.smoothing_steps = []
if 'detrending_active' not in st.session_state:
    st.session_state.detrending_active = False
if 'smoothing_active' not in st.session_state:
    st.session_state.smoothing_active = False
if 'plotting_active' not in st.session_state:
    st.session_state.plotting_active = False
if 'data_info' not in st.session_state:
    st.session_state.data_info = {
        'use_full_dataset': True,
        'total_rows': 0
    }

def handle_revert():
    """Handle revert to original dataset"""
    current_timestamp = datetime.now()
    # Update both df and current_df to ensure consistency
    st.session_state.df = st.session_state.original_df.copy()
    st.session_state.current_df = st.session_state.original_df.copy()
    # Reset analysis type and processed columns
    st.session_state.current_analysis_type = None
    st.session_state.processed_columns = {
        'detrended': set(),
        'smoothed': set(),
        'downsampled': set()
    }
    st.session_state.is_downsampled = False
    st.session_state.downsampled_df = None
    revert_message = f"[{current_timestamp.strftime('%Y-%m-%d %H:%M')}] Reverted to original dataset"
    st.session_state.analysis_history.append(revert_message)
    # Reset
    if st.session_state.ts_subpage == "Detrend Data":
        st.session_state.detrend_steps = []
        st.session_state.detrending_active = False
        st.session_state.show_plots = False
    elif st.session_state.ts_subpage == "Smooth Data":
        st.session_state.smoothing_steps = []
        st.session_state.smoothing_active = False
    elif st.session_state.ts_subpage == "Downsample Data":
        st.session_state.downsample_steps = []
        st.session_state.downsampling_active = False
        st.session_state.show_downsampling_plots = False


# navigation sidebar
with st.sidebar:
    st.subheader("Time Series Tools")
    if st.button("📋 Overview", use_container_width=True, key="ts_overview"):
        st.session_state.ts_subpage = "Overview"
    if st.button("📥 Load Data", use_container_width=True, key="ts_load"):
        st.session_state.ts_subpage = "Load Data"
    if st.button("📊 Plot Data", use_container_width=True, key="ts_plot"):
        st.session_state.ts_subpage = "Plot Data"
    if st.button("🧰 Detrend Data", use_container_width=True, key="ts_detrend"):
        st.session_state.ts_subpage = "Detrend Data"
    if st.button("🧰 Smooth Data", use_container_width=True, key="ts_smooth"):
        st.session_state.ts_subpage = "Smooth Data"
    if st.button("🧰 Downsample Data", use_container_width=True, key="ts_downsample"):
        st.session_state.ts_subpage = "Downsample Data"
    if st.button("📤 Upload Results", use_container_width=True, key="ts_upload"):
        st.session_state.ts_subpage = "Upload Results"


# Subpages
if st.session_state.ts_subpage == "Overview":
    st.title("⏳ Time Series Analysis")
    st.write("Time Series Analysis section.")
    st.markdown("""
           ### Available Tools
           - **Load Data**: Download and process data from FURTHRmind database
           - **Plot Data**: Visualize data        
           - **Detrend Data**: Perform detrending on the data
           - **Smooth Data**: Smooth noisy data using exponential smoothing
           - **Downsample Data**: Reduce data size while preserving all the important patterns and features           
           - **Upload Results**: Save your analyzed data back to FURTHRmind
           """)

# Configure each subpage
#LOAD DATA PAGE
#Download and process data
if st.session_state.ts_subpage == "Load Data":
    st.title("📥 Load Data")
    # display current dataset info if exists
    if st.session_state.df is not None:
        info_message = ["📊 Current Dataset Information:"]
        if 'original_filename' in st.session_state:
            info_message.append(f"• Current dataset: {st.session_state.original_filename}")
        latest_df = st.session_state.get("current_df", st.session_state.df)
        if st.session_state.get("use_full_dataset", False):
            info_message.append(f"• Using complete dataset ({len(latest_df):,} rows)")
        else:
            info_message.append(f"• Using selected range ({len(latest_df):,} rows)")
        # display last analysis message
        last_analysis = None
        if st.session_state.detrending_active:
            last_analysis = "Detrending"
        elif st.session_state.smoothing_active:
            last_analysis = "Smoothing"
        elif st.session_state.downsampling_active:
            last_analysis = "Downsampling"
        if last_analysis:
            info_message.append(f"• Last analysis visited: {last_analysis}")
        # display current data info and the OPTION to clear current data and load a new dataset
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div style="border: 2px solid #f1f1f1; padding: 10px; background-color: #f5f5f5;">'
                        f'{"<br>".join(info_message)}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Clear Current Data", type="primary"):
                #reset evrything
                keys_to_clear = [
                    'df', 'original_df', 'detrending_active', 'smoothing_active', 'downsampling_active',
                    'analysis_history', 'original_filename', 'current_dataset_name',
                    'detrend_steps', 'smoothing_steps', 'current_analysis_type', 'processing_complete',
                    'plotting_active'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # Show data download section
    if st.session_state.df is None:
        st.write("Download your data from the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database.")
        try:
            from utils.dataloader import DataLoader
            from utils.furthr import FURTHRmind

            data = None
            # Download data
            with st.expander("Step 1: Download Data", expanded=True):
                with st.spinner('Downloading data...'):
                    downloader = FURTHRmind()
                    downloader.file_extension = "csv"
                    downloader.select_file()
                    data, filename = downloader.download_string_button() or (None, None)
                    if data is not None:
                        is_new_dataset = ('original_filename' in st.session_state and
                                         st.session_state.original_filename != filename)
                        st.session_state.original_filename = filename
                        data.name = filename
                        st.success("Data downloaded!")
            # process data
            if data is not None:
                with st.expander("Step 2: Process Data", expanded=False):
                    with st.spinner('Processing...'):
                        st.session_state.show_download_options = True
                        dl = DataLoader(csv=data)
                        df = dl.get_dataframe()
                        if df is not None and not df.empty:
                            st.session_state.df = df
                            st.session_state.processing_complete = True
                            # reset session states for a new dataset
                            if is_new_dataset:
                                st.session_state.analysis_history = []
                                st.session_state.detrending_active = False
                                st.session_state.smoothing_active = False
                                st.session_state.downsampling_active = False
                                st.session_state.detrend_steps = []
                                st.session_state.smoothing_steps = []
                                st.session_state.show_plots = False
        except ImportError as e:
            st.error(f"Import error: {e}. Please ensure that the `cogniforge` package is correctly installed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

#DATA VISUALIZATION PAGE
#Provides interactive visualization of time series data.
elif st.session_state.ts_subpage == "Plot Data":
    from utils.plotting import plot_sampled
    st.title("📊 Plot Data")
    with st.expander("ℹ️**How to Use**"):
        st.markdown("""
        ##### Time Series Data Visualization
        **This tool helps you to:**
        1. Visualize your data over time
        2. Normalize your data to compare different columns more effectively
        3. Zoom in on specific parts of the time series

        **How to Use:**
        - Select the column(s) you want to visualize from the available data
        - Optionally, choose to normalize the data to scale values between 0 and 1
        - Use the interactive chart to zoom in and explore the data more closely
        """)
    # show plotting if a dataset has already been loaded
    if st.session_state.current_df is not None:
        plot_sampled(st.session_state.current_df)
    else:
        st.warning("Please load and process data first in the Load Data section.")

# DETRENDING PAGE
elif st.session_state.ts_subpage == "Detrend Data":
    from utils.detrending import analyze_detrend
    from utils.session_state_management import update_session_state

    st.title("🧰 Detrend Data")
    with st.expander("**ℹ️How to Use**"):
        st.markdown("""
        **This tool helps you:**
        1. Detect trends in your selected column
        2. Preview detrending methods
        3. Visualize the impact of detrending before applying it.

        **How to Use:**
        - Select the column(s) to analyze
        - If a trend is detected, a detrend method is suggested. Users can also choose manually
        - Review trend statistics and visualization
        - Decide whether to apply detrending
        """)
    # show detredning if a dataset has already been loaded
    if st.session_state.current_df is not None:
        st.session_state.data_info['total_rows'] = len(st.session_state.current_df)
        try:
            detrended_df = analyze_detrend(st.session_state.current_df)
            if detrended_df is not None:
                update_session_state(detrended_df, analysis_type='detrend')

                if st.session_state.detrend_steps:
                    st.session_state.current_analysis_type = "detrend"
                    col1, col2 = st.columns([1, 4])
                    # Show revert button if detrending has been applied
                    with col1:
                        if st.button("Revert to Original Data"):
                            st.session_state.show_plots = False
                            handle_revert()
                            st.success("Successfully reverted to original data")
                            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during detrending: {str(e)}")
            st.exception(e)
    else:
        st.warning("Please load and process data first in the Load Data section.")

# SMOOTHING PAGE
elif st.session_state.ts_subpage == "Smooth Data":
    from utils.smoothing import analyze_smooth
    st.title("🧰 Smooth Data")
    with st.expander("**ℹ️How to Use**"):
        st.markdown("""
            **This tool helps you:**
            1. Smooth time series data using exponential smoothing to reduce noise and fluctuations
            4. Visualize the before-and-after effects of smoothing on your data
            5. Apply smoothing to the dataset and preview the results
    
            **Steps:**
            - Select the columns you want to smooth from the list
            - The tool will automatically suggest an initial smoothing factor (α) based on the column's volatility. Users can also adjust manually
            - Click "Plot Data" to review the visual comparison of original and smoothed data
            - Apply smoothing to the dataset
            - Optionally, revert back to the original data if needed by clicking "Revert to Original Data"
        """)
    # show smoothing if a dataset has already been loaded
    if st.session_state.current_df is not None:
        st.session_state.data_info['total_rows'] = len(st.session_state.current_df)
        try:
            smoothed_df = analyze_smooth(st.session_state.current_df)
            if smoothed_df is not None:
                update_session_state(smoothed_df, analysis_type='smooth')
                # Option to revert
                if st.session_state.smoothing_steps:
                    st.session_state.current_analysis_type = "smooth"
                    if st.button("Revert to Original Data"):
                        handle_revert()
                        st.success("Successfully reverted to original data")
                        st.rerun()
        except Exception as e:
            st.error(f"An error occurred during smoothing: {str(e)}")
            st.exception(e)
    else:
        st.warning("Please load and process data first in the Load Data section.")


# DOWNSAMPLING PAGE
elif st.session_state.ts_subpage == "Downsample Data":
    st.title("🧰 Data Downsampling")
    with st.expander("ℹ️ **How to use**"):
        st.markdown("""
                   **This tool helps you:**
                   1. Reduces the dataset size using LTTB downsampling
                   2. Compare original and downsampled data to evaluate the impact of downsampling
                   3. Apply downsampling to the dataset and preview the results

                   **Steps:**
                   - Select the number of points you'd like to downsample the data to using the slider
                   - Click on "Plot Data" to review the visual comparison of original and downsampled data
                   - If satisfied, click "Apply Downsample Data" to apply downsampling to your dataset
                   - Option to revert back to the original data if needed by clicking "Revert to Original Data"
                   """)

    if st.session_state.df is not None:
        st.session_state.current_analysis_type = "downsample"
        downsampled_df = downsampling_page(st.session_state.current_df)
        if downsampled_df is not None:
            update_session_state(downsampled_df, analysis_type='downsample')
        if st.button("Revert to Original Data"):
            if st.session_state.original_df is not None:
                handle_revert()
                st.success("Successfully reverted to original data")
                st.rerun()
    else:
        st.warning("Please load and process data first in the Load Data section.")

elif st.session_state.ts_subpage == "Upload Results":
    import time
    from utils.upload_analyzed_data import (
        show_analysis_history,
        setup_upload_location,
        prepare_dataframe,
        get_upload_data,
        generate_filename,
        upload_analyzed_data
    )
    st.title("📤 Upload Results")
    # Initialize session state variables
    if 'upload_clicked' not in st.session_state:
        st.session_state.upload_clicked = False
    if 'selected_location' not in st.session_state:
        st.session_state.selected_location = None
    if 'upload_message' not in st.session_state:
        st.session_state.upload_message = None

    def handle_upload():
        st.session_state.upload_clicked = True
    if st.session_state.df is not None:
        try:
            show_analysis_history()
            # Choose Upload Location
            with st.expander("Step 1: Choose Upload Location", expanded=True):
                fm, group, experiment, furthr = setup_upload_location()
                if all([fm,experiment]):
                    st.session_state.selected_location = {
                        'fm': fm,
                        'experiment': experiment,
                        'group': group
                    }
                    st.success(f"Selected folder: {experiment.name}")

            # Select Data to Upload
            if st.session_state.selected_location:
                with st.expander("Step 2: Select Data to Upload", expanded=True):
                    widget_key_prefix = f"upload_{int(time.time())}"
                    data_to_upload, analysis_types = get_upload_data(widget_key_prefix)

                    if data_to_upload is not None:
                        name = generate_filename(analysis_types, widget_key_prefix)
                        prepared_data = prepare_dataframe(data_to_upload)
                        # Preview
                        st.write("##### Preview of Selected Data:")
                        preview_df = data_to_upload.head(15)
                        display_df = preview_df.copy()
                        numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
                        for col in numeric_cols:
                            display_df[col] = display_df[col].apply(
                                lambda x: f'{float(x):.6f}'.replace(',', '.') if pd.notnull(x) else x)
                            display_df.index = range(1, len(display_df) + 1)
                        st.dataframe(display_df, use_container_width=True, height=350)

                        # File name and upload button
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.info(f"File will be uploaded as: **{name}**")
                        with col2:
                            upload_button = st.button(
                                "Upload Data",
                                key=f"{widget_key_prefix}_upload_button",
                                type="primary",
                                on_click=handle_upload
                            )
                        # Upload process
                        if st.session_state.upload_clicked:
                            st.session_state.upload_clicked = False
                            with st.spinner("Uploading data..."):
                                try:
                                    location = st.session_state.selected_location
                                    upload_success, message = upload_analyzed_data(
                                        analyzed_df=prepared_data,
                                        analysis_name=name,
                                        analysis_history=st.session_state.analysis_history,
                                        original_filename=st.session_state.get('original_filename'),
                                        fm=location['fm'],
                                        experiment=location['experiment'],
                                        group=location['group']
                                    )
                                    if upload_success:
                                        st.session_state.upload_message = message
                                        st.session_state.selected_location = None
                                        st.success(message)
                                    else:
                                        st.error(f"Upload failed: {message}")

                                except Exception as upload_error:
                                    st.error(f"Upload error: {str(upload_error)}")
                                    st.exception(upload_error)
                    else:
                        st.error("No data available for upload")

        except Exception as e:
            st.error(f"Setup error: {str(e)}")
            st.exception(e)
    else:
        st.warning("Please load and analyze data first before uploading.")