import streamlit as st
from datetime import datetime
from furthrmind import Furthrmind as API
from furthrmind.file_loader import FileLoader

#test #test....
if 'ts_subpage' not in st.session_state:
    st.session_state.ts_subpage = "Overview"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
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
if 'data_info' not in st.session_state:
    st.session_state.data_info = {
        'use_full_dataset': True,
        'total_rows': 0
    }
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = None


def handle_revert():
    """Handle revert to original dataset"""
    st.session_state.df = st.session_state.original_df.copy()
    st.session_state.current_analysis_type = None
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    revert_message = f"Data reverted to original state ({timestamp})"
    st.session_state.analysis_history.append(revert_message)
    if st.session_state.ts_subpage == "Detrend Data":
        st.session_state.detrend_steps = []
        st.session_state.detrending_active = False
    elif st.session_state.ts_subpage == "Smooth Data":
        st.session_state.smoothing_steps = []
        st.session_state.smoothing_active = False


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
    if st.button("📉 Smooth Data", use_container_width=True, key="ts_smooth"):
        st.session_state.ts_subpage = "Smooth Data"
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
           - **Smooth Data**: Smooth noisy data using moving average
           - **Upload Results**: Save your analyzed data back to FURTHRmind
           """)

# Configure each subpage
if st.session_state.ts_subpage == "Load Data":
    st.title("📥 Load Data")
    if st.session_state.df is not None:
        info_message = ["📊 Current Dataset Information:"]
        if 'original_filename' in st.session_state:
            info_message.append(f"• Current dataset: {st.session_state.original_filename}")
        if st.session_state.use_full_dataset:
            info_message.append(f"• Using complete dataset ({len(st.session_state.df):,} rows)")
        else:
            info_message.append(f"• Using selected range ({len(st.session_state.df):,} rows)")

        if st.session_state.detrending_active:
            info_message.append("• Active detrending analysis in progress")
        if st.session_state.smoothing_active:
            info_message.append("• Active smoothing analysis in progress")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.info('\n'.join(info_message))
        with col2:
            if st.button("Clear Current Data", type="primary"):
                keys_to_clear = [
                    'df', 'original_df', 'detrending_active', 'smoothing_active',
                    'analysis_history', 'original_filename', 'current_dataset_name',
                    'detrend_steps', 'smoothing_steps', 'current_analysis_type'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    st.write("Download your data from the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database.")
    try:
        from utils.dataloader import DataLoader
        from utils.furthr import FURTHRmind
        data = None
        with st.expander("Step 1: Download Data", expanded=True):
            with st.spinner('Downloading data...'):
                data, filename = FURTHRmind().download_csv() or (None, None)
                if data is not None:
                    is_new_dataset = ('original_filename' in st.session_state and
                                      st.session_state.original_filename != filename)
                    has_active_analysis = (st.session_state.get('detrending_active', False) or
                                           st.session_state.get('smoothing_active', False))

                    if is_new_dataset:
                        st.info(f"New dataset selected: {filename}")

                    st.session_state.original_filename = filename
                    data.name = filename
                    st.success("Data downloaded!")

        if data is not None:
            with st.expander("Step 2: Process Data", expanded=False):
                with st.spinner('Processing...'):
                    dl = DataLoader(csv=data)
                    df = dl.get_dataframe()
                    if df is not None and not df.empty:
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        if is_new_dataset:
                            st.session_state.analysis_history = []
                            st.session_state.detrending_active = False
                            st.session_state.smoothing_active = False
                            st.session_state.detrend_steps = []
                            st.session_state.smoothing_steps = []
    except ImportError as e:
        st.error(f"Import error: {e}. Please ensure that the `cogniforge` package is correctly installed.")
    except Exception as e:
        st.error(f"An error occurred: {e}")



elif st.session_state.ts_subpage == "Plot Data":
    from utils.plotting import plot_sampled
    st.title("📊 Plot Data")
    if st.session_state.df is not None:
        st.subheader("Data Visualization")
        plot_sampled(st.session_state.df)
    else:
        st.warning("Please load and process data first in the Load Data section.")



elif st.session_state.ts_subpage == "Detrend Data":
    from utils.detrending import analyze_detrend
    st.title("📉 Detrend Data")
    if st.session_state.df is not None:
        st.session_state.data_info['total_rows'] = len(st.session_state.df)
        try:
            detrended_df = analyze_detrend(st.session_state.df)
            if detrended_df is not None:
                st.session_state.df = detrended_df
                if st.session_state.detrend_steps:
                    st.session_state.current_analysis_type = "detrend" #set analysis type
                    st.write("### Detrending Summary")
                    st.write(f"Total columns detrended: {len(st.session_state.detrend_steps)}")
                    # Option to revert
                    if st.button("Revert to Original Data"):
                        handle_revert()
                        st.success("Successfully reverted to original data")
                        st.rerun()

        except Exception as e:
            st.error(f"An error occurred during detrending: {str(e)}")
            st.exception(e)
    else:
        st.warning("Please load and process data first in the Load Data section.")


elif st.session_state.ts_subpage == "Smooth Data":
    from utils.smoothing import analyze_smooth
    st.title("📉 Smooth Data")
    if st.session_state.df is not None:
        st.session_state.data_info['total_rows'] = len(st.session_state.df)
        try:
            # smoothing analysis
            smoothed_df = analyze_smooth(st.session_state.df)
            if smoothed_df is not None:
                st.session_state.df = smoothed_df
                # Option to revert
                if st.session_state.smoothing_steps:
                    st.session_state.current_analysis_type = "smooth"
                    if st.button("Revert to Original Data"):
                        handle_revert()
                        st.success("Successfully reverted to original data")
                        st.write("### Smoothing Summary")
                        st.write("Total columns smoothed: 0")
                        st.rerun()

                if st.session_state.smoothing_steps:
                        st.write("### Smoothing Summary")
                        st.write(f"Total columns smoothed: {len(st.session_state.smoothing_steps)}")
        except Exception as e:
            st.error(f"An error occurred during smoothing: {str(e)}")
            st.exception(e)
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
                if all([fm, group, experiment]):
                    st.session_state.selected_location = {
                        'fm': fm,
                        'group': group,
                        'experiment': experiment
                    }
                    st.success(f"Selected folder: {group.name}")

            # Select Data to Upload
            if st.session_state.selected_location:
                with st.expander("Step 2: Select Data to Upload", expanded=True):
                    widget_key_prefix = f"upload_{int(time.time())}"
                    data_to_upload, analysis_types = get_upload_data(widget_key_prefix)

                    if data_to_upload is not None:
                        name = generate_filename(analysis_types, widget_key_prefix)
                        prepared_data = prepare_dataframe(data_to_upload)
                        # Preview
                        st.write("### Preview of Selected Data:")
                        st.dataframe(prepared_data.head(), use_container_width=True)
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
                                        group=location['group'],
                                        experiment=location['experiment']
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