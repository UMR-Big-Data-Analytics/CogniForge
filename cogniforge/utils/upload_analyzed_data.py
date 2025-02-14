import os
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime
from furthrmind import Furthrmind as API
from furthrmind.collection import Experiment, Group
from furthrmind.file_loader import FileLoader

# test test
def show_analysis_history():
    """Display the history of analysis steps performed"""
    st.write("### Analysis Steps Performed")
    if st.session_state.analysis_history:
        for step in st.session_state.analysis_history:
            st.write(f"- {step}")
    else:
        st.write("No analysis steps performed yet")


def setup_upload_location():
    """Setup the FURTHRmind upload location"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text('Initializing...')
        progress_bar.progress(20)
        from utils.furthr import FURTHRmind
        furthr = FURTHRmind(id="upload_widget")
        st.write("### Select Upload Location")

        status_text.text('Loading project settings...')
        progress_bar.progress(60)

        col1, col2, col3 = st.columns(3)

        fm = None
        group = None
        experiment = None

        with col1:
            fm = furthr.setup_project()

        if fm:
            with col2:
                group = furthr.select_group()

            if group:
                with col3:
                    experiment = furthr.select_experiment(group)

        status_text.text('Ready for data selection')
        progress_bar.progress(100)
        import time
        time.sleep(0.1)

        progress_bar.empty()
        status_text.empty()

        return fm, group, experiment, furthr

    except Exception as e:
        st.error(f"Error setting up upload location: {str(e)}")
        return None, None, None, None


def prepare_dataframe(df):
    """Prepare dataframe for upload by formatting headers and numeric values"""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0]}[{col[1]}]" if pd.notna(col[1]) else col[0]
                      for col in df.columns]
    else:
        def format_header(col):
            col_str = str(col).strip('()"\' ')
            if '[' in col_str and ']' in col_str:
                return col_str
            parts = col_str.split('_')
            if len(parts) > 1:
                return f"{parts[0]}[{'_'.join(parts[1:])}]"
            return col_str

        df.columns = [format_header(col) for col in df.columns]

    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: str(x).replace('.', ',') if pd.notnull(x) else x)
    return df


def get_upload_data(widget_key_prefix=""):
    """Get data to upload based on user selection"""
    upload_choice = st.radio(
        "Choose data to upload:",
        ["Current State (with all analyses)",
         "Original Data",
         "Specific Analysis Steps"],
        key=f"{widget_key_prefix}_upload_data_selection"
    )

    if upload_choice == "Original Data":
        data_to_upload = st.session_state.original_df.copy()
        analysis_types = ["original"]
    elif upload_choice == "Specific Analysis Steps":
        if st.session_state.analysis_history:
            selected_steps = st.multiselect(
                "Select analysis steps to include:",
                st.session_state.analysis_history,
                default=st.session_state.analysis_history,
                key=f"{widget_key_prefix}_analysis_steps_selection"
            )
            data_to_upload = st.session_state.df.copy()
            analysis_types = []
            for step in selected_steps:
                if "Detrending applied to" in step:
                    analysis_types.append("detrend")
                elif "Smoothing applied to" in step:
                    analysis_types.append("smoothed")
            analysis_types = list(set(analysis_types))
        else:
            st.warning("No analysis steps available to select.")
            analysis_types = []
            data_to_upload = st.session_state.df.copy()
    else:
        data_to_upload = st.session_state.df.copy()
        analysis_types = []
        for step in st.session_state.analysis_history:
            if "Detrending applied to" in step:
                analysis_types.append("detrend")
            elif "Smoothing applied to" in step:
                analysis_types.append("smoothed")
        analysis_types = list(set(analysis_types))

    return data_to_upload, analysis_types


def generate_filename(analysis_types, widget_key_prefix=""):
    """Generate filename in format xxxx_detrend_smoothed_timestamp.csv"""
    original_filename = st.session_state.get('original_filename', "TS_PL_107")
    base_name = original_filename.split('_')[0:3]
    base_name = '_'.join(base_name)
    analysis_suffix = "_" + "_".join(analysis_types) if analysis_types else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    generated_filename = f"{base_name}{analysis_suffix}_{timestamp}.csv"
    st.text(f"Generated filename: {generated_filename}")

    return generated_filename


def upload_data():
    """Wrapper function for upload process"""
    try:
        import time
        widget_key_prefix = f"upload_{int(time.time())}"

        data_to_upload, analysis_types = get_upload_data(widget_key_prefix)
        if data_to_upload is None or data_to_upload.empty:
            st.error("No data available for upload. Please check your data source.")
            return False
        filename = generate_filename(analysis_types, widget_key_prefix)
        with st.spinner("Preparing and uploading data..."):
            upload_success, message = upload_analyzed_data(
                analyzed_df=data_to_upload,
                analysis_name=filename,
                analysis_history=st.session_state.get('analysis_history', []),
                original_filename=st.session_state.get('original_filename')
            )
        if upload_success:
            st.success(message)
            return True
        else:
            st.error(message)
            return False

    except Exception as e:
        st.error(f"An unexpected error occurred during upload: {str(e)}")
        st.exception(e)
        return False

def upload_analyzed_data(
        analyzed_df: pd.DataFrame,
        analysis_name: str = "analyzed_data",
        analysis_history: list = None,
        original_filename: str = None,
        fm: API = None,
        group: Group = None,
        experiment: Experiment = None
) -> tuple[bool, str]:
    """Upload analyzed data to FURTHRmind without repeating folder selection."""
    try:
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        # Carry out some checks
        if analyzed_df is None or analyzed_df.empty:
            raise ValueError("No data provided for upload")
        if not isinstance(analyzed_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if len(analyzed_df.columns) == 0:
            raise ValueError("DataFrame contains no columns")
        if not all([fm, group, experiment]):
            raise ValueError("Upload location information is missing")

        new_filename = analysis_name
        # Upload Process
        status_placeholder.text("Uploading data to FURTHRmind...")
        progress_placeholder.progress(0.6)
        with tempfile.TemporaryDirectory() as tmpdirname:
            csv_path = os.path.join(tmpdirname, new_filename)
            analyzed_df.to_csv(
                csv_path,
                index=False,
                sep=";",
                decimal=",",
                encoding='latin1'
            )
            file_loader = FileLoader(fm.host, fm.api_key)
            upload_response = file_loader.uploadFile(
                csv_path,
                parent={
                    "project": fm.project_id,
                    "type": "experiment",
                    "id": experiment.id,
                }
            )
            if not upload_response:
                raise ConnectionError("File upload failed - no response from server")

        progress_placeholder.progress(1.0)
        # Prepare success message
        success_message = (
            f"✅ Upload completed successfully!\n\n"
            f"File: {new_filename}\n"
        )
        return True, success_message
    except Exception as e:
        error_message = ""
        if isinstance(e, ValueError):
            error_message = f"Data validation error: {str(e)}"
        elif isinstance(e, ConnectionError):
            error_message = f"Connection error: {str(e)}\nPlease verify your internet connection and FURTHRmind server status."
        else:
            error_message = f"Upload failed: {str(e)}"

        return False, error_message


