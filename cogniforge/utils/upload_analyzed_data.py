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
    if st.session_state.analysis_history:
        with st.expander("Analysis History", expanded=False):
            for step in st.session_state.analysis_history:
                st.write(step)
    else:
        with st.expander("Analysis History", expanded=False):
            st.info("No analysis steps performed yet")


def setup_upload_location():
    """Setup the FURTHRmind upload location"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text('Initializing...')
        progress_bar.progress(20)
        from utils.furthr import FURTHRmind
        furthr = FURTHRmind(id="upload_widget")
        furthr.container_category = "experiment"
        st.write("### Select Upload Location")

        status_text.text('Loading project settings...')
        progress_bar.progress(60)
        furthr.select_container()
        status_text.text('Ready for data selection')
        progress_bar.progress(100)
        import time
        time.sleep(0.1)
        progress_bar.empty()
        status_text.empty()
        return furthr.fm, furthr.selected
    except Exception as e:
        st.error(f"Error setting up upload location: {str(e)}")
        return None, None


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


def get_upload_data():
    """Get data to upload based on user selection"""
    # FIXME upload_choice unused
    # upload_choice = st.radio(
    #     "Choose data to upload:",
    #     ["Current State (with all analyses)",
    #      "Original Data",
    #     "Specific Analysis Steps"]
    # )
    # TODO: the other 2 options
    if st.session_state.analysis_history:
        data_to_upload = st.session_state.df.copy()
        analysis_types = []
        if hasattr(st.session_state, 'detrend_steps') and st.session_state.detrend_steps:
            analysis_types.append("detrend")
        if hasattr(st.session_state, 'smoothing_steps') and st.session_state.smoothing_steps:
            analysis_types.append("smooth")
    else:
        st.warning("No analysis steps available to select.")
        analysis_types = []
        data_to_upload = st.session_state.df.copy()

    return data_to_upload, analysis_types

def generate_filename(analysis_types):
    """Generate filename based on analysis types"""
    original_filename = st.session_state.get('original_filename', "TS_PL_107")
    base_filename = original_filename.rsplit('.', 1)[0]
    # get file name
    if not analysis_types:
        suffix = "original"
    elif set(analysis_types) == {"detrend", "smooth"}:
        suffix = "analysed"
    elif "detrend" in analysis_types and "smooth" not in analysis_types:
        suffix = "detrended"
    elif "smooth" in analysis_types and "detrend" not in analysis_types:
        suffix = "smoothed"
    else:
        suffix = "_".join(analysis_types)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    generated_filename = f"{base_filename}_{suffix}_{timestamp}.csv"
    st.text(f"Upload filename: {generated_filename}")
    return generated_filename


def upload_analyzed_data(
        analyzed_df: pd.DataFrame,
        analysis_name: str = "analyzed_data",
        fm: API = None,
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
        if not all([fm, experiment]):
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


