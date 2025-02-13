import os
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime
from furthrmind import Furthrmind as API
from furthrmind.collection import Experiment, Group
from furthrmind.file_loader import FileLoader

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
            analysis_types = [step.split()[0].lower() for step in selected_steps]
        else:
            st.warning("No analysis steps available to select.")
            analysis_types = []
            data_to_upload = st.session_state.df.copy()
    else:
        data_to_upload = st.session_state.df.copy()
        analysis_types = [step["method"].lower() for step in st.session_state.analysis_history if isinstance(step, dict) and "method" in step]


    return data_to_upload, analysis_types


def generate_filename(analysis_types, widget_key_prefix=""):
    """Generate filename in format xxxx_detrend_smoothed"""
    original_filename_key = f"{widget_key_prefix}_original_filename"
    generated_filename_key = f"{widget_key_prefix}_generated_filename"

    if not st.session_state.get('original_filename'):
        st.session_state.original_filename = st.text_input(
            "Original filename",
            value="TS_PL_107",
            key=original_filename_key
        )

    # Get base name without extension
    base_name = os.path.splitext(st.session_state.original_filename)[0]
    original_ext = os.path.splitext(st.session_state.original_filename)[1]

    # Create analysis suffix based on unique analysis types
    unique_analysis = []
    if 'detrend' in analysis_types:
        unique_analysis.append('detrend')
    if 'smoothing' in analysis_types:
        unique_analysis.append('smoothed')

    # Track analysis that has been performed & Create filena,e
    analysis_suffix = "_" + "_".join(unique_analysis) if unique_analysis else ""
    default_name = f"{base_name}{analysis_suffix}{original_ext}"

    return st.text_input(
        "File name",
        value=default_name,
        key=generated_filename_key
    )


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
            upload_success = upload_analyzed_data(
                analyzed_df=data_to_upload,
                analysis_name=filename,
                analysis_history=st.session_state.get('analysis_history', []),
                original_filename=st.session_state.get('original_filename')
            )
        if upload_success:
            st.success(f"✅ Data uploaded successfully as '{filename}'")
            return True
        else:
            st.error("Upload process did not complete successfully. Please try again.")
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
) -> bool:
    """Upload analyzed data to FURTHRmind without repeating folder selection."""
    try:
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        # Carry out some checkss
        if analyzed_df is None or analyzed_df.empty:
            raise ValueError("No data provided for upload")
        if not isinstance(analyzed_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if len(analyzed_df.columns) == 0:
            raise ValueError("DataFrame contains no columns")
        if not all([fm, group, experiment]):
            raise ValueError("Upload location information is missing")

        # Prepare filename
        status_placeholder.text("Preparing file for upload...")
        progress_placeholder.progress(0.4)
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base_name = os.path.splitext(original_filename)[0] if original_filename else analysis_name
        extension = os.path.splitext(original_filename)[1] if original_filename else '.csv'
        analysis_types = "_".join(step.split()[0].lower() for step in (analysis_history or []))
        new_filename = (f"{base_name}_&_{analysis_types}_{file_timestamp}{extension}"
                        if analysis_types else f"{base_name}_&_{file_timestamp}{extension}")

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

        # Calculate summary statistics
        total_rows = len(analyzed_df)
        non_null_rows = len(analyzed_df.dropna(how='all'))
        # Message
        st.success(
            f"✅ Upload completed successfully!\n\n"
            f"File: {new_filename}\n"
            f"Dataset Summary:\n"
            f"- Total rows: {total_rows:,}\n"
            f"- Data rows: {non_null_rows:,}\n"
            f"- Empty rows: {total_rows - non_null_rows:,}\n"
            f"- Columns: {len(analyzed_df.columns):,}"
        )
        return True
    except Exception as e:
        if isinstance(e, ValueError):
            st.error(f"Data validation error: {str(e)}")
        elif isinstance(e, ConnectionError):
            st.error(f"Connection error: {str(e)}")
            st.info("Please verify your internet connection and FURTHRmind server status.")
        else:
            st.error(f"Upload failed: {str(e)}")
            st.exception(e)

        return False


