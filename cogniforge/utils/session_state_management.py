import streamlit as st
import pandas as pd
"""
Problem: downsample df is not updated in other pages
- When downsample occurs: (1) flag is raised, (2) current df is updated to the downsamled version
- Subsequent analyses will be done on downsampled data if downsamped has been performed
"""

def update_session_state(df, dataset_name=None, analysis_type=None):
    """
    Update the current dataset while preserving original data and analysis states.
    """
    st.session_state.current_df = df.copy()
    # original processed df
    if 'original_df' not in st.session_state or st.session_state.original_df is None:
        st.session_state.original_df = df.copy()
    if dataset_name:
        st.session_state.current_dataset_name = dataset_name
    st.session_state.processing_complete = True
    # Handle downsampling specifically
    if analysis_type == 'downsample':
        st.session_state.is_downsampled = True
        st.session_state.downsampled_df = df.copy()
        st.session_state.current_df = df.copy()
        print(f"Downsampled: Rows = {len(df)}")
    # Handle other analyses
    if analysis_type == 'detrend':
        st.session_state.detrending_active = True
        st.session_state.smoothing_active = False
        st.session_state.downsampling_active = False
    elif analysis_type == 'smooth':
        st.session_state.detrending_active = False
        st.session_state.smoothing_active = True
        st.session_state.downsampling_active = False
    # Ensure downsampled state is preserved across analyses
    if st.session_state.get('is_downsampled', False):
        downsampled_df = st.session_state.downsampled_df.copy()
        if analysis_type in ['detrend', 'smooth']:
            st.session_state.current_df = df.copy()
            print(f"df: Rows = {len(downsampled_df)}")

    st.session_state.full_analyzed_df = df.copy()
    st.session_state.last_analysis_type = analysis_type