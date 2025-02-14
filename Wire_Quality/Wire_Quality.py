import streamlit as st
from utils.data_analysis import analyze_spucker
from utils.plotting import plot_sampled
from utils.state_button import button

st.set_page_config(
    page_title="CogniForge | Wire Quality",
    page_icon="🔌",
)

st.write("# Wire Quality")
st.write(
    "Welcome to the Wire Quality tool. Here you can analyze and visualize the quality of your wire."
)

# use data loaded
if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("Please load data first in the Load Data section.")
else:
    df = st.session_state.df

    st.write("### Current Dataset")
    st.write(f"Dataset: {st.session_state.get('original_filename', 'Unknown')}")
    st.write(f"Total rows: {len(df):,}")

    with st.expander("Plot Data"):
        try:
            if button("Plot data", "plot_data_wire", True):
                plot_sampled(df)
        except Exception as e:
            st.error(f"Error plotting data: {str(e)}")

    with st.expander("Spucker Analysis"):
        if st.session_state.df is not None:
            analyze_spucker(st.session_state.df)
        else:
            st.warning("Please load and process data first.")