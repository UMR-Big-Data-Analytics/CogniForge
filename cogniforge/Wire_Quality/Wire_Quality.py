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


if 'current_df' in st.session_state and st.session_state.current_df is not None:
    df = st.session_state.current_df.copy()
else:
    df = None

if df is None:
    st.error("Please load data first using the Load Data Page.")
else:
    st.write("### Current Dataset")
    st.write(f"Dataset: {st.session_state.get('original_filename', 'Unknown')}")
    st.write(f"Total rows: {len(df):,}")

    # Plot Data Expander
    with st.expander("Plot Data"):
        try:
            if button("Plot data", "plot_data_wire", True):
                plot_sampled(df)
        except Exception as e:
            st.error(f"Error plotting data: {str(e)}")

    # Spucker Analysis Expander
    with st.expander("Spucker Analysis"):
        if df is not None:
            analyze_spucker(df)
        else:
            st.warning("Please load and process data first.")
