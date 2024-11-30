import sys
import streamlit as st
from pathlib import Path

current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from cogniforge.utils.dataloader import DataLoader
from cogniforge.utils.furthr import FURTHRmind
from cogniforge.utils.plotting import plot_sampled
from cogniforge.utils.data_analysis import analyze_spucker


st.set_page_config(page_title="CogniForge | Wire Quality", page_icon="🔌")


if 'df' not in st.session_state:
    st.session_state.df = None

st.title("Wire Quality")
st.write("Upload your data to the [FURTHRmind](https://furthr.informatik.uni-marburg.de/) database.")

# tab name for reference
tab1, tab2, tab3 , tab4= st.tabs(["Load Data", "Plot Data", "Data Analysis", "Time trend(preliminary)"])

# Load Data tab
with tab1:
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

# Plot Data tab
with tab2:
    if st.session_state.df is not None:
        plot_sampled(st.session_state.df)
    else:
        st.warning("Please load and process data first.")

# Data Analysis tab
with tab3:
    if st.session_state.df is not None:
        analyze_spucker(st.session_state.df)
    else:
        st.warning("Please load and process data first.")

# time trend analysis???
with tab4:
    if st.session_state.df is not None:
        analyze_spucker(st.session_state.df)
    else:
        st.warning("Please load and process data first.")