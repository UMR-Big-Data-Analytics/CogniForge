import streamlit as st
from cogniforge.utils.dataloader import DataLoader
from cogniforge.utils.furthr import FURTHRmind
from utils.data_analysis import analyze_spucker
from utils.plotting import plot_sampled, plot_xy_chart
from utils.state_button import button

st.set_page_config(
    page_title="CogniForge | Wire Quality",
    page_icon="🔌",
)

st.write("# Wire Quality")
st.write(
    "Welcome to the Wire Quality tool. Here you can analyze and visualize the quality of your wire."
)
if "fileName" not in st.session_state:
    st.session_state.fileName = None
with st.status("Download Data from FURTHRmind", expanded=True):
    downloader = FURTHRmind("download")
    downloader.file_extension = "csv"
    downloader.select_file()
    data, filename = downloader.download_string_button() or (None, None)
if data is not None:
    tabs = st.tabs(["Data Preview", "Plot Data", "Spucker Analysis"])
    with tabs[0]:
        dl = DataLoader(csv=data)
        st.session_state.fileName = filename
        df = dl.get_processedDataFrame()
    if data is not None and df is not None:
        filtered_columns = [col for col in df.columns if not col.lower().startswith(("zeit", "time"))]
        with tabs[1]:
                 plot_xy_chart(df)

        with tabs[2]:
            if df is not None:
                analyze_spucker(df)
            else:
                st.warning("Please load and process data first.")
