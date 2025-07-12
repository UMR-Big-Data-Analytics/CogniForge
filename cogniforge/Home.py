import streamlit as st

# Define all pages
home = st.Page("main.py", title="Home", icon="🏠", default=True)
timeseries = st.Page("data_preprocessing/time_series_data.py", title="Time Series", icon="⏳")
image = st.Page("data_preprocessing/image_data.py", title="Image Data", icon="📩")
layer = st.Page("layer_quality/layer_thickness.py", title="Layer Thickness", icon="💦")
anomaly = st.Page("fft_analysis/anomaly_detection.py", title="Anomaly Detection", icon="🚨")
roughness = st.Page("steel_quality/roughness.py", title="Roughness Estimation", icon="🗻")
rust = st.Page("steel_quality/rust.py", title="Rust Detection", icon="🧱")
wire = st.Page("wire_quality/wire_quality.py", title="Wire Quality", icon="🔌")
photo = st.Page("Photo.py", title="Photo Viewer", icon="🖼️")
fft = st.Page("fft_analysis/fft_analysis.py",title="FFT Analysis", icon="📈")

pg = st.navigation({
    "": [home],
    "Data Preprocessing": [timeseries, image],
    "Wire Quality": [wire],
    "Steel Surface": [rust, roughness],
    "Layer Quality": [layer],
    "Outlier Detection": [fft, anomaly],
    "Misc": [photo]
})

pg.run()