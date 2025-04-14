import streamlit as st

# Define all pages
home = st.Page("main.py", title="Home", icon="🏠", default=True)
timeseries = st.Page("Data_Preprocessing/Time_Series_Data.py", title="Time Series", icon="⏳")
image = st.Page("Data_Preprocessing/Image_Data.py", title="Image Data", icon="📩")
layer = st.Page("layer_quality/Layer_Thickness.py", title="Layer Thickness", icon="💦")
roughness = st.Page("steel_quality/Roughness.py", title="Roughness Estimation", icon="🗻")
rust = st.Page("steel_quality/Rust.py", title="Rust Detection", icon="🧱")
wire = st.Page("Wire_Quality/Wire_Quality.py", title="Wire Quality", icon="🔌")
photo = st.Page("Photo.py", title="Photo Viewer", icon="🖼️")
fft = st.Page("FFT_Analysis/FFT_Analysis.py",title="FFT Analysis", icon="📈")

pg = st.navigation({
    "": [home],
    "Data Preprocessing": [timeseries, image],
    "Wire Quality": [wire],
    "Steel Surface": [rust, roughness],
    "Layer Quality": [layer],
    "FFT Analysis": [fft],
    "Misc": [photo]
})

pg.run()