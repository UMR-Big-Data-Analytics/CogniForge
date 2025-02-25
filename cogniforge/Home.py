import streamlit as st

# Define all pages
home = st.Page("main.py", title="Home", icon="🏠", default=True)
timeseries = st.Page("Data_Preprocessing/Time_Series_Data.py", title="Time Series", icon="⏳")
image = st.Page("Data_Preprocessing/Image_Data.py", title="Image Data", icon="📩")
layer = st.Page("layer_quality/Layer_Thickness.py", title="Layer Thickness", icon="💦")
roughness = st.Page("steel_quality/Roughness.py", title="Roughness", icon="🗻")
rust = st.Page("steel_quality/Rust.py", title="Rust", icon="🧱")
wire = st.Page("wire_quality/Wire_Quality.py", title="Wire Quality", icon="🔌")
photo = st.Page("Photo.py", title="Photo Viewer", icon="🖼️")


pg = st.navigation({
    "": [home],
    "Data Preprocessing": [timeseries, image],
    "Wire Quality": [wire],
    "Roughness": [roughness, rust],
    "Layer Quality": [layer],
    "Misc": [photo]
})


pg.run()