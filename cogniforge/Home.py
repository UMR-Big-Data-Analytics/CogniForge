import streamlit as st


home = st.Page(
    "main.py", title="Home", icon="🏠", default=True
)

timeseries = st.Page(
    "data_preprocessing/Time_Series_Data.py", title="Time Series", icon="⏳"
)

image = st.Page(
    "data_preprocessing/Image_Data.py", title="Image Data", icon="📩"
)

layer = st.Page(
    "layer_quality/Layer_Thickness.py", title="Layer Thickness", icon="📚"
)

roughness = st.Page(
    "steel_quality/Roughness.py", title="Roughness", icon="🦿"
)

rust = st.Page(
    "steel_quality/Rust.py", title="Rust", icon="💨"
)

wire = st.Page(
    "wire_quality/Wire_Quality.py", title="Wire Quality", icon="🔌"
)



pg = st.navigation(
    {
        "" : [home],
        "Data Preprocessing" : [timeseries,image],
        "Wire Quality" : [wire],
        "Steel Quality" : [roughness,rust],
        "Layer Quality" : [layer]
    }
)

pg.run()