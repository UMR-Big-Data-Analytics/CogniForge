import streamlit as st


home = st.Page(
    "pages/main.py", title="Home", icon="", default=True
)

timeseries = st.Page(
    "pages/1_data_preprocessing/Time_Series_Data.py", title="Time Series", icon=""
)

image = st.Page(
    "pages/1_data_preprocessing/Image_Data.py", title="Image Data", icon=""
)

layer = st.Page(
    "pages/4_layer_quality/Layer_Thickness.py", title="Layer Thickness", icon=""
)

roughness = st.Page(
    "pages/3_steel_quality/Roughness.py", title="Roughness", icon=""
)

rust = st.Page(
    "pages/3_steel_quality/Rust.py", title="Rust", icon=""
)

wire = st.Page(
    "pages/2_wire_quality/Wire_Quality.py", title="Wire Quality", icon=""
)



pg = st.navigation(
    {
        "🏠 Home" : [home],
        "⏩ Data Preprocessing" : [timeseries,image],
        "🔌 Wire Quality" : [wire],
        "🗻 Steel Quality" : [roughness,rust],
        "💦 Layer Quality" : [layer]
    }
)

pg.run()