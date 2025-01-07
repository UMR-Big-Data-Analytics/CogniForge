import streamlit as st

# Define all pages
home = st.Page("main.py", title="Home", icon="🏠", default=True)
timeseries = st.Page("Data_Preprocessing/Time_Series_Data.py", title="Time Series", icon="⏳")
image = st.Page("Data_Preprocessing/Image_Data.py", title="Image Data", icon="📩")
layer = st.Page("layer_quality/3_💦_Layer_Thickness.py", title="Layer Thickness", icon="💦")
roughness = st.Page("Roughness/2_🗻_Roughness.py", title="Roughness", icon="🦿")
wire = st.Page("Wire_Quality/1_🔌_Wire_Quality.py", title="Wire Quality", icon="🔌")

# Primary navigation
pg = st.navigation({
    "": [home],
    "Data Preprocessing": [timeseries, image],
    "Wire Quality": [wire],
    "Roughness": [roughness],
    "Layer Quality": [layer],
})

# Run primary navigation
pg.run()