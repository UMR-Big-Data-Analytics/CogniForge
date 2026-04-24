import streamlit as st

pg = st.navigation({
    "": [
        st.Page("main.py", title="Home", icon="🏠", default=True)
    ],
    "Data Preprocessing": [
        st.Page("Data_Preprocessing/Time_Series_Data.py", title="Time Series", icon="⏳")
    ],
    "Wire Quality": [
        st.Page("Wire_Quality/Wire_Quality.py", title="Wire Quality", icon="🔌")
    ],
    "Steel Surface": [
        st.Page("steel_quality/RustInference.py", title="Rust Detection - Inference", icon="🧱"),
        st.Page("steel_quality/RustTraining.py", title="Rust Detection - Training", icon="🧱"),
        st.Page("steel_quality/Roughness.py", title="Roughness Estimation", icon="🗻")
    ],
    "Layer Quality": [
        st.Page("layer_quality/Layer_Thickness.py", title="Layer Thickness", icon="💦")
    ],
    "Outlier Detection": [
        st.Page("FFT_Analysis/FFT_Analysis.py",title="FFT Analysis", icon="📈"),
        st.Page("FFT_Analysis/anomaly_detection.py", title="Anomaly Detection", icon="🚨")
    ],
    "Misc": [
        st.Page("Photo.py", title="Photo Viewer", icon="🖼️")
    ]
})

pg.run()