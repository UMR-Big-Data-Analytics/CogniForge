import streamlit as st

# Page definitions
st.set_page_config(
    page_title="CogniForge | Home",
    page_icon="🏠",
)

st.title("🏠 Welcome to CogniForge")
st.write("Explore the different tools available for data analysis and quality assessment.")

st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to navigate between pages:")

st.sidebar.markdown("""
- [Data Preprocessing](#data-preprocessing)
  - Time Series
  - Image Data
- [Wire Quality](#wire-quality)
- [Steel Quality](#steel-quality)
  - Roughness
  - Rust
- [Layer Quality](#layer-quality)
""")
