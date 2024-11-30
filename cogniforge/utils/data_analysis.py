import streamlit as st

def analyze_wire_data():
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("### Statistical Summary")
        st.dataframe(df.describe())
    else:
        st.warning("Load data first")