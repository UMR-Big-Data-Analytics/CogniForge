import streamlit as st
import plotly.express as px

def plot_wire_data():
    if st.session_state.df is not None:
        df = st.session_state.df
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.columns)
        fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig)
    else:
        st.warning("Load data first")