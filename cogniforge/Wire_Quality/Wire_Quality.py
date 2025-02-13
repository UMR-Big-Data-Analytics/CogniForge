import streamlit as st
from cogniforge.algorithms.anomaly_detection import SpuckerCounter
from cogniforge.utils.plotting import plot_sampled
from cogniforge.utils.state_button import button

st.set_page_config(
    page_title="CogniForge | Wire Quality",
    page_icon="🔌",
)

st.write("# Wire Quality")
st.write(
    "Welcome to the Wire Quality tool. Here you can analyze and visualize the quality of your wire."
)

# Check if data is available in session state
if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("Please load data first in the Load Data section.")
else:
    df = st.session_state.df

    st.write("### Current Dataset")
    st.write(f"Dataset: {st.session_state.get('original_filename', 'Unknown')}")
    st.write(f"Total rows: {len(df):,}")

    with st.expander("Plot Data"):
        try:
            if button("Plot data", "plot_data_wire", True):
                if not df.empty and len(df) > 0:
                    plot_sampled(df)
                else:
                    st.warning("Cannot plot empty dataset")
        except Exception as e:
            st.error(f"Error plotting data: {str(e)}")

    with st.expander("Spucker Analysis"):
        st.markdown("""
        ### What is a Spucker?
        A Spucker is an anomaly detection method used to identify unexpected or unusual data points 
        in a time series. The analysis considers:
        - **Threshold**: A critical value above which a point is considered anomalous
        - **Context Window**: Number of surrounding points to consider
        - **Ignore Range**: Specific sections of data to exclude from analysis
        """)

        if len(df) > 0:
            st.write("## Spucker Count Configuration")
            column = st.selectbox("Select column for analysis", df.columns)
            column_idx = df.columns.get_loc(column)

            # Safely handle min and max values
            column_data = df.iloc[:, column_idx]
            max_val = float(column_data.max()) if not column_data.empty else 70.0

            threshold = st.slider(
                "Threshold Value",
                0.0,
                max_val,
                min(70.0, max_val),
                help="Points above this value will be considered for anomaly detection"
            )

            # Safely handle index range
            if len(df) > 1:
                index_range = st.slider(
                    "Analysis Range",
                    0.0,
                    float(df.index[-1]),
                    (0.0, float(df.index[-1])),
                    help="Select the specific range of data to analyze"
                )
                context_window = st.slider(
                    "Context Window Size",
                    0,
                    int(df.index[-1]),
                    (1, min(5, int(df.index[-1]))),
                    help="Number of surrounding points to consider when detecting anomalies"
                )

                # Visualization to help understand parameters
                st.write("### Parameter Explanation")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"📊 Selected Column: **{column}**")
                    st.write(f"🔺 Threshold: **{threshold}**")
                with col2:
                    st.write(f"📍 Analysis Range: **{index_range[0]}** to **{index_range[1]}**")
                    st.write(f"🔍 Context Window: **{context_window}**")

                if button("Start Spucker Analysis", "count_spucker", True):
                    with st.spinner("Analyzing data for Spuckers..."):
                        counter = SpuckerCounter(
                            threshold=threshold,
                            ignore_begin_seconds=index_range[0],
                            ignore_end_seconds=df.index[-1] - index_range[1],
                            context_window=context_window,
                        )

                        count = counter.count(df.iloc[:, column_idx])

                        # Detailed result interpretation
                        st.success(f"🔍 Spucker Count Analysis Complete")

                        # Interpret the results with context
                        if count == 0:
                            st.info("No significant anomalies detected in the selected range.")
                        elif count < 3:
                            st.warning(f"Low number of anomalies detected: {count}")
                        else:
                            st.error(f"High number of anomalies detected: {count}")

                        # Additional guidance
                        st.markdown("""
                        ### Understanding the Results
                        - **0 Spuckers**: Consistent data, minimal unexpected variations
                        - **1-2 Spuckers**: Occasional minor anomalies
                        - **3+ Spuckers**: Significant data inconsistencies

                        💡 **Tip**: Adjust threshold and context window to fine-tune detection
                        """)
            else:
                st.warning("Not enough data for Spucker analysis")
        else:
            st.warning("No data available for analysis")