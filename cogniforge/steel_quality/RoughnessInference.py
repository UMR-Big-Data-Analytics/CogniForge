import numpy as np
import pandas as pd
import streamlit as st
from utils import furthr, ui
from utils.ml import CogniForgeModel, load_images

st.set_page_config(page_title="CogniForge | Roughness Inference", page_icon="🗻")

st.markdown("""# Roughness Estimation - Inference

This page is for the Roughness tool developed by Valerie Durbach.

You can *evaluate* an already trained model on a given image dataset
and analyze the results here.""")


def format_output(ids, names, predictions):
    min_roughness = np.min(predictions)
    max_roughness = np.max(predictions)
    avg_roughness = np.average(predictions)

    df = pd.DataFrame({
        "Filename": names,
        "Roughness": predictions,
        "Link": ["/Photo?file_id=" + o for o in ids]
    })
    df = df.sort_values(by="Roughness", ascending=False)
    return min_roughness, max_roughness, avg_roughness, df


tab_data, tab_model, tab_prediction = st.tabs(["Data", "Model Selection", "Prediction Analysis"])


with tab_data:
    st.markdown("""## Choose Images without Rust

The datasets listed below are known to contain images of stainless steel.""")
    images = ui.furthr_open_collection(
        key="image",
        kind=furthr.Sample,
        container_fielddata={
            'Image Width': "ANY",
            'Image Height': "ANY",
            'Data Label': "NoRust"
        },
        file_extension="tiff"
    )
    ui.resolution(images)


with tab_model:
    st.markdown("## Choose Model")

    if not images:
        st.markdown("Please select the data first. Then compatible models will be shown.")
    else:
        st.markdown("Only roughness estimation models compatible with " \
        "the resolution of the selected data get shown below.")
        model = CogniForgeModel.open_dropdown(False, images)

        if model:
            model.render_settings()


with tab_prediction:
    st.markdown("## Choose where to store Results")
    placeholder = ui.furthr_save_collection(
        "dest",
        furthr.ResearchItem,
        "Analysis"
    )
    st.markdown("## Run Prediction")
    is_prediction_blocked = not (images and model and placeholder)

    if st.button("Predict", disabled=is_prediction_blocked):
        model.download()
        ids, names, X, _ = load_images(
            False,
            images,
            model.architecture,
            model.grayscale,
            model.pretrain,
            model.fft
        )
        predictions = model.predict(X)
        min_roughness, max_roughness, avg_roughness, df = format_output(ids, names, predictions)
        collection = placeholder.create()
        collection.add_link_to(model.container)
        collection.add_link_to(images)
        collection.upload_content(df, "Inference Results.csv")

        st.info("The results have been stored in the database. You can also view them below.")
        col1, col2 = st.columns(2)
        col1.metric(label="Minimum Roughness", value=f"{min_roughness:.2f} μm")
        col2.metric(label="Maximum Roughness", value=f"{max_roughness:.2f} μm")
        st.metric(label="Average Roughness", value=f"{avg_roughness:.2f} μm")
        st.dataframe(
            df,
            column_config={
                "Link": st.column_config.LinkColumn(display_text="Open image")
            }
        )
    
    if is_prediction_blocked:
        st.markdown("Please select the data, a model and result destination first.")
