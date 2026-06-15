import numpy as np
import pandas as pd
import streamlit as st
from utils import furthr, ui
from utils.ml import CogniForgeModel, load_images

st.set_page_config(page_title="CogniForge | Rust Inference", page_icon="🧱")

st.markdown("""# Rust Detection - Inference

This page is for the Rust tool developed by Valerie Durbach.

You can *evaluate* an already trained model on a given image dataset
and analyze the results here.""")


def format_output(images_container, images_result, predictions):
    rust_image_count = np.count_nonzero(predictions)
    total_image_count = predictions.size
    rust_percent = rust_image_count * 100 / total_image_count

    df = pd.DataFrame({
        "Filename": [o[1] for o in images_result],
        "Has rust": predictions,
        "Link": ["/Photo?file_id=" + file.id for file in images_container.files]
    })
    df = df.sort_values(by="Has rust", ascending=False)
    return rust_percent, df


tab_data, tab_model, tab_prediction = st.tabs(["Data", "Model Selection", "Prediction Analysis"])


with tab_data:
    st.markdown("## Choose Images")
    images = ui.furthr_open_collection(
        key="image",
        kind=furthr.Sample,
        container_fielddata={
            'Image Width': "ANY",
            'Image Height': "ANY"
        },
        file_extension="tiff"
    )
    ui.resolution(images)


with tab_model:
    st.markdown("## Choose Model")

    if not images:
        st.markdown("Please select the data first. Then compatible models will be shown.")
    else:
        st.markdown("Only rust detection models compatible with the resolution of the selected data get shown below.")
        model = CogniForgeModel.open_dropdown(True, images)

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
        images_result, preprocessed_images = load_images(
            images,
            model.architecture,
            model.grayscale,
            model.pretrain,
            model.fft
        )
        predictions = model.predict(preprocessed_images)
        rust_percent, df = format_output(images.raw, images_result, predictions)
        collection = placeholder.create()
        collection.add_link_to(model.container)
        collection.add_link_to(images)
        collection.upload_content(df, "Inference Results.csv")

        st.info("The results have been stored in the database. You can also view them below.")
        st.metric(label="Rust Samples", value=f"{rust_percent:.1f} %")
        st.dataframe(
            df,
            column_config={
                "Link": st.column_config.LinkColumn(display_text="Open image")
            }
        )
    
    if is_prediction_blocked:
        st.markdown("Please select the data, a model and result destination first.")
