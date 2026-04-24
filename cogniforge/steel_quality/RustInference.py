import config
import numpy as np
import pandas as pd
import steel_quality.widgets as ui
import streamlit as st
from utils.ml import load_images, load_model, predict

st.set_page_config(page_title="CogniForge | Rust Inference", page_icon="🧱")

st.markdown("""# Rust Detection - Inference

This page is for the Rust tool developed by Valerie Durbach.

You can evaluate an already trained model on a given image dataset
and analyze the results here.""")


@st.cache_data
def format_output(_images_container, _images_result, _predictions, custom_cache_key):
    rust_image_count = np.count_nonzero(_predictions)
    total_image_count = _predictions.size
    rust_percent = rust_image_count * 100 / total_image_count

    df = pd.DataFrame({
        "Filename": [o[1] for o in _images_result],
        "Has rust": _predictions,
        "Link": ["/Photo?file_id=" + file.id for file in _images_container.files]
    })
    df = df.sort_values(by="Has rust", ascending=False)
    return rust_percent, df


tab_data, tab_model, tab_prediction = st.tabs(["Data", "Model Selection", "Prediction Analysis"])


with tab_data:
    st.markdown("## Choose Images")
    images = ui.furthr_selectbox(
        key="image",
        collection_type=ui.collection.Sample,
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
        model = ui.furthr_selectbox(
            key="model",
            collection_type=ui.collection.ResearchItem,
            collection_category="Code",
            container_fielddata={
                'Model Purpose': "Rust Detection",
                'Image Width': images.image_width,
                'Image Height': images.image_height,
                'Model Architecture': "ANY",
                'Image Grayscaling': "ANY",
                'Pretrained Weights': "ANY",
                'Optimizer': "ANY",
                'Activation Function': "ANY",
                'Loss Function': "ANY"
            },
            force_group_id=config.furthr['model_group_id'],
            file_extension="tiff"
        )

        if model:
            st.markdown("### Model Properties")
            st.table({
                'Model Architecture': model.model_architecture,
                'Expected Resolution': f"{model.image_width}x{model.image_height} px",
                'Image Grayscaling': str(model.image_grayscaling),
                'Pretrained Weights': str(model.pretrained_weights),
                'Optimizer': model.optimizer,
                'Activation Function': model.activation_function,
                'Loss Function': model.loss_function
            })


with tab_prediction:
    is_prediction_blocked = not (images and model)

    if st.button("Predict", disabled=is_prediction_blocked):
        model2 = load_model(model.raw.files[0])
        images_result, preprocessed_images = load_images(
            images.raw,
            model.model_architecture,
            model.image_grayscaling,
            model.pretrained_weights,
            False
        )
        custom_cache_key = (model.raw.id, images.raw.id)
        predictions = predict(model2, preprocessed_images, True, custom_cache_key)
        rust_percent, df = format_output(images.raw, images_result, predictions, custom_cache_key)

        st.metric(label="Rust Samples", value=f"{rust_percent:.1f} %")
        st.dataframe(
            df,
            column_config={
                "Link": st.column_config.LinkColumn(display_text="Open image")
            }
        )
    
    if is_prediction_blocked:
        st.markdown("Please select the data and a model first.")
