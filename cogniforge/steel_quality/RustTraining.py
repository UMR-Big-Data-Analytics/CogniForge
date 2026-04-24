import numpy as np
import pandas as pd
from cogniforge.utils.ml import AVAILABLE_ACTIVATIONS, AVAILABLE_ARCHITECTURES, AVAILABLE_LOSSES, AVAILABLE_OPTIMIZERS
import steel_quality.widgets as ui
import streamlit as st

st.set_page_config(page_title="CogniForge | Rust Training", page_icon="🧱")

st.markdown("""# Rust Detection - Training

This page is for the Rust tool developed by Valerie Durbach.

You can *train* a new image classification model here.""")


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


tab_data, tab_model, tab_training = st.tabs(["Data", "Model Config", "Training Process"])


with tab_data:
    st.markdown("""## Choose Images with Rust

The datasets listed below are known to contain images of rusty steel.""")
    rusty = ui.furthr_selectbox(
        key="rust",
        collection_type=ui.collection.Sample,
        container_fielddata={
            'Data Label': 'Rust',
            'Image Width': "ANY",
            'Image Height': "ANY"
        },
        file_extension="tiff"
    )
    stainless = None

    if rusty:
        ui.resolution(rusty)
        st.markdown("""## Choose Images without Rust

Only those image datasets can be selected below,
which have the *same resolution* as the previous.""")
        stainless = ui.furthr_selectbox(
            key="stainless",
            collection_type=ui.collection.Sample,
            container_fielddata={
                'Data Label': 'NoRust',
                'Image Width': rusty.image_width,
                'Image Height': rusty.image_height
            },
            file_extension="tiff"
        )

        if stainless:
            st.markdown("All necessary training data selected. Proceed to the next tab.")


with tab_model:
    st.markdown("## Choose Model Settings")
    model = ui.form("settings", {
        'Model Architecture': AVAILABLE_ARCHITECTURES,
        'Image Grayscaling': [True, False],
        'Pretrained Weights': [True, False],
        'Optimizer': AVAILABLE_OPTIMIZERS,
        'Activation Function': AVAILABLE_ACTIVATIONS,
        'Loss Function': AVAILABLE_LOSSES,
        'Pooling (????????)': [None]
    })


with tab_training:
    is_training_blocked = not (stainless and model)

    if st.button("Train", disabled=is_training_blocked):
        pass
    
    if is_training_blocked:
        st.markdown("Please select the data and configure model settings first.")
