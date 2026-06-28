import streamlit as st
from steel_quality.TrainingTab import training_tab_content
from utils import furthr, ui
from utils.ml import CogniForgeModel

st.set_page_config(page_title="CogniForge | Rust Training", page_icon="🧱")

st.markdown("""# Rust Detection - Training

This page is for the Rust tool developed by Valerie Durbach.

You can *train* a new image classification model here.""")


tab_data, tab_model, tab_training = st.tabs(["Data", "Model Config", "Training Process"])


with tab_data:
    st.markdown("""## Choose Images with Rust

The datasets listed below are known to contain images of rusty steel.""")
    rusty = ui.furthr_open_collection(
        key="rust",
        kind=furthr.Sample,
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
        stainless = ui.furthr_open_collection(
            key="stainless",
            kind=furthr.Sample,
            container_fielddata={
                'Data Label': 'NoRust',
                'Image Width': rusty.image_width,
                'Image Height': rusty.image_height
            },
            file_extension="tiff"
        )

        if stainless:
            st.success("All necessary training data selected. Proceed to the next tab.")


with tab_model:
    models = CogniForgeModel.open_form(True, rusty)


with tab_training:
    training_tab_content(models, [stainless])
