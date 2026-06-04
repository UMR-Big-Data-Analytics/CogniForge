import itertools

import numpy as np
import pandas as pd
import steel_quality.widgets as ui
import streamlit as st
from sklearn.model_selection import train_test_split

from cogniforge.utils.ml import (
    AVAILABLE_ACTIVATIONS,
    AVAILABLE_ARCHITECTURES,
    AVAILABLE_LOSSES,
    AVAILABLE_OPTIMIZERS,
    AVAILABLE_POOLING,
    build_model,
    load_images,
    train_model,
)

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


def generate_setting_combinations(form: dict[str, list]):
    return itertools.product(
        form['Model Architecture'],
        form['Loss Function'],
        form['Activation Function'],
        form['Optimizer'],
        form['Image Grayscaling'],
        form['Pretrained Weights'],
        form['Pooling Mode']
    )


tab_data, tab_model, tab_training = st.tabs(["Data", "Model Config", "Training Process"])


with tab_data:
    st.markdown("""## Choose Images with Rust

The datasets listed below are known to contain images of rusty steel.""")
    rusty = ui.furthr_open_collection(
        key="rust",
        kind=ui.collection.Sample,
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
            kind=ui.collection.Sample,
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
    st.markdown("## Choose Model Settings")
    settings = ui.form("settings", {
        'Model Architecture': AVAILABLE_ARCHITECTURES,
        'Image Grayscaling': [True, False],
        'Pretrained Weights': [True, False],
        'Optimizer': AVAILABLE_OPTIMIZERS,
        'Activation Function': AVAILABLE_ACTIVATIONS,
        'Loss Function': AVAILABLE_LOSSES,
        'Pooling Mode': AVAILABLE_POOLING
    })


with tab_training:
    is_training_blocked = not (stainless and settings)

    if st.button("Train", disabled=is_training_blocked):
        # for each setting building the corresponding model and evaluate the model
        combinations = generate_setting_combinations(settings)
        total = 0

        for architecture, loss, activation, optimizer, grayscale, pretrain, pool in combinations:
            ui.log(f"""Selecting the following training settings:

**Model Architecture:** {architecture}  
**Image Grayscaling:** {grayscale}  
**Pretrained Weights:** {pretrain}  
**Optimizer:** {optimizer}  
**Activation Function:** {activation}  
**Loss Function:** {loss}  
**Pooling Mode:** {pool}""")
            ui.log("Preparing images and new model for training ...")
            # preprocess the data for training and testing
            rusty_raw, rusty_preprocessed = load_images(
                rusty,
                architecture,
                grayscale,
                pretrain,
                False
            )
            stainless_raw, stainless_preprocessed = load_images(
                stainless,
                architecture,
                grayscale,
                pretrain,
                False
            )
            input_size = rusty_preprocessed[0].shape
            X = np.append(rusty_preprocessed, stainless_preprocessed, axis=0)
            del rusty_preprocessed
            del stainless_preprocessed
            Y = np.empty(len(rusty_raw) + len(stainless_raw), np.uint8)
            Y[:len(rusty_raw)] = 1
            Y[len(rusty_raw):] = 0
            # Splitting the data into train,test and validation data, with random_state= 42 to ensure that every model
            # will be trained with the same data for better comparison.
            X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, random_state=42)
            X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=0.5, random_state=42)
            # building the corresponding model
            model = build_model(architecture, input_size, activation, optimizer, loss, pretrain, pool)
            ui.log("Running training process. This can take a long time.")
            history = train_model(X_train, Y_train, X_val, Y_val, model)
            ui.log("Finished training using previously selected settings.")
            total += 1

        ui.log(f"Trained {total} different models. Overall process complete.")
    
    if is_training_blocked:
        st.markdown("Please select the data and configure model settings first.")
