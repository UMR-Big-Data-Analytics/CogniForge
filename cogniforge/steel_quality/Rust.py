import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.furthr import FURTHRmind
import config

st.set_page_config(page_title="CogniForge | Rust", page_icon="⛓️")

st.write("""# Rust Prediction

This page is for the Rust tool developed by Valerie Durbach.""")


def load_and_preprocess_data(images_bytes, normalize, GrayScale, pretrain, name):
    image_arrays = []

    # loading the images and extracting Rz values as labels
    for img_file in images_bytes:
        with Image.open(img_file) as im:
            if GrayScale:
                im_array = np.array(im.convert("RGB"))
                im = tf.image.rgb_to_grayscale(im_array).numpy()
            image_arrays.append(np.array(im))

    X = image_arrays
    X = np.array([np.array(val) for val in X])

    if pretrain:
        # FIXME
        preprocess_function = getattr(tf.keras.applications, name).preprocess_input
        X = preprocess_function(X)
    else:
        if normalize:
            X = X / 225.0

    return X


def predict(model, X, classification):
    # Make predictions
    predictions = np.asarray(model.predict(X))

    if classification:
        # only for the Classification Task
        predictions = np.argmax(predictions, axis=1)

    return predictions

tab_data, tab_training, tab_model, tab_prediction = st.tabs(["Data", "Model Training", "Model Selection", "Prediction Analysis"])

with tab_data:
    st.write("## Choose Images")
    images_widget = FURTHRmind(id="image")
    images_widget.file_extension = "tiff"
    images_widget.container_category = "sample"
    images_widget.expected_fielddata = {
        'Image Width': "ANY",
        'Image Height': "ANY"
    }
    images_widget.select_container()

    if images_widget.selected:
        for single_field in images_widget.selected.fielddata:
            if single_field.field_name == 'Image Width':
                images_width = single_field.value
            if single_field.field_name == 'Image Height':
                images_height = single_field.value

with tab_training:
    st.write("Not implemented yet.")

with tab_model:
    st.write("## Choose Model")
    model_widget = FURTHRmind(id="model")

    if images_widget.selected:
        st.write("Only rust prediction models compatible with the resolution of the selected data get shown below.")
        model_widget.file_extension = "keras"
        model_widget.force_group_id = config.furthr['model_group_id']
        model_widget.container_category = "Code"
        model_widget.expected_fielddata = {
            'Model Purpose': "Rust Prediction",
            'Image Width': images_width,
            'Image Height': images_height,
            'Model Architecture': "ANY",
            'Data Normalization': "ANY",
            'Image Grayscaling': "ANY",
            'Data Preprocessing': "ANY"
        }
        model_widget.select_container()
    else:
        st.write("Please select the data first. Then compatible models will be shown.")

with tab_prediction:
    is_prediction_blocked = not (images_widget.selected and model_widget.selected)

    if st.button("Predict", disabled=is_prediction_blocked):
        with st.spinner("Loading model... (1/4)"):
            model_bytes, _ = model_widget.download_bytes(model_widget.selected.files[0], confirm_load=False)

            for single_field in model_widget.selected.fielddata:
                if single_field.field_name == 'Model Architecture':
                    model_name = single_field.value
                elif single_field.field_name == 'Data Normalization':
                    normalize = single_field.value
                elif single_field.field_name == 'Image Grayscaling':
                    grayscale = single_field.value
                elif single_field.field_name == 'Data Preprocessing':
                    pretrained = single_field.value

            with tempfile.NamedTemporaryFile(delete_on_close=False, suffix=".keras") as fh:
                fh.write(model_bytes.getvalue())
                fh.close()
                model = tf.keras.models.load_model(fh.name)

        with st.spinner("Loading images... (2/4)"):
            images_result = images_widget.download_bytes(confirm_load=False)
            images_bytes = [o[0] for o in images_result]
            preprocessed_images = load_and_preprocess_data(
                images_bytes, normalize, grayscale, pretrained, model_name
            )

        with st.spinner("Running prediction... (3/4)"):
            predictions = predict(model, preprocessed_images, classification=True)

        with st.spinner("Preparing output... (4/4)"):
            rust_image_count = np.count_nonzero(predictions)
            total_image_count = predictions.size
            rust_percent = rust_image_count * 100 / total_image_count

            df = pd.DataFrame({
                "Filename": [o[1] for o in images_result],
                "Has rust": predictions,
                "Link": ["/Photo?file_id=" + file.id for file in images_widget.selected.files]
            })
            df = df.sort_values(by="Has rust", ascending=False)

        st.metric(label="Rust Samples", value=f"{rust_percent:.1f} %")
        st.dataframe(
            df,
            column_config={
                "Link": st.column_config.LinkColumn(display_text="Open image")
            }
        )
    
    if is_prediction_blocked:
        st.write("Please select the data and a model first.")
