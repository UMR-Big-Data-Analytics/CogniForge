import tempfile

import numpy as np
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
    for img_file in images_result:
        with Image.open(img_file) as im:
            if GrayScale:
                im_array = np.array(im.convert("RGB"))
                im = tf.image.rgb_to_grayscale(im_array).numpy()
            image_arrays.append(np.array(im))

    X = image_arrays
    X = np.array([np.array(val) for val in X])

    if pretrain:
        preprocess_function = getattr(tf.keras.applications, name).preprocess_input
        X = preprocess_function(X)
    else:
        if normalize:
            X = X / 225.0

    return X


def parse_model_name_and_normalize(filename):
    # Remove the .keras extension
    if filename.endswith(".keras"):
        model_string = filename[: -len(".keras")]
    else:
        # should never happen: prevented by filter on model selection widget
        raise ValueError("The file does not have a '.keras' extension.")

    # Split the string by hyphens to isolate components
    parts = model_string.split("-")

    # Extract the model name (assuming it is the first part)
    model_name = parts[0]

    # Determine normalization status directly
    normalize = "normalize-True" in parts
    grayscale = "GrayScale-True" in parts
    pretrained = "Pretrained-True" in parts

    return model_name, normalize, grayscale, pretrained


def predict(model, X, classification):
    # Make predictions
    predictions = np.asarray(model.predict(X))

    if classification:
        # only for the Classification Task
        predictions = np.argmax(predictions, axis=1)

    return predictions


col1, col2 = st.columns(2)
model = images_result = None # else last if can fail

with col1:
    st.write("## Choose Model")
    model_widget = FURTHRmind(id="model")
    model_widget.force_group_id = config.furthr['models']['group_id']
    model_widget.container_category = "Code"
    model_widget.expected_fielddata = {
        'Model Purpose': "Rust Prediction"
    }
    model_widget.select_container()

    if model_widget.selected is not None:
        model_result = model_widget.download_bytes(model_widget.selected.files[0])
        if model_result is not None:
            model_bytes, model_name = model_result

            with tempfile.NamedTemporaryFile(delete_on_close=False, suffix=".keras") as fh:
                fh.write(model_bytes.getvalue())
                fh.close()
                model = tf.keras.models.load_model(fh.name)
            st.write("load complete")

with col2:
    st.write("## Choose Images")
    images_widget = FURTHRmind(id="image")
    images_widget.file_extension = "tiff"
    images_widget.select_container()
    images_result = images_widget.download_bytes()
    if images_result is not None:
        images_bytes = [o[0] for o in images_result]
        st.write("loaded", len(images_bytes), "samples")

if images_result is not None and model is not None:
    if st.button("Predict"):
        model_name, normalize, grayscale, pretrained = parse_model_name_and_normalize(
            model_name
        )
        preprocessed_images = load_and_preprocess_data(
            images_bytes, normalize, grayscale, pretrained, model_name
        )

        predictions = predict(model, preprocessed_images, classification=True)
        it = np.nditer(predictions, flags=['c_index'])

        for prediction in it:
            if prediction:
                first_rust_index = it.index
                break

        if first_rust_index is None:
            st.write("Every patch is rustless.")
        else:
            rust_img = Image.open(images_bytes[first_rust_index])
            st.image(rust_img, caption="Sample found with rust")
