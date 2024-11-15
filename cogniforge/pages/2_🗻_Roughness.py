import json
import os
import sys
import tempfile
import time
import zipfile
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image

sys.path.append("../cogniforge")
from cogniforge.utils.furthr import FURTHRmind

load_dotenv()
st.set_page_config(page_title="CogniForge | Roughness", page_icon="🗻")

st.write("""# Roughness Prediction

This page is for the Roughness tool developed by Valerie Durbach.""")


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
    # Start inference timing
    start_time = time.time()

    # Make predictions
    predictions = np.asarray(model.predict(X))

    if classification:
        # only for the Classification Task
        predictions = np.argmax(predictions, axis=1)

    # Calculate the inference time
    inference_time = time.time() - start_time

    return predictions, inference_time


col1, col2 = st.columns(2)

with col1:
    st.write("## Choose Model")
    model_result = FURTHRmind(id="model", file_type="keras").download_bytes()
    if model_result is not None:
        model_bytes, model_name = model_result

        with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
            f.write(model_bytes.getvalue())
            f.close()
            model = tf.keras.models.load_model(f.name)
        st.write("load complete")

with col2:
    st.write("## Choose Images")
    images_result = FURTHRmind(id="image", file_type="tiff").download_experiment()
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

        predictions, inference_time = predict(model, preprocessed_images, classification=False)

        st.write(f"Inference time: {inference_time:.4f} seconds")
        st.write("Minimum:", np.min(predictions))
        st.write("Maximum:", np.max(predictions))
