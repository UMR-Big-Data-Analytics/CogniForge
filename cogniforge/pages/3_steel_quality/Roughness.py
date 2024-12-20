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

st.header("Roughness Placeholder")
st.write(
    "This page is a placeholder for the Roughness tool developed by Valerie Durbach."
)


def load_and_preprocess_data(image_bytes, normalize, GrayScale, pretrain, name):
    image_arrays = []

    # loading the images and extracting Rz values as labels
    with Image.open(image_bytes) as im:
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


def predict(model, X, classification=True):
    # Start inference timing
    start_time = time.time()

    # Make predictions
    predictions = model.predict(X)

    # only for the Classification Task
    predictions = np.asarray(predictions)

    if classification:
        predictions = np.argmax(predictions, axis=1)

    return predictions


classification = (
    st.radio("Task", options=["Rust Classification", "Roughness Prediction"])
    == "Rust Classification"
)


col1, col2 = st.columns(2)

with col1:
    st.write("## Choose Model")
    model_result = FURTHRmind(id="model", file_type="keras").download_bytes()
    if model_result is not None:
        model_bytes, model_name = model_result

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, model_name)
            with open(model_path, "wb") as f:
                f.write(model_bytes.getvalue())
            model = tf.keras.models.load_model(model_path)
        st.write("load complete")

with col2:
    st.write("## Choose Image")
    image_result = FURTHRmind(id="image", file_type="tiff").download_bytes()
    if image_result is not None:
        image_bytes, _ = image_result
        image = Image.open(image_bytes)
        st.image(image, caption="Chosen Image", use_column_width=True)

if image_result is not None and model_result is not None:
    if st.button("Predict"):
        model_name, normalize, grayscale, pretrained = parse_model_name_and_normalize(
            model_name
        )
        images = load_and_preprocess_data(
            image_bytes, normalize, grayscale, pretrained, model_name
        )

        predictions = predict(model, images, classification)
        st.write(predictions.tolist())
