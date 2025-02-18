import numpy as np
import streamlit as st
import tempfile
import tensorflow as tf
from PIL import Image
from utils.furthr import download_item_bytes, hash_furthr_item


@st.cache_resource(hash_funcs={"furthrmind.collection.file.File": hash_furthr_item})
def load_model(model_file):
    model_bytes, _ = download_item_bytes(model_file)

    with tempfile.NamedTemporaryFile(delete_on_close=False, suffix=".keras") as fh:
        fh.write(model_bytes.getvalue())
        fh.close()
        model = tf.keras.models.load_model(fh.name)
    return model


# need to do spinner ourself, else inner progress bar gets hidden
@st.cache_data(show_spinner=False, hash_funcs={"furthrmind.collection.sample.Sample": hash_furthr_item})
def load_images(images_container, architecture, grayscale, pretrain):
    with st.spinner("Running `load_images(...)`."):
        images_result = download_item_bytes(images_container)
        image_arrays = []

        for img_file, _ in images_result:
            with Image.open(img_file) as im:
                if grayscale:
                    im_array = np.array(im.convert("RGB"))
                    im = tf.image.rgb_to_grayscale(im_array).numpy()
                image_arrays.append(np.array(im))

        X = image_arrays
        X = np.array([np.array(val) for val in X])

        if pretrain:
            model_class = getattr(tf.keras.applications, architecture)
            preprocess_function = getattr(model_class, 'preprocess_input', None)

            if preprocess_function:
                X = preprocess_function(X)
            
    return images_result, X


@st.cache_data
def predict(_model, _X, _classification, custom_cache_key):
    # Make predictions
    predictions = _model.predict(_X)

    if _classification:
        # only for the Classification Task
        predictions = np.asarray(predictions).argmax(axis=1)

    return predictions