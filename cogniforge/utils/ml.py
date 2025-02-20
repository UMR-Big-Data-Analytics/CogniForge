import numpy as np
from scipy.fftpack import fft2, fftshift
import streamlit as st
import tempfile
import tensorflow as tf
from PIL import Image
from utils.furthr import download_item_bytes, hash_furthr_item


@st.cache_resource(hash_funcs={"furthrmind.collection.file.File": hash_furthr_item})
def load_model(model_file):
    model_bytes, _ = download_item_bytes(model_file)

    with tempfile.NamedTemporaryFile(suffix=".keras") as fh:
        fh.write(model_bytes.getvalue())
        fh.flush()
        model = tf.keras.models.load_model(fh.name)
    return model


# need to do spinner ourself, else inner progress bar gets hidden
@st.cache_data(show_spinner=False, hash_funcs={"furthrmind.collection.sample.Sample": hash_furthr_item})
def load_images(images_container, architecture, grayscale, pretrain, fft=True):
    grayscale = False
    pretrain = True
    fft=False
    architecture="EfficientNetB0"
    with st.spinner("Running `load_images(...)`."):
        images_result = download_item_bytes(images_container)
        image_arrays = []

        for img_file, _ in images_result:
            with Image.open(img_file) as im:
                if grayscale or fft:
                    im = im.convert("L")

                if fft:
                    fft_image = apply_fft(np.array(im))
                    print(fft_image.shape)
                    image_arrays.append(fft_image)
                else:
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
    elif predictions.ndim > 1:
        # needed for ResNet50 and maybe others
        predictions = predictions.ravel()

    return predictions


def apply_fft(image):
    # Remove the channel dimension temporarily if it exists
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]  # Remove the last dimension (channel)

    # Perform 2D FFT and shift the zero frequency component to the center
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)

    # Calculate the magnitude spectrum and take the logarithm
    magnitude_spectrum = np.log(np.abs(fft_image_shifted) + 1)

    # Normalize to 0-255
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (
        magnitude_spectrum.max() - magnitude_spectrum.min()
    )
    magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)

    # Add back the single channel dimension to keep shape consistent
    magnitude_spectrum = magnitude_spectrum[:, :, np.newaxis]

    return magnitude_spectrum