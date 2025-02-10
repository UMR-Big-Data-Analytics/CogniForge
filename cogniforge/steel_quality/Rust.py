import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.furthr import FURTHRmind, download_item_bytes, hash_furthr_item
import config

st.set_page_config(page_title="CogniForge | Rust", page_icon="⛓️")

st.write("""# Rust Prediction

This page is for the Rust tool developed by Valerie Durbach.""")


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
    predictions = np.asarray(_model.predict(_X))

    if _classification:
        # only for the Classification Task
        predictions = np.argmax(predictions, axis=1)

    return predictions


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
                images_width = int(single_field.value)
            if single_field.field_name == 'Image Height':
                images_height = int(single_field.value)
        
        st.write(f"Resolution: {images_width}x{images_height} px")

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
            'Image Grayscaling': "ANY",
            'Data Preprocessing': "ANY",
            'Optimizer': "ANY",
            'Activation Function': "ANY",
            'Loss Function': "ANY"
        }
        model_widget.select_container()

        if model_widget.selected:
            for single_field in model_widget.selected.fielddata:
                if single_field.field_name == 'Model Architecture':
                    model_name = single_field.value
                elif single_field.field_name == 'Image Grayscaling':
                    grayscale = bool(single_field.value)
                elif single_field.field_name == 'Data Preprocessing':
                    pretrained = bool(single_field.value)
                elif single_field.field_name == 'Optimizer':
                    optimizer = single_field.value
                elif single_field.field_name == 'Activation Function':
                    activation = single_field.value
                elif single_field.field_name == 'Loss Function':
                    loss = single_field.value
            
            st.write("### Model Properties")
            st.table({
                'Model Architecture': model_name,
                'Expected Resolution': f"{images_width}x{images_height} px",
                'Image Grayscaling': str(grayscale),
                'Data Preprocessing': str(pretrained),
                'Optimizer': optimizer,
                'Activation Function': activation,
                'Loss Function': loss
            })
    else:
        st.write("Please select the data first. Then compatible models will be shown.")

with tab_prediction:
    is_prediction_blocked = not (images_widget.selected and model_widget.selected)

    if st.button("Predict", disabled=is_prediction_blocked):
        model = load_model(model_widget.selected.files[0])
        images_result, preprocessed_images = load_images(images_widget.selected, model_name, grayscale, pretrained)
        custom_cache_key = (model_widget.selected.id, images_widget.selected.id)
        predictions = predict(model, preprocessed_images, True, custom_cache_key)
        rust_percent, df = format_output(images_widget.selected, images_result, predictions, custom_cache_key)

        st.metric(label="Rust Samples", value=f"{rust_percent:.1f} %")
        st.dataframe(
            df,
            column_config={
                "Link": st.column_config.LinkColumn(display_text="Open image")
            }
        )
    
    if is_prediction_blocked:
        st.write("Please select the data and a model first.")
