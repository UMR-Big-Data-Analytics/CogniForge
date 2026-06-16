import numpy as np
import pandas as pd
import streamlit as st

from utils.furthr import FURTHRmind
from utils.ml import load_model, load_images, predict
import config

st.set_page_config(page_title="CogniForge | Roughness", page_icon="🗻")

st.write("""# Roughness Estimation

This page is for the Roughness tool developed by Valerie Durbach.""")


@st.cache_data
def format_output(_images_container, _images_result, _predictions, custom_cache_key):
    min_roughness = np.min(predictions)
    max_roughness = np.max(predictions)
    avg_roughness = np.average(predictions)

    df = pd.DataFrame({
        "Filename": [o[1] for o in _images_result],
        "Roughness": _predictions,
        "Link": ["/Photo?file_id=" + file.id for file in _images_container.files]
    })
    df = df.sort_values(by="Roughness", ascending=False)
    return min_roughness, max_roughness, avg_roughness, df


tab_data, tab_training, tab_model, tab_prediction = st.tabs(["Data", "Model Training", "Model Selection", "Prediction Analysis"])

with tab_data:
    st.write("## Choose Images")
    st.write("They must have no rust. Samples with rust get hidden below.")
    images_widget = FURTHRmind(id="image")
    images_widget.file_extension = "tiff"
    images_widget.container_category = "sample"
    images_widget.expected_fielddata = {
        'Image Width': "ANY",
        'Image Height': "ANY",
        'Data Label': "NoRust"
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
        st.write("Only roughness estimation models compatible with the resolution of the selected data get shown below.")
        model_widget.file_extension = "keras"
        model_widget.force_group_id = config.furthr['model_group_id']
        model_widget.container_category = "Code"
        model_widget.expected_fielddata = {
            'Model Purpose': "Roughness Estimation",
            'Image Width': images_width,
            'Image Height': images_height,
            'Model Architecture': "ANY",
            'Image Grayscaling': "ANY",
            'FFT Images': 'ANY',
            'Pretrained Weights': "ANY",
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
                elif single_field.field_name == 'FFT Images':
                    fft = bool(single_field.value)
                elif single_field.field_name == 'Pretrained Weights':
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
                'FFT Images': str(fft),
                'Pretrained Weights': str(pretrained),
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
        images_result, preprocessed_images = load_images(images_widget.selected, model_name, grayscale, pretrained, fft)
        custom_cache_key = (model_widget.selected.id, images_widget.selected.id)
        predictions = predict(model, preprocessed_images, False, custom_cache_key)
        min_roughness, max_roughness, avg_roughness, df = format_output(images_widget.selected, images_result, predictions, custom_cache_key)

        col1, col2 = st.columns(2)
        col1.metric(label="Minimum Roughness", value=f"{min_roughness:.2f} μm")
        col2.metric(label="Maximum Roughness", value=f"{max_roughness:.2f} μm")
        st.metric(label="Average Roughness", value=f"{avg_roughness:.2f} μm")
        st.dataframe(
            df,
            column_config={
                "Link": st.column_config.LinkColumn(display_text="Open image")
            }
        )
    
    if is_prediction_blocked:
        st.write("Please select the data and a model first.")
