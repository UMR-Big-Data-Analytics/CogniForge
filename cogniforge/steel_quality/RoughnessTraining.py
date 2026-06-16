import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.errors import InvalidArgumentError  # type: ignore
from utils import furthr, ui
from utils.ml import CogniForgeModel, load_images

st.set_page_config(page_title="CogniForge | Rust Training", page_icon="🧱")

st.markdown("""# Roughness Estimation - Training

This page is for the Roughness tool developed by Valerie Durbach.

You can *train* a new image classification model here.""")


tab_data, tab_model, tab_training = st.tabs(["Data", "Model Config", "Training Process"])


with tab_data:
    st.markdown("""## Choose Images without Rust

The datasets listed below are known to contain images of stainless steel.""")
    stainless = ui.furthr_open_collection(
        key="stainless",
        kind=furthr.Sample,
        container_fielddata={
            'Data Label': 'NoRust',
            'Image Width': "ANY",
            'Image Height': "ANY"
        },
        file_extension="tiff"
    )

    if stainless:
        st.success("All necessary training data selected. Proceed to the next tab.")


with tab_model:
    models = CogniForgeModel.open_form(False, stainless)


with tab_training:
    st.markdown("## Run Training")
    is_training_blocked = not (stainless and models)

    if st.button(f"Train {len(models)} models" if models else "Train", disabled=is_training_blocked):
        # for each setting building the corresponding model and evaluate the model
        successes = 0

        for model in models:
            ui.log("Selecting the following training settings:")
            model.render_settings()

            try:
                model.check_settings()
            except ValueError as ex:
                st.warning(str(ex) + ". Skipping...")
                continue

            ui.log("Preparing images and new model for training ...")
            # preprocess the data for training and testing
            _, _, X, Y = load_images(
                False,
                stainless,
                model.architecture,
                model.grayscale,
                model.pretrain,
                model.fft
            )
            # Splitting the data into train,test and validation data, with random_state= 42 to ensure that every model
            # will be trained with the same data for better comparison.
            X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, random_state=42)
            X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=0.5, random_state=42)
            del X, X_test_val, Y, Y_test_val

            try:
                model.build()
                ui.log("Running training process. This can take a long time.")
                history = model.train(X_train, Y_train, X_val, Y_val)
            except (ValueError, InvalidArgumentError) as ex:
                message = str(ex).replace("`", "'") # basic markdown escape
                st.error(f"""Failed to train. Propably the model is not compatible with the selected data and settings.

Original Message: `{message}`""")
                continue

            model_container = model.upload()
            model_container.add_link_to(stainless)
            ui.log("Finished training using previously selected settings.")
            ui.log("Saving model evaluation metrics ...")
            eval_container = model.evaluate(X_test, Y_test, history)
            eval_container.add_link_to(model_container)
            eval_container.add_link_to(stainless)
            ui.log("Stored the model and corresponding metrics in the database.")
            successes += 1

        ui.log(f"Trained {successes} different models. Overall process complete.")
    
    if is_training_blocked:
        st.markdown("Please select the data and configure model settings first.")
