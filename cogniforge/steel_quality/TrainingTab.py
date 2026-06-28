import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import furthr, ml, ui


def _prepare_datasets(
    model: ml.CogniForgeModel,
    datasets: list[furthr.CollectionWrapper[furthr.Sample]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ui.log("Preparing images and new model for training ...")
    # preprocess the data for training and testing
    X, Y = ml.load_multiple_datasets(
        model.classification,
        datasets,
        model.architecture,
        model.grayscale,
        model.pretrain,
        model.fft
    )
    # Splitting the data into train,test and validation data, with random_state= 42 to ensure that every model
    # will be trained with the same data for better comparison.
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=0.5, random_state=42)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def _train_single_model(
    model: ml.CogniForgeModel,
    datasets: list[furthr.CollectionWrapper[furthr.Sample]]
) -> bool:
    ui.log("Selecting the following training settings:")
    model.render_settings()

    try:
        model.check_settings()
    except ValueError as ex:
        st.warning(str(ex) + ". Skipping...")
        return False

    X_train, Y_train, X_test, Y_test, X_val, Y_val = _prepare_datasets(model, datasets)

    try:
        model.build()
        ui.log("Running training process. This can take a long time.")
        history = model.train(X_train, Y_train, X_val, Y_val)
    except (ValueError, tf.errors.InvalidArgumentError) as ex:
        message = str(ex).replace("`", "'") # basic markdown escape
        st.error(f"""Failed to train. Propably the model is not compatible with the selected data and settings.

Original Message:

```
{message}
```""")
        return False

    model_container = model.upload()

    for dataset in datasets:
        model_container.add_link_to(dataset)

    ui.log("Finished training using previously selected settings.")
    ui.log("Saving model evaluation metrics ...")
    eval_container = model.evaluate(X_test, Y_test, history)
    eval_container.add_link_to(model_container)

    for dataset in datasets:
        eval_container.add_link_to(dataset)

    ui.log("Stored the model and corresponding metrics in the database.")
    return True


def _do_training_loop(
    models: ml.ModelGenerator,
    datasets: list[furthr.CollectionWrapper[furthr.Sample]]
):
    # for each setting building the corresponding model and evaluate the model
    info = st.progress(1 / len(models), "Working on model 1")

    successes = 0

    for pos, model in enumerate(models):
        info.progress((pos + 1) / len(models), f"Working on model {pos + 1}")

        if _train_single_model(model, datasets):
            successes +=1

    message = f"Trained **{successes} / {len(models)}** different models. Overall process complete."

    if successes == len(models):
        info.success(message)
    elif successes > 0:
        info.warning(message)
    else:
        info.error(message)


def training_tab_content(
    models: ml.ModelGenerator,
    datasets: list[furthr.CollectionWrapper[furthr.Sample]]
):
    st.markdown("## Run Training")
    is_training_blocked = not (models and all(datasets))

    if st.button(f"Train {len(models)} models" if models else "Train", disabled=is_training_blocked):
        _do_training_loop(models, datasets)
    
    if is_training_blocked:
        st.markdown("Please select the data and configure model settings first.")