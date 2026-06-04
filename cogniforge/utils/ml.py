import os
import tempfile
from io import BytesIO
from math import ceil

import config
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from scipy.fftpack import fft2, fftshift
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, History, ReduceLROnPlateau
from utils import furthr

AVAILABLE_LOSSES = [
    "binary_crossentropy",
    "binary_focal_crossentropy",
    "categorical_crossentropy",
    "categorical_focal_crossentropy",
    "sparse_categorical_crossentropy",
    "poisson",
    "ctc",
    "kl_divergence",
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "cosine_similarity",
    "huber_loss",
    "log_cosh",
    "tversky",
    "dice",
    "hinge",
    "squared_hinge",
    "categorical_hinge",
    "categorical_generalized_cross_entropy",
    "circle"
]
AVAILABLE_OPTIMIZERS = [
    "SGD",
    "rmsprop",
    "adam",
    "adamw",
    "adadelta",
    "adagrad",
    "adamax",
    "adafactor",
    "nadam",
    "ftrl",
    "lion",
    "lamb",
    "muon"
]
AVAILABLE_ACTIVATIONS = [
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    "exponential"
]
AVAILABLE_ARCHITECTURES = [
    "EfficientNetV2B0",
    "EfficientNetB0",
    "ConvNeXtTiny",
    "DenseNet201",
    "DenseNet169",
    "ResNet50",
    "ResNet50V2",
    "VGG16",
    "VGG19"
]
AVAILABLE_POOLING = [
    None,
    "avg",
    "max"
]


@st.cache_resource
def load_model(model_container: furthr.CollectionWrapper) -> tf.keras.Model:
    model_bytes, _ = model_container.download_files()[0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as fh:
        fh.write(model_bytes.getvalue())
        fh.close()
        model = tf.keras.models.load_model(fh.name)
        os.remove(fh.name)
    return model


# need to do spinner ourself, else inner progress bar gets hidden
@st.cache_data(show_spinner=False)
def load_images(
    images_container: furthr.CollectionWrapper,
    architecture: str,
    grayscale: bool,
    pretrain: bool,
    fft: bool
) -> tuple[list[tuple[BytesIO, str]], np.ndarray]:
    with st.spinner("Running `load_images(...)`."):
        images_result = images_container.download_files()
        image_arrays = []

        for img_file, _ in images_result:
            with Image.open(img_file) as im:
                if grayscale or fft:
                    im = im.convert("L")

                np_im = np.array(im)

                if fft:
                    np_im = apply_fft(np_im)

                image_arrays.append(np_im)

        X = image_arrays
        X = np.array([np.array(val) for val in X])

        if pretrain:
            model_class = getattr(tf.keras.applications, architecture)
            preprocess_function = getattr(model_class, 'preprocess_input', None)

            if preprocess_function:
                X = preprocess_function(X)
            
    return images_result, X


class PredictionProgressBar(Callback):
    def __init__(self, X: np.ndarray):
        self.batches = ceil(len(X) / config.ml['batch_size'])
        self.bar = st.progress(0, "Running prediction ...")

    def on_predict_batch_end(self, batch, logs):
        percent_complete = (batch + 1) / self.batches
        self.bar.progress(percent_complete, "Running prediction ...")

    def on_predict_end(self, logs):
        self.bar.empty()


def predict(model: tf.keras.Model, X: np.ndarray, classification: bool) -> np.ndarray:
    # Make predictions
    progress = PredictionProgressBar(X)
    predictions = model.predict(
        X,
        batch_size=config.ml['batch_size'],
        callbacks=[progress],
        verbose=2
    )

    if classification:
        # only for the Classification Task
        return np.asarray(predictions).argmax(axis=1)
    
    if predictions.ndim > 1:
        # needed for ResNet50 and maybe others
        return predictions.ravel()

    return predictions


def apply_fft(image: np.ndarray) -> np.ndarray:
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


def build_model(
    architecture: str,
    input_size: tuple,
    activation: str,
    optimizer: str,
    loss: str,
    pretrain: bool,
    pool: str | None
) -> tf.keras.Model:
    # getting the model name and loss function dynamically
    model_class = getattr(tf.keras.applications, architecture)
    loss_function = getattr(tf.keras.losses, loss)

    if pretrain:
        weights = "imagenet"
        top = False
    else:
        weights = None
        top = True

    # building model
    model = model_class(
        include_top=top,
        weights=weights,
        input_tensor=None,
        input_shape=input_size,
        pooling=pool,
        classes=2,
        classifier_activation=activation
    )

    # adding the appropiate layers for transfer learning
    if pretrain:
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        output = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs=model.input, outputs=output)

    # compiling model
    model.compile(optimizer=optimizer, metrics=["acc"], loss=loss_function)
    return model


class TrainingProgressBar(Callback):
    def __init__(self, X_train: np.ndarray, X_val: np.ndarray):
        train_batches = ceil(len(X_train) / config.ml['batch_size'])
        val_batches = ceil(len(X_val) / config.ml['batch_size'])
        self.max_steps = config.ml['epochs'] * (train_batches + val_batches)
        self.current_step = 0
        self.bar = st.progress(0, f"Epoch 1 / {config.ml['epochs']}")

    def __render(self):
        percent_complete = self.current_step / self.max_steps
        self.bar.progress(percent_complete, f"Epoch {self.current_epoch} / {config.ml['epochs']}")
    
    def __increment(self):
        self.current_step += 1
        self.__render()

    def on_epoch_begin(self, epoch, logs):
        # count from 1 to be more user friendly
        self.current_epoch = epoch + 1
        self.__render()

    def on_train_batch_end(self, batch, logs):
        self.__increment()

    def on_test_batch_end(self, batch, logs):
        self.__increment()

    def on_train_end(self, logs):
        self.bar.empty()


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    model: tf.keras.Model
) -> History:
    # Calculate class weights usefull if the classes are not balanced
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    #Callbacks, early stopping so the model does not overfits and reduce_lr for better accuracy
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
    progress = TrainingProgressBar(X_train, X_val)

    #training the model
    return model.fit(
        X_train,
        Y_train,
        epochs=config.ml['epochs'],
        batch_size=config.ml['batch_size'],
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping, reduce_lr, progress],
        class_weight=class_weight_dict,
        verbose=2
    )