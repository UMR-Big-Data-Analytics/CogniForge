import itertools
import os
import tempfile
from collections.abc import Generator
from datetime import date
from io import BytesIO
from math import ceil

import config
import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from scipy.fftpack import fft2, fftshift
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, History, ReduceLROnPlateau  # type: ignore
from utils import furthr, ui

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
MODEL_GROUP = furthr.CollectionWrapper(furthr.get_furthr_client()[0].Group.get(config.furthr['model_group_id']))


# need to do spinner ourself, else inner progress bar gets hidden
@st.cache_data(show_spinner=False, max_entries=2)
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


class CogniForgeModel:
    classification: bool
    architecture: str
    width: int
    height: int
    grayscale: bool
    pretrain: bool
    fft: bool
    optimizer: str
    activation: str
    loss: str
    pool: str | None
    container: furthr.CollectionWrapper[furthr.ResearchItem] | None
    model: tf.keras.Model | None

    def __init__(
        self,
        classification: bool,
        architecture: str,
        width: int,
        height: int,
        grayscale: bool,
        pretrain: bool,
        fft: bool,
        optimizer: str,
        activation: str,
        loss: str,
        pool: str | None
    ):
        self.classification = classification
        self.architecture = architecture
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.pretrain = pretrain
        self.fft = fft
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.pool = pool
    
    @staticmethod
    def from_container(container: furthr.CollectionWrapper[furthr.ResearchItem]) -> 'CogniForgeModel':
        wrapper = CogniForgeModel(
            container.model_purpose == "Rust Detection",
            container.model_architecture,
            container.image_width,
            container.image_height,
            container.image_grayscaling,
            container.pretrained_weights,
            getattr(container, 'fft_images', False),
            container.optimizer,
            container.activation_function,
            container.loss_function,
            None
        )
        wrapper.container = container
        return wrapper
    
    @staticmethod
    def open_dropdown(
        classification: bool,
        images: furthr.CollectionWrapper[furthr.Sample]
    ) -> 'CogniForgeModel | None':
        expected = {
            'Image Width': images.image_width,
            'Image Height': images.image_height,
            'Model Architecture': "ANY",
            'Image Grayscaling': "ANY",
            'Pretrained Weights': "ANY",
            'Optimizer': "ANY",
            'Activation Function': "ANY",
            'Loss Function': "ANY"
        }

        if classification:
            expected['Model Purpose'] = "Rust Detection"
        else:
            expected['Model Purpose'] = "Roughness Estimation"
            expected['FFT Images'] = "ANY"

        container = ui.furthr_open_collection(
            key="model",
            kind=furthr.ResearchItem,
            category="Code",
            container_fielddata=expected,
            force_group_id=config.furthr['model_group_id'],
            file_extension="keras"
        )
        return CogniForgeModel.from_container(container) if container else None
    
    @staticmethod
    def open_form(
        classification: bool,
        images: furthr.CollectionWrapper[furthr.Sample] | None
    ) -> list['CogniForgeModel'] | None:
        st.markdown("## Choose Model Settings")
        inputs = {
            'Model Architecture': AVAILABLE_ARCHITECTURES,
            'Image Grayscaling': [True, False],
            'Pretrained Weights': [True, False],
            'Optimizer': AVAILABLE_OPTIMIZERS,
            'Activation Function': AVAILABLE_ACTIVATIONS,
            'Loss Function': AVAILABLE_LOSSES,
            'Pooling Mode': AVAILABLE_POOLING
        }

        if not classification:
            inputs['FFT Images'] = [True, False]

        settings = ui.form("settings", inputs)

        if not settings or not images:
            return None
        
        return [
            CogniForgeModel(
                classification=classification,
                architecture=o[0],
                width=images.image_width,
                height=images.image_height,
                grayscale=o[1],
                pretrain=o[2],
                fft=False if classification else o[7],
                optimizer=o[3],
                activation=o[4],
                loss=o[5],
                pool=o[6]
            )
            for o in itertools.product(*settings.values())
        ]

    def render_settings(self):
        st.markdown("### Model Properties")
        st.table({
            'Model Architecture': self.architecture,
            'Expected Resolution': f"{self.width}x{self.height} px",
            'Image Grayscaling': str(self.grayscale),
            'Pretrained Weights': str(self.pretrain),
            'Optimizer': self.optimizer,
            'Activation Function': self.activation,
            'Loss Function': self.loss
        })

    @st.cache_resource(ttl=config.furthr['file_ttl'], show_spinner="Running `CogniForgeModel.download(...)`.")
    @staticmethod
    def __cached_download(container: furthr.CollectionWrapper[furthr.ResearchItem]) -> tf.keras.Model:
        data, _ = container.download_files()[0]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as fh:
            fh.write(data.getvalue())
            fh.close()
            model = tf.keras.models.load_model(fh.name)
            os.remove(fh.name)
        
        return model
    
    def download(self):
        self.model = CogniForgeModel.__cached_download(self.container)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        progress = PredictionProgressBar(X)
        predictions = self.model.predict(
            X,
            batch_size=config.ml['batch_size'],
            callbacks=[progress],
            verbose=2
        )

        if self.classification:
            # only for the Classification Task
            return np.asarray(predictions).argmax(axis=1)

        if predictions.ndim > 1:
            # needed for ResNet50 and maybe others
            return predictions.ravel()

        return predictions
    
    def build(self):
        # getting the model name and loss function dynamically
        model_class = getattr(tf.keras.applications, self.architecture)
        loss_function = tf.keras.losses.get(self.loss)

        if self.pretrain:
            weights = "imagenet"
            top = False
        else:
            weights = None
            top = True

        # building model
        self.model = model_class(
            include_top=top,
            weights=weights,
            input_tensor=None,
            input_shape=(self.width, self.height) if self.grayscale else (self.width, self.height, 3),
            pooling=self.pool,
            classes=2,
            classifier_activation=self.activation
        )

        # adding the appropiate layers for transfer learning
        if self.pretrain:
            x = tf.keras.layers.GlobalAveragePooling2D()(self.model.output)
            output = tf.keras.layers.Dense(2, activation="softmax")(x)
            self.model = tf.keras.Model(inputs=self.model.input, outputs=output)

        # compiling model
        self.model.compile(optimizer=self.optimizer, metrics=["acc"], loss=loss_function)
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
    ) -> History:
        # Calculate class weights usefull if the classes are not balanced
        class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Callbacks, early stopping so the model does not overfits and reduce_lr for better accuracy
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
        progress = TrainingProgressBar(X_train, X_val)

        # Training the model
        return self.model.fit(
            X_train,
            Y_train,
            epochs=config.ml['epochs'],
            batch_size=config.ml['batch_size'],
            validation_data=(X_val, Y_val),
            callbacks=[early_stopping, reduce_lr, progress],
            class_weight=class_weight_dict,
            verbose=2
        )
    
    def upload(self, number: int) -> furthr.CollectionWrapper[furthr.ResearchItem]:
        date_str = date.today().strftime('%Y-%m-%d')
        long_name = f"Model {number}_{self.architecture}_{date_str}"
        model_container = MODEL_GROUP.sub_collection(long_name, furthr.ResearchItem, "Code").create()
        model_container.upload_content(self.model, f"{self.architecture}.keras")
        model_container.model_architecture = self.architecture
        model_container.image_width = self.width
        model_container.image_height = self.height
        model_container.image_grayscaling = self.grayscale
        model_container.pretrained_weights = self.pretrain
        model_container.optimizer = self.optimizer
        model_container.activation_function = self.activation
        model_container.loss_function = self.loss
        return model_container
    
    def evaluate(
        self,
        number: int,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        history: History
    ) -> furthr.CollectionWrapper[furthr.ResearchItem]:
        predictions = self.predict(X_test)
        # creating folder for the model
        date_str = date.today().strftime('%Y-%m-%d')
        long_name = f"Evaluation {number}_{self.architecture}_{date_str}"
        eval_container = MODEL_GROUP.sub_collection(long_name, furthr.ResearchItem, "Analysis").create()
        # plot confusion matrix and saving .png
        disp = ConfusionMatrixDisplay.from_predictions(Y_test, predictions)
        eval_container.upload_content(disp.figure_, "cm_plot.png")
        # plot model accuracy and saving .png
        plt.figure()
        plt.plot(history.history['acc'], marker='o')
        plt.plot(history.history['val_acc'], marker='o')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='lower right')
        eval_container.upload_content(plt.gcf(), "accuracy_plot.png")
        plt.close()
        # plot model loss and saving .png
        plt.figure()
        plt.plot(history.history['loss'], marker='o')
        plt.plot(history.history['val_loss'], marker='o')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        eval_container.upload_content(plt.gcf(), "loss_plot.png")
        plt.close()
        return eval_container