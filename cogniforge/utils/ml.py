import itertools
import json
import os
import tempfile
from datetime import datetime
from math import ceil, sqrt

import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from scipy.fftpack import fft2, fftshift
from sklearn.metrics import ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, History, ReduceLROnPlateau
from utils import furthr, ui
from utils.config import FURTHR_MIND, MACHINE_LEARNING

AVAILABLE_CLASSIFICATION_LOSSES = [
    "sparse_categorical_crossentropy",
]
AVAILABLE_REGRESSION_LOSSES = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "cosine_similarity",
    "log_cosh",
    "huber",
]
AVAILABLE_OPTIMIZERS = [
    "Adam",
    "SGD",
    "RMSprop",
    "Adadelta",
    "AdamW",
    "Adagrad",
    "Adamax",
    "Adafactor",
    "Nadam",
    "Ftrl",
    "Lion",
    "LossScaleOptimizer"
]
AVAILABLE_ACTIVATIONS = [
    "relu",
    "leaky_relu",
    "relu6",
    "softmax",
    "celu",
    "elu",
    "selu",
    "softplus",
    "softsign",
    "squareplus",
    "soft_shrink",
    "sparse_plus",
    "silu",
    "gelu",
    "glu",
    "tanh",
    "tanh_shrink",
    "threshold",
    "sigmoid",
    "sparse_sigmoid",
    "exponential",
    "hard_sigmoid",
    "hard_silu",
    "hard_tanh",
    "hard_shrink",
    "linear",
    "mish",
    "log_softmax",
    "log_sigmoid",
    "sparsemax"
]
AVAILABLE_ARCHITECTURES = [
    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtXLarge",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "EfficientNetV2B0",
    "EfficientNetV2B1",
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2S",
    "EfficientNetV2M",
    "EfficientNetV2L",
    "InceptionResNetV2",
    "InceptionV3",
    "MobileNet",
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "NASNetLarge",
    "NASNetMobile",
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",
    "ResNet50",
    "ResNet50V2",
    "VGG16",
    "VGG19",
    "Xception"
]
AVAILABLE_POOLING = [
    None,
    "avg",
    "max"
]
MODEL_GROUP = furthr.CollectionWrapper(furthr.get_furthr_client()[0].Group.get(FURTHR_MIND.get('ModelGroupId')))


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


# need to do spinner ourself, else inner progress bar gets hidden
@st.cache_data(show_spinner=False, max_entries=2)
def load_images(
    classification: bool,
    images_container: furthr.CollectionWrapper,
    architecture: str,
    grayscale: bool,
    pretrain: bool,
    fft: bool
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with st.spinner("Running `load_images(...)`."):
        images_raw = images_container.download_files()
        X_shape = (
            len(images_raw),
            images_container.image_width,
            images_container.image_height,
            1 if grayscale else 3
        )
        X = np.empty(X_shape, dtype=np.uint8)

        if not classification:
            allowed = MACHINE_LEARNING.getrange('Roughness')
            Y = np.empty(len(images_raw), dtype=np.float32)
        elif images_container.data_label == "Rust":
            Y = np.ones(len(images_raw), dtype=np.uint8)
        else:
            Y = np.zeros(len(images_raw), dtype=np.uint8)

        ids = []
        names = []

        for img_bytes, img_name, img_id in images_raw:
            with Image.open(img_bytes) as im:
                if not classification:
                    # Tag number 270 is the ImageDescription tag
                    tags = json.loads(im.tag_v2[270])
                    Rz = tags['roughness']['Rz']

                    if Rz < allowed.start or Rz > allowed.stop:
                        continue
                    
                    Y[len(ids)] = Rz

                if grayscale:
                    im = im.convert("L")
                    np_im = np.array(im)
                    np_im = np.expand_dims(np_im, axis=-1)
                else:
                    np_im = np.array(im)

                if fft:
                    np_im = apply_fft(np_im)

                X[len(ids)] = np_im
                names.append(img_name)
                ids.append(img_id)

        if pretrain:
            model_class = getattr(tf.keras.applications, architecture)
            preprocess_function = getattr(model_class, 'preprocess_input', None)

            if preprocess_function:
                X = preprocess_function(X)

    return ids, names, X[:len(ids)], Y[:len(ids)]


def load_multiple_datasets(
    classification: bool,
    datasets: list[furthr.CollectionWrapper[furthr.Sample]],
    architecture: str,
    grayscale: bool,
    pretrain: bool,
    fft: bool
):
    if len(datasets) == 1:
        _, _, X, Y = load_images(classification, datasets[0], architecture, grayscale, pretrain, fft)
        return X, Y
    
    X_list = []
    Y_list = []

    for dataset in datasets:
        _, _, X, Y = load_images(classification, dataset, architecture, grayscale, pretrain, fft)
        X_list.append(X)
        Y_list.append(Y)

    return np.concatenate(X_list), np.concatenate(Y_list)


class PredictionProgressBar(Callback):
    def __init__(self, X: np.ndarray):
        self.batches = ceil(len(X) / MACHINE_LEARNING.getint('BatchSize'))
        self.bar = st.progress(0, "Running prediction ...")

    def on_predict_batch_end(self, batch, logs):
        percent_complete = (batch + 1) / self.batches
        self.bar.progress(percent_complete, "Running prediction ...")

    def on_predict_end(self, logs):
        self.bar.empty()


class TrainingProgressBar(Callback):
    def __init__(self, X_train: np.ndarray, X_val: np.ndarray):
        train_batches = ceil(len(X_train) / MACHINE_LEARNING.getint('BatchSize'))
        val_batches = ceil(len(X_val) / MACHINE_LEARNING.getint('BatchSize'))
        self.max_steps = MACHINE_LEARNING.getint('Epochs') * (train_batches + val_batches)
        self.current_step = 0
        self.bar = st.progress(0, f"Epoch 1 / {MACHINE_LEARNING.getint('Epochs')}")

    def __render(self):
        percent_complete = self.current_step / self.max_steps
        self.bar.progress(percent_complete, f"Epoch {self.current_epoch} / {MACHINE_LEARNING.getint('Epochs')}")
    
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


class ModelGenerator:
    def __init__(self, classification: bool, images: furthr.CollectionWrapper, settings: dict):
        self.classification = classification
        self.width = images.image_width
        self.height = images.image_height
        self.setting_values = settings.values()

    def __iter__(self):
        for o in itertools.product(*self.setting_values):
            yield CogniForgeModel(
                classification=self.classification,
                architecture=o[0],
                width=self.width,
                height=self.height,
                grayscale=o[1],
                pretrain=o[2],
                fft=False if self.classification else o[7],
                optimizer=o[3],
                activation=o[4],
                loss=o[5],
                pool=o[6]
            )
    
    def __len__(self):
        try:
            return self.__count
        except AttributeError:
            self.__count = 1

            for selected_options in self.setting_values:
                self.__count *= len(selected_options)
            
            return self.__count


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
        self.container = None
        self.model = None
    
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
            force_group_id=FURTHR_MIND.get('ModelGroupId'),
            file_extension="keras"
        )
        return CogniForgeModel.from_container(container) if container else None
    
    @staticmethod
    def open_form(
        classification: bool,
        images: furthr.CollectionWrapper[furthr.Sample] | None
    ) -> ModelGenerator | None:
        st.markdown("## Choose Model Settings")
        inputs = {
            'Model Architecture': AVAILABLE_ARCHITECTURES,
            'Image Grayscaling': [True, False],
            'Pretrained Weights': [True, False],
            'Optimizer': AVAILABLE_OPTIMIZERS,
            'Activation Function': AVAILABLE_ACTIVATIONS,
            'Loss Function': AVAILABLE_CLASSIFICATION_LOSSES if classification else AVAILABLE_REGRESSION_LOSSES,
            'Pooling Mode': AVAILABLE_POOLING
        }

        if not classification:
            inputs['FFT Images'] = [True, False]

        settings = ui.form("settings", inputs)

        if settings and images:
            return ModelGenerator(classification, images, settings)
        else:
            return None

    def render_settings(self):
        settings = {
            'Model Architecture': self.architecture,
            'Expected Resolution': f"{self.width}x{self.height} px",
            'Image Grayscaling': str(self.grayscale),
            'Pretrained Weights': str(self.pretrain),
            'Optimizer': self.optimizer,
            'Activation Function': self.activation,
            'Loss Function': self.loss
        }

        if not self.classification:
            settings['FFT Images'] = str(self.fft)

        st.markdown("### Model Properties")
        st.table(settings)

    def check_settings(self):
        if self.grayscale and self.pretrain:
            raise ValueError("Grayscaling cannot be used with pretrained models")
        
        if self.fft and not self.grayscale:
            raise ValueError("FFT can only be ap plied when grayscaling")

    @st.cache_resource(ttl=FURTHR_MIND.get('FileTTL'), show_spinner="Running `CogniForgeModel.download(...)`.")
    @staticmethod
    def __cached_download(container: furthr.CollectionWrapper[furthr.ResearchItem]) -> tf.keras.Model:
        # prevent OOM
        tf.keras.utils.clear_session()
        data = container.download_files()[0][0]

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
            batch_size=MACHINE_LEARNING.getint('BatchSize'),
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
        # prevent OOM
        tf.keras.utils.clear_session()
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
            input_shape=(self.width, self.height, 1 if self.grayscale else 3),
            pooling=self.pool,
            classes=2 if self.classification else 1,
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
            epochs=MACHINE_LEARNING.getint('Epochs'),
            batch_size=MACHINE_LEARNING.getint('BatchSize'),
            validation_data=(X_val, Y_val),
            callbacks=[early_stopping, reduce_lr, progress],
            class_weight=class_weight_dict,
            verbose=2
        )
    
    def upload(self) -> furthr.CollectionWrapper[furthr.ResearchItem]:
        label = f"{self.architecture} @ {datetime.now():%d.%m.%y %H:%M:%S}"
        self.container = MODEL_GROUP.sub_collection(label, furthr.ResearchItem, "Code").create()
        self.container.upload_content(self.model, f"{self.architecture}.keras")
        self.container.model_architecture = self.architecture
        self.container.image_width = self.width
        self.container.image_height = self.height
        self.container.image_grayscaling = self.grayscale
        self.container.pretrained_weights = self.pretrain
        self.container.optimizer = self.optimizer
        self.container.activation_function = self.activation
        self.container.loss_function = self.loss

        if self.classification:
            self.container.model_purpose = "Rust Detection"
        else:
            self.container.model_purpose = "Roughness Estimation"
            self.container.fft_images = self.fft

        return self.container
    
    def evaluate(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        history: History
    ) -> furthr.CollectionWrapper[furthr.ResearchItem]:
        # predict classes/values
        predictions = self.predict(X_test)
        # creating folder for the model
        label = self.container.raw.name
        eval_container = MODEL_GROUP.sub_collection(label, furthr.ResearchItem, "Analysis").create()

        if self.classification:
            metrics = self.__evaluate_classification(eval_container, predictions, Y_test, history)
        else:
            metrics = self.__evaluate_regression(eval_container, predictions, Y_test)
        
        eval_container.upload_content(metrics, "metrics.json")
        return eval_container
    
    def __evaluate_classification(
        self,
        eval_container: furthr.CollectionWrapper,
        predictions: np.ndarray,
        Y_test: np.ndarray,
        history: History
    ) -> dict[str, float]:
        # plot confusion matrix and saving .png
        disp = ConfusionMatrixDisplay.from_predictions(Y_test, predictions, labels=[0, 1])
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
        # calculate evaluation metrics
        TN, FP, FN, TP = disp.confusion_matrix.ravel()
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        return {
            'accuracy': (TP + TN) / (TP + TN + FP + FN),
            'precision': precision,
            'recall': recall,
            'specificity': TN / (TN + FP) if (TN + FP) != 0 else 0,
            'sensitivity': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0,
            "confusion_matrix" : {
                'TN': int(TN),
                'FP': int(FP),
                'FN': int(FN),
                'TP': int(TP)
            }
        }
    
    def __evaluate_regression(
        self,
        eval_container: furthr.CollectionWrapper,
        predictions: np.ndarray,
        Y_test: np.ndarray
    ) -> dict[str, float]:
        # plot true vs predicted values and saving .png
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, predictions, alpha=0.5)
        plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("True vs Predicted Values")
        eval_container.upload_content(plt.gcf(), "TrueVSPred_plot.png")
        plt.close()
        # Plot difference between predicted and actual values and saving .png
        differences = predictions - Y_test
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=50, alpha=0.7)
        plt.xlabel("Difference (Predicted - Actual)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Differences between Predicted and Actual Values")
        eval_container.upload_content(plt.gcf(), "freq_plot.png")
        plt.close()
        # calculate evaluation metrics
        mse = mean_squared_error(Y_test, predictions)
        metrics = {
            'mean_absolute_error': float(mean_absolute_error(Y_test, predictions)),
            'mean_squared_error': float(mse),
            'root_mean_squared_error': sqrt(mse),
            'r_squared': float(r2_score(Y_test, predictions)),
            'bias': float(np.mean(predictions - Y_test)),
        }
        lower = 25
        breakpoints = [50, 75, 100, 125, float('inf')]

        for upper in breakpoints:
            mask = (Y_test >= lower) & (Y_test < upper)
            key = f"mae[{lower}-{upper}]"
            metrics[key] = float(mean_absolute_error(Y_test[mask], predictions[mask])) if np.sum(mask) > 0 else None
            lower = upper

        return metrics