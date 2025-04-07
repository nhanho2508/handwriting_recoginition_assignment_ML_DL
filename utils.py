import cv2
import numpy as np
import typing

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
import os
from datetime import datetime
from mltu.configs import BaseModelConfigs
import typing
import tensorflow as tf
import tarfile
from tqdm import tqdm
from tensorflow import keras
from keras import layers
from urllib.request import urlopen
from io import BytesIO
from keras import layers
from keras.models import Model
from zipfile import ZipFile

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()


        self.model_path = os.path.join(
            "Models/handwriting_recognition",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )


        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0


        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20


class ImageToText(OnnxInferenceModel):
    def __init__(self, characters: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.characters = characters

    def predict(self, input_image: np.ndarray) -> str:
        resized_image = cv2.resize(input_image, tuple(self.input_shapes[0][2:0:-1]))
        image_for_prediction = np.expand_dims(resized_image, axis=0).astype(np.float32)
        predictions = self.model.run(self.output_names, {self.input_names[0]: image_for_prediction})[0]
        decoded_text = ctc_decoder(predictions, self.characters)[0]
        return decoded_text
    
def apply_activation(layer, activation_type: str = "relu", alpha_value: float = 0.1) -> tf.Tensor:
    if activation_type == "relu":
        return layers.ReLU()(layer)
    elif activation_type == "leaky_relu":
        return layers.LeakyReLU(alpha=alpha_value)(layer)
    return layer

def build_residual_block(
    input_tensor: tf.Tensor,
    num_filters: int,
    stride_value: typing.Union[int, list] = 2,
    kernel_dim: typing.Union[int, list] = 3,
    use_skip_conv: bool = True,
    padding_type: str = "same",
    kernel_init: str = "he_uniform",
    activation_type: str = "relu",
    dropout_rate: float = 0.2
) -> tf.Tensor:
    skip_connection = input_tensor

    x = layers.Conv2D(
        num_filters, kernel_dim, strides=stride_value, padding=padding_type, kernel_initializer=kernel_init
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = apply_activation(x, activation_type=activation_type)

    x = layers.Conv2D(
        num_filters, kernel_dim, padding=padding_type, kernel_initializer=kernel_init
    )(x)
    x = layers.BatchNormalization()(x)

    if use_skip_conv:
        skip_connection = layers.Conv2D(
            num_filters, 1, strides=stride_value, padding=padding_type, kernel_initializer=kernel_init
        )(skip_connection)

    x = layers.Add()([x, skip_connection])
    x = apply_activation(x, activation_type=activation_type)

    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x

def fetch_and_extract(url, target_dir="Datasets", buffer_size=1024*1024):
    response = urlopen(url)

    content = b""
    total_chunks = response.length // buffer_size + 1
    for _ in tqdm(range(total_chunks)):
        content += response.read(buffer_size)

    with ZipFile(BytesIO(content)) as zip_file:
        zip_file.extractall(path=target_dir)

def save_data_providers(train_data_provider, val_data_provider, configs):
    train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
