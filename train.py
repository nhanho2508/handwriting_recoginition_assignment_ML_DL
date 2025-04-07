import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen

from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model
from utils import ModelConfigs, save_data_providers
from preprocessing import fetch_and_extract, process_word_data, create_data_providers

import os
import tarfile

dataset_path = os.path.join("Datasets", "IAM_Words")
if not os.path.exists(dataset_path):
    fetch_and_extract("https://git.io/J0fjL", extract_to="Datasets")

    file = tarfile.open(os.path.join(dataset_path, "words.tgz"))
    file.extractall(os.path.join(dataset_path, "words"))

words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
dataset, vocab, max_len = process_word_data(words, dataset_path)

configs = ModelConfigs()
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

train_data_provider, val_data_provider = create_data_providers(dataset, configs)

train_data_provider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

callbacks = [
        EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min"),
        ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER", verbose=1, save_best_only=True, mode="min"),
        TrainLogger(configs.model_path),
        TensorBoard(f"{configs.model_path}/logs", update_freq=1),
        ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="min"),
        Model2onnx(f"{configs.model_path}/model.keras")
    ]

model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=callbacks,
    )

save_data_providers(train_data_provider, val_data_provider, configs)