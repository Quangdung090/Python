import tensorflow as tf
import stow

import os

from io import BytesIO
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage,Image

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

import os
from tqdm import tqdm

from config import ModelConfigs
from model import train_model


print(tf.__version__)

sentences_txt_path = stow.join('/content','metadata','sentences.txt')
sentences_folder_path = stow.join('/content', 'IAM_Sentence')

dataset, vocab, max_len = [], set(), 0

words = open(sentences_txt_path, "r").readlines()
for line in tqdm(words):
  if line.startswith("#"):
    continue

  line_split = line.split(" ")
  if line_split[2] == "err":
    continue
  file_name = line_split[0] + ".png"
  label = line_split[-1].rstrip('\n')

  label = label.replace('|', ' ')

  rel_path = stow.join(sentences_folder_path, file_name)
  if not os.path.isfile(rel_path):
    continue
  dataset.append([rel_path, label])
  vocab.update(list(label))
  max_len = max(max_len, len(label))

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()
# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)
# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
]

from keras.models import Sequential, load_model
# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
)
model.summary(line_length=110)
# Define callbacks
earlystopper = EarlyStopping(monitor='val_CER', patience=20, verbose=1, mode='min')
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode='auto')
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)