from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import functools
import logging
import os
import time

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
dataset_url = 'dataset.csv'
BATCH_SIZE = 32
EPOCHS = 20
DATASET_SIZE = 8416
OUTPUT_COLUMN = "mushroom"
steps = 300
SELECTED_COLUMNS = ['cap-shape', 'gill-attachment',
                    'gill-spacing', 'ring-type', 'gill-color']


CATEGORIES = {
    'cap-shape': ['CONVEX', 'FLAT', 'BELL', 'SUNKEN', 'KNOBBED', 'CONICAL'],
    'cap-surface': ['SMOOTH', 'FIBROUS', 'SCALY', 'GROOVES'],
    'cap-color': ['WHITE', 'YELLOW', 'BROWN', 'GRAY', 'RED', 'PINK', 'PURPLE', 'GREEN', 'BUFF', 'CINNAMON'],
    'bruises': ['BRUISES', 'NO'],
    'odor': ['ALMOND', 'ANISE', 'NONE', 'PUNGENT', 'CREOSOTE', 'FOUL', 'FISHY', 'SPICY', 'MUSTY'],
    'gill-attachment': ['FREE', 'ATTACHED'],
    'gill-spacing': ['CROWDED', 'CLOSE'],
    'gill-size': ['NARROW', 'BROAD'],
    'gill-color': ['WHITE', 'PINK', 'BROWN', 'GRAY', 'BLACK', 'CHOCOLATE', 'PURPLE', 'GREEN', 'RED', 'BUFF', 'YELLOW', 'ORANGE'],
    'stalk-shape': ['TAPERING', 'ENLARGING'],
    'stalk-root': ['BULBOUS', 'CLUB', 'ROOTED', 'EQUAL', 'UNKNOWN'],
    'stalk-surface-above-ring': ['SMOOTH', 'FIBROUS', 'SILKY', 'SCALY'],
    'stalk-surface-below-ring': ['SMOOTH', 'SCALY', 'FIBROUS', 'SILKY'],
    'stalk-color-above-ring': ['WHITE', 'PINK', 'GRAY', 'BUFF', 'BROWN', 'RED', 'CINNAMON', 'YELLOW', 'ORANGE'],
    'stalk-color-below-ring': ['WHITE', 'PINK', 'GRAY', 'BUFF', 'BROWN', 'RED', 'YELLOW', 'CINNAMON', 'ORANGE'],
    'veil-type': ['PARTIAL'],
    'veil-color': ['WHITE', 'YELLOW', 'ORANGE', 'BROWN'],
    'ring-number': ['ONE', 'TWO', 'NONE'],
    'ring-type': ['PENDANT', 'EVANESCENT', 'LARGE', 'FLARING', 'NONE'],
    'spore-print-color': ['PURPLE', 'BROWN', 'BLACK', 'CHOCOLATE', 'GREEN', 'WHITE', 'YELLOW', 'ORANGE', 'BUFF'],
    'population': ['SEVERAL', 'SCATTERED', 'NUMEROUS', 'SOLITARY', 'ABUNDANT', 'CLUSTERED'],
    'habitat': ['WOODS', 'MEADOWS', 'GRASSES', 'PATHS', 'URBAN', 'LEAVES', 'WASTE']
}

features_names = list(CATEGORIES.keys())


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        ignore_errors=True,
        label_name=OUTPUT_COLUMN,
        **kwargs)
    return dataset


dataset = get_dataset(dataset_url)

# show_batch(dataset)

 
testSize = int(0.30 * DATASET_SIZE)
dataset = dataset.shuffle(500)
dataset = dataset.repeat()
test_dataset = dataset.take(testSize)
train_dataset = dataset.skip(testSize)


categorical_columns = []
for feature, vocab in CATEGORIES.items():

    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))


preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns)
 
model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(train_dataset,
          epochs=EPOCHS,
          verbose=1,  # Suppress chatty output; use Tensorboard instead
          steps_per_epoch=steps,
          validation_data=test_dataset,
          callbacks=[tensorboard_callback])


# Prediction
# 'CONVEX','FREE','CROWDED','PENDANT','WHITE' -> 0


t = time.time()

export_path_keras = "./{}".format(int(t))
print(export_path_keras)

model.save(export_path_keras)