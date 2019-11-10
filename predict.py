from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import functools
import logging
import os
import time

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

export_path_keras = '1573341301'

reloaded = tf.keras.models.load_model(export_path_keras)

x = {
    'cap-shape': [],
    'cap-surface': [],
    'cap-color': [],
    'bruises': [],
    'odor': [],
    'gill-attachment': [],
    'gill-spacing': [],
    'gill-size': [],
    'gill-color': [],
    'stalk-shape': [],
    'stalk-root': [],
    'stalk-surface-above-ring': [],
    'stalk-surface-below-ring': [],
    'stalk-color-above-ring': [],
    'stalk-color-below-ring': [],
    'veil-type': [],
    'veil-color': [],
    'ring-number': [],
    'ring-type': [],
    'spore-print-color': [],
    'population': [],
    'habitat': []
}


def resetData(x):
    for item in x.keys():
        x[item] = []
    return x


def loadData(x: dict, value: str):
    v = value.split(",")
    l = len(list(x.keys()))
    ll = len(v)
    if(ll != l):
        return None

    i = 0
    for item in x.keys():
        x[item].append(v[i])
        i += 1
    return x


def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


inputs = "CONVEX,SMOOTH,YELLOW,BRUISES,ANISE,FREE,CROWDED,NARROW,WHITE,TAPERING,BULBOUS,SMOOTH,SMOOTH,WHITE,WHITE,PARTIAL,WHITE,ONE,PENDANT,BROWN,SEVERAL,WOODS"

x = resetData(x)
x = loadData(x, inputs)
y = reloaded.predict(input_fn(x))
y = y[0][0]

if(y <= 0.5):
    print("EDIBLE | 0")
else:
    print("POISONOUS | 1")
