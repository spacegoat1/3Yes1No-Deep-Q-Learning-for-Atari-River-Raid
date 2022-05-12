import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dq_model(input_shape, hidden_size, num_actions):
    inputs = layers.Input(shape=input_shape)

    layer1 = layers.Conv2D(32, 12, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 6, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 4, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(hidden_size, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

