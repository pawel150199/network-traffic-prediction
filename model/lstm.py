import warnings
import numpy as np
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential


warnings.filterwarnings("ignore")

def get_lstm(loss: str="mse", learning_rate: float=0.001, metrics: list=["mean_absolute_percentage_error"]):
    model = Sequential()
    model.add(layers.Input(100,3))
    model.add(layers.LSTM(units=128, activation="tanh"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics)

    return model