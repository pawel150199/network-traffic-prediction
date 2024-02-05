import warnings
import numpy as np
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential


warnings.filterwarnings("ignore")

def get_cnn(loss: str="mse", learning_rate: float=0.001, metrics: list=["mean_absolute_percentage_error"]):
    model = Sequential()
    model.add(layers.Con2D(filter=32, kernel_size=(3,3), padding="same", activation="relu", input_shape=(25,12,1)))
    model.add(layers.Conv2D(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1))

    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=metrics)

    return model