import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from helpers.import_data import import_data

import warnings


warnings.filterwarnings("ignore")

class LSTM(tf.Module):
    def __init__(self):
        super().__init__()
    

    def build_model(self, loss: str="mse", learning_rate: float=0.001, metrics: list=["mean_absolute_percentage_error"]):
        self.model = Sequential([layers.Input((100, 3)),
                    layers.LSTM(128),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1)
                ])
        self.model.compile(loss=loss,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=metrics)
        return self

    def fit(self, X: np.array, y: np.array, epochs: int=200, verbose: int=0):        
        self.model.fit(X, y, epochs=epochs, verbose=verbose)
        return self

    def evaluate(self, X: np.array, y: np.array, epochs: int=200, verbose: int=0):
        test_loss, test_acc = self.model.evaluate(X, y, epochs=epochs, verbose=verbose)
        print(f'Accuracy: {test_acc}\nTest loss: {test_loss}')

    def predict(self, X: np.array, verbose: int=0):
        pred = self.model.predict(X, verbose=verbose)
        return np.array(pred).flatten()

if __name__ == "__main__":
    data, labels = import_data("Euro28")
    results = labels

    X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=1410)

    cnn = LSTM()
    model = cnn.build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(pred)