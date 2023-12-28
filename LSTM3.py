import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers

import warnings


warnings.filterwarnings("ignore")

class NeuralNetworkModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
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
        self.model.compile(loss="mse",
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_error'])
        return self.model

    def train(self, X_train, y_train, epochs=200):
        self.model.fit(X_train, y_train, epochs=epochs)

    def evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'Test accuracy: {test_acc}\nTest loss: {test_loss}')

    def predict(self, X):
        return self.model.predict(X).flatten()