import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split

from helpers.import_data import import_data

import warnings

warnings.filterwarnings("ignore")

class CNN(tf.Module):
    def __init__(self):
        super().__init__()
        

    def build_model(self, loss: str="mse", learning_rate: float=0.001, metrics: list=["mean_absolute_error"]):
        self.model = Sequential([layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", input_shape=(25,12,1)),
                    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(1)
                ])
        self.model.compile(loss=loss,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=metrics)

        return self

    def fit(self, X: np.array, y: np.array, epochs: int=200):
        X = np.reshape(X, (100,25,12,1))
        self.model.fit(X, y, epochs=epochs)
        return self

    def evaluate(self, X: np.array, y: np.array):
        X = np.reshape(X, (100,25,12,1))
        test_loss, test_acc = self.model.evaluate(X, y)
        print(f'Accuracy: {test_acc}\nLoss: {test_loss}')

    def predict(self, X: np.array):
        pred = self.model.predict(X)
        return np.array(pred).flatten()

if __name__ == "__main__":
    data, labels = import_data("Euro28")
    data = np.reshape(data, (100,25,12,1))
    results = labels[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=1410)

    cnn = CNN()
    model = cnn.build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(pred)
