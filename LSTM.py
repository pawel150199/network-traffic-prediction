import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error
from keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from helpers.import_data import import_data
from helpers.metrics import accuracy
import warnings


warnings.filterwarnings("ignore")
tf.config.set_visible_devices([], 'GPU')

data, labels = import_data("Euro28")
results = labels[:, 1]
#data = data.reshape(100,300)

X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=1410)

#model = Sequential([layers.Input((100, 3)),
#                    layers.LSTM(128),
#                    layers.Dense(256, activation="relu"),
#                    layers.Dense(256, activation="relu"),
#                    layers.Dense(512, activation="relu"),
#                    layers.Dense(256, activation="relu"),
#                    layers.Dense(128, activation="relu"),
#                    layers.Dense(64, activation="relu"),
#                    layers.Dense(32, activation="relu"),
#                    layers.Dense(1)])

model = Sequential([layers.Input((100, 3)),
                    layers.LSTM(21),
                    layers.Dense(20)])

model.compile(loss="mse",
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_percentage_error'])

    
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))
pred = model.predict(X_test).flatten()
mae = mean_absolute_percentage_error(y_test, pred)

plt.figure(figsize=(12,7))
plt.plot(pred, linewidth=0.8, color="red", label="Real Values")
plt.plot(y_test, linewidth=0.5, color="blue", linestyle="dotted", label="Predicted Values")
plt.title(f"Model predictions MAE: {mae:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("real-vs-pred-lstm.png")

plt.figure(figsize=(12,7))
plt.plot(history.history['loss'], linewidth=0.5, color="red", label="Training Loss")
plt.plot(history.history['val_loss'], linewidth=0.5, color="blue", label="Validation Loss")
plt.title("Model Fitting")
plt.legend()
plt.tight_layout()
plt.savefig("learnig-curves-lstm.png")
