import random
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings

from helpers.metrics import accuracy
from helpers.import_data import import_data

warnings.filterwarnings("ignore")

data, labels = import_data("Euro28")
#data = data.reshape(100,300)

print(data.shape)

data = np.reshape(data, (100,25,12,1))

print(data.shape)

tf.config.set_visible_devices([], 'GPU')


results = labels[:, 2]

X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=1410)

model = Sequential([layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", input_shape=(25,12,1)),
                    layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
                    layers.MaxPooling2D(pool_size=(2,2)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(1)])

model.compile(loss="mse",
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_error'])

earlyStopping = EarlyStopping(monitor='val_loss',
                            patience=3,
                            verbose=1)

modelCheckpoint = ModelCheckpoint(filepath="best_lstm.h5",
                                monitor='val_loss',
                                save_best_only=True)
    
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

pred = model.predict(X_test)
print(pred.shape)
pred = pred.flatten()

mae = mean_absolute_percentage_error(y_test, pred) 


print(f"\n\nPredicted values: {pred}")
print(f"Real values: {y_test}")

plt.figure(figsize=(12,7))
plt.plot(pred, linewidth=0.8, color="red", label="Real Values")
plt.plot(y_test, linewidth=0.5, color="blue", linestyle="dotted", label="Predicted Values")
plt.title(f"Model predictions MAE {mae:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("real-vs-pred-cnn.png")

plt.figure(figsize=(12,7))
plt.plot(history.history['loss'], linewidth=0.5, color="red", label="Training Loss")
plt.plot(history.history['val_loss'], linewidth=0.5, color="blue", label="Validation Loss")
plt.title("Model Fitting")
plt.legend()
plt.tight_layout()
plt.savefig("learnig-curves-cnn.png")
