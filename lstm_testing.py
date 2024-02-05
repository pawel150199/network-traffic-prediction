from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import layers
import tensorflow as tf
import warnings

from helpers.import_data import import_data
from model.lstm import get_lstm


warnings.filterwarnings("ignore")
tf.config.set_visible_devices([], 'GPU')

data, labels = import_data("Euro28")
results = labels[:, 1]

X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=1410)

model = get_lstm()
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))
pred = model.predict(X_test).flatten()
mae = mean_absolute_percentage_error(y_test, pred)

plt.figure(figsize=(12,7))
plt.plot(history.history['loss'], linewidth=0.5, color="red", label="Training Loss")
plt.plot(history.history['val_loss'], linewidth=0.5, color="blue", label="Validation Loss")
plt.title("Model Fitting")
plt.legend()
plt.tight_layout()
plt.savefig("learnig-curves-lstm.png")
