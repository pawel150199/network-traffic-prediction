import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from import_data import import_data
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

dirname = "Euro28"
data, results = import_data(dirname)

# Data cleaning
data = data.mean(axis=1)
results = results[:, 3]

# Data split
X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2)

# Learning
mlp = MLPRegressor(hidden_layer_sizes=100, batch_size=20, max_iter=1000, activation="identity")
model = mlp.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)


print(f"Mean square error: {mse}")
print(f"Real data: {y_test}")
print(f"Predicted data: {y_pred}")

# Visualizations
plt.figure(figsize=(10,5))
plt.plot(y_pred, color="blue", linewidth=0.8, linestyle="dotted", label="Predicted data")
plt.plot(y_test, color="red", linewidth=0.6, label="Real data")
plt.ylabel("avgHighestSlot")
plt.title("Real vs Predicted Values")
plt.grid()
plt.legend()
plt.savefig("img/real_vs_predicted.png")

plt.figure(figsize=(10,5))
plt.plot(mlp.loss_curve_, color="blue", linestyle="dotted", linewidth=0.8, label="Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.legend()
plt.savefig("img/loss_curve.png")