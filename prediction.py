import os
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--verbose",
        "-v",
        help="Verbose",
        action="store_true",
        required=False
    )

    args = parser.parse_args()

    dirname = "Euro28"
    data, results = import_data(dirname)

    # Data cleaning
    data = data.mean(axis=1)
    results = results[:, 3]

    if args.verbose:
        print(f"Data shape: {data.shape}\n\n")
        print(f"Result shape: {results.shape}\n\n")

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2)

    # Learning
    mlp = MLPRegressor(hidden_layer_sizes=100, batch_size=20, max_iter=1000, activation="identity")
    model = mlp.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test).astype(int)

    mse = mean_squared_error(y_test, y_pred)

    if args.verbose:
        print(f"Mean square error: {mse}\n\n")
        print(f"Real data: {y_test}\n\n")
        print(f"Predicted data: {y_pred}\n\n")

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