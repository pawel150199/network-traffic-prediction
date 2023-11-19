import numpy as np
import matplotlib.pyplot as plt
from helpers.import_data import import_data
from helpers.feature_selection import FeatureSelection
from sklearn.neural_network import MLPRegressor
from helpers.loggers import configureLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    RANDOM_STATE = 1410
    logger = configureLogger()

    dataset = "Euro28"
    data, results = import_data(dataset)

    X = data.reshape(100,300)
    y = results

    length = y.shape[1]

    y_pred_t = []
    y_pred = []

    fig, ax = plt.subplots(4,2, figsize=(12,12))
    axs = ax.flatten()

    for i in range(length):
        X_train, X_test, y_train, y_test = train_test_split(X, y[:, i], test_size=0.2, random_state=RANDOM_STATE)
        fs = FeatureSelection(random_state=RANDOM_STATE)
        fs.fit(X_train, y_train)

        X_train_t = fs.transform(X_train)
        X_test_t = fs.transform(X_test)

        mlp_t = MLPRegressor(hidden_layer_sizes=1000, batch_size=30, activation="identity", solver="adam", random_state=RANDOM_STATE)
        mlp_t.fit(X_train_t, y_train)
        pred_t = mlp_t.predict(X_test_t)
        y_pred_t.append(pred_t)
        mae_t = (mean_absolute_error(y_test, pred_t))

        mlp = MLPRegressor(hidden_layer_sizes=1000, batch_size=30, activation="identity", solver="adam", random_state=RANDOM_STATE)
        mlp.fit(X_train, y_train)
        pred = mlp.predict(X_test)
        y_pred.append(pred)
        mae = (mean_absolute_error(y_test, pred))
    
        axs[i].plot(y_pred_t[i], color="blue", linewidth=0.8, linestyle="dotted", label="Predicted data")
        axs[i].plot(y_test, color="red", linewidth=0.6, label="Real data")
        axs[i].set_xlabel("Iteration")
        axs[i].set_title(f"Selected Features Prediction -> MAE: {mae_t:.2f}")
        
        axs[i+4].plot(y_pred[i], color="blue", linewidth=0.8, linestyle="dotted", label="Predicted data")
        axs[i+4].plot(y_test, color="red", linewidth=0.6, label="Real data")
        axs[i+4].set_xlabel("Iteration")
        axs[i+4].set_title(f"Original Features -> MAE: {mae:.2f}")

    plt.suptitle(f"Prediction each parameter ({dataset})")
    plt.tight_layout()
    plt.savefig(f"img/selected_feature_experiment_{dataset}.png")
    logger.info("Image succesfully saved.")
