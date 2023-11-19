import numpy as np
import matplotlib.pyplot as plt
from helpers.import_data import import_data
from helpers.feature_selection import FeatureSelection
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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

    y_pred_st = []
    y_pred_t = []
    y_pred = []

    fig, ax = plt.subplots(3,4, figsize=(20,15))
    axs = ax.flatten()

    # Uncomment choosen classifier
    estimator = MLPRegressor(hidden_layer_sizes=1000, batch_size=30, activation="identity", solver="adam", random_state=RANDOM_STATE)
    estimator_name = "MLP"

    # estimator = SVR(kernel="linear")
    # estimator_name = "SVR"

    # estimator = RandomForestRegressor(criterion="absolute_error")
    # estimator_name = "MLP"

    for i in range(length):
        X_train, X_test, y_train, y_test = train_test_split(X, y[:, i], test_size=0.2, random_state=RANDOM_STATE)

        sc = StandardScaler()
        sc.fit(X_train, y_train)
        X_train_s = sc.transform(X_train)
        X_test_s = sc.transform(X_test)

        fs = FeatureSelection(random_state=RANDOM_STATE)
        fs.fit(X_train_s, y_train)
        X_train_st = fs.transform(X_train_s)
        X_test_st = fs.transform(X_test_s)

        fs = FeatureSelection(random_state=RANDOM_STATE)
        fs.fit(X_train, y_train)
        X_train_t = fs.transform(X_train)
        X_test_t = fs.transform(X_test)

        mlp_st = estimator
        mlp_st.fit(X_train_st, y_train)
        pred_st = mlp_st.predict(X_test_st)
        y_pred_st.append(pred_st)
        mae_st = (mean_absolute_error(y_test, pred_st))

        mlp_t = estimator
        mlp_t.fit(X_train_t, y_train)
        pred_t = mlp_t.predict(X_test_t)
        y_pred_t.append(pred_t)
        mae_t = (mean_absolute_error(y_test, pred_t))

        mlp = estimator
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

        axs[i+8].plot(y_pred_st[i], color="blue", linewidth=0.8, linestyle="dotted", label="Predicted data")
        axs[i+8].plot(y_test, color="red", linewidth=0.6, label="Real data")
        axs[i+8].set_xlabel("Iteration")
        axs[i+8].set_title(f"Selected Features Predition with Scaler -> MAE: {mae_st:.2f}")

    plt.suptitle(f"Prediction each parameter ({dataset}) ({estimator_name})")
    plt.tight_layout()
    plt.savefig(f"img/selected_feature_experiment_{dataset}_{estimator_name}.png")
    logger.info("Image succesfully saved.")
