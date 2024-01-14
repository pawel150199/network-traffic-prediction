from helpers.import_data import import_data
import numpy as np
from helpers.feature_selection import FeatureSelection
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from helpers.loggers import configureLogger
from helpers.metrics import accuracy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from helpers.evaluation import Evaluator
from sklearn.tree import DecisionTreeRegressor
from helpers.feature_selection import FeatureSelection
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 1410


def main(number):
    clfs = {
        "KNN": KNeighborsRegressor(),
        "CART": RandomForestRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(kernel="poly"),
        "TREE": DecisionTreeRegressor(),
    }

    metrics = {
        "Acc": accuracy,
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "R2S": r2_score,
    }
    parameters = [
        "highestSlot",
        "avgHighestSlot",
        "sumOfSlots",
        "avgActiveTransceivers",
    ]

    data, results = import_data("Euro28")

    data = data.reshape(100, 300)
    results = results[:, number]

    rfk = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
    fig, axes = plt.subplots(nrows=len(clfs), ncols=1, figsize=(12, 7 * len(clfs)))
    for i, (clf_name, clf_model) in enumerate(clfs.items()):
        fold_y_pred = []
        fold_y_pred_fs = []
        fold_y_test = []
        fold_y_test_fs = []

        for train_index, test_index in rfk.split(data, results):
            X_train, X_test = data[train_index], data[test_index]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            X_train_fs = X_train
            X_test_fs = X_test

            y_train, y_test = results[train_index], results[test_index]
            y_train_fs, y_test_fs = results[train_index], results[test_index]
            clf = clf_model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            r2_score_metric = r2_score(y_test, y_pred)
            fold_y_pred.append(y_pred)
            fold_y_test.append(y_test)

            fs = FeatureSelection(RANDOM_STATE)
            fs.fit(X_train_fs, y_train)
            X_train_t = fs.transform(X_train_fs)
            X_test_t = fs.transform(X_test_fs)

            clf_fs = clf_model
            clf_fs.fit(X_train_t, y_train)
            y_pred_fs = clf_fs.predict(X_test_t)
            r2_score_metric = r2_score(y_test_fs, y_pred_fs)
            fold_y_pred_fs.append(y_pred_fs)
            fold_y_test_fs.append(y_test_fs)

        dimensions = [arr.shape[0] for arr in fold_y_pred]
        min_dimension = min(dimensions)
        fold_y_pred_adjusted = [arr[:min_dimension] for arr in fold_y_pred]
        avg_y_pred = np.mean(np.vstack(fold_y_pred_adjusted), axis=0)

        dimensions_fs = [arr.shape[0] for arr in fold_y_pred_fs]
        min_dimension_fs = min(dimensions_fs)
        fold_y_pred_fs_adjusted = [arr[:min_dimension_fs] for arr in fold_y_pred_fs]
        avg_y_pred_fs = np.mean(np.vstack(fold_y_pred_fs_adjusted), axis=0)

        dimensions_test = [arr.shape[0] for arr in fold_y_test_fs]
        min_dimension_test = min(dimensions_test)
        fold_y_pred_adjusted_test = [arr[:min_dimension_test] for arr in fold_y_test]
        avg_y_test = np.mean(np.vstack(fold_y_pred_adjusted_test), axis=0)

        plt.subplot(len(clfs), 1, i + 1)

        plt.plot(
            avg_y_test,
            linewidth=0.5,
            color="black",
            linestyle="dotted",
            label="Real Values",
        )
        plt.plot(
            avg_y_pred_fs,
            linewidth=0.8,
            color="green",
            label="Predicted Values Feature Sel.",
        )
        plt.plot(
            avg_y_pred,
            linewidth=0.8,
            color="blue",
            label="Predicted Values",
        )
        plt.title(f"{clf_name} - Avg R^2 Score: {r2_score_metric:.4f}")
        plt.legend()
        fold_y_pred.clear()
        fold_y_pred_fs.clear()
        fold_y_test.clear()
        fold_y_test_fs.clear()
    plt.subplots_adjust(top=3, bottom=0.05, hspace=0.9)
    plt.tight_layout()
    plt.savefig(f"img/basic_regressors_{parameters[number]}.png")


if __name__ == "__main__":
    main(0)
