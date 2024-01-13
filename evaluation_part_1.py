from helpers.import_data import import_data
import numpy as np
from helpers.feature_selection import FeatureSelection
from helpers.loggers import configureLogger
from helpers.metrics import accuracy
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from helpers.evaluation import Evaluator

import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 1410

def main():
    # Import data
    data_dir = "Euro28"
    X, y = import_data(data_dir)

    # CLassificators
    clfs = {
        'KNN' : KNeighborsRegressor(),
        'CART' : RandomForestRegressor(random_state=RANDOM_STATE),
        'SVR' : SVR(kernel="poly")
    }

    # Metrics
    metrics = {
    'Acc' : accuracy,
    'MAE' : mean_absolute_error,
    'MSE' : mean_squared_error,
    'R2S' : r2_score
    }

    

    ev = Evaluator(X=X, y=y, storage_dir="results", random_state=RANDOM_STATE, metrics=metrics)
    ev.evaluate(clfs, result_name=f"scores_euro28")
    ev.process_ranks(result_name=f"ranks_euro28")

if __name__ == "__main__":
    main()