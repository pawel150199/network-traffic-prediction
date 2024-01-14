import warnings
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from helpers.import_data import import_data
from helpers.evaluation import Evaluator
from LSTMModel import LSTM
from GRUModel import GRU
from CNNModel import CNN


warnings.filterwarnings("ignore")

RANDOM_STATE=1410

def main(name: str, dataset: str):

    # Import data
    X, y = import_data(dataset)

    # Clasificators
    clfs = {
        "CART" : DecisionTreeRegressor(random_state=RANDOM_STATE),
        "KNN" : KNeighborsRegressor(),
        "SVR" : SVR(kernel="poly"),
        "RF" : RandomForestRegressor(random_state=RANDOM_STATE),
        "MLP" : MLPRegressor(hidden_layer_sizes=50, batch_size=25, random_state=RANDOM_STATE, warm_start=True),
        "LR" : LinearRegression(),
        #"LSTM" : LSTM().build_model(),
        #"GRU" : GRU().build_model(),
        #"CNN" : CNN().build_model()
    }

    # Metrics
    metrics = {
        "MAPE" : mean_absolute_percentage_error
    }

    ev = Evaluator(storage_dir="results", X=X, y=y, random_state=RANDOM_STATE, metrics=metrics)
    ev.evaluate(clfs, result_name=f"scores_{name}")
    ev.process_ranks(result_name=f"ranks_{name}")

if __name__ == "__main__":
    main(name="main_evaluation_euro28", dataset="Euro28")
    #main(name="main_evaluation_us26", dataset="US26")
