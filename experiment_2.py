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
from helpers.statistic_test_evaluation import StatisticTest
from LSTMModel import LSTM
from GRUModel import GRU
from CNNModel import CNN
from model.cnn import get_cnn
from model.gru import get_gru
from model.lstm import get_lstm


warnings.filterwarnings("ignore")

RANDOM_STATE = 1410

def experiment2(name: str, dataset: str):
    # Import data
    X, y = import_data(dataset)

    # Clasificators
    clfs = {
        "CART" : DecisionTreeRegressor(random_state=RANDOM_STATE),
        "SVR" : SVR(kernel="poly"),
        "KNN" : KNeighborsRegressor(),
        "RF" : RandomForestRegressor(random_state=RANDOM_STATE),
        "LR" : LinearRegression(),
        "MLP" : MLPRegressor(hidden_layer_sizes=50, batch_size=25, random_state=RANDOM_STATE, warm_start=True),
        "LSTM" : get_lstm(),
        "GRU" : get_gru(),
        "CNN" : get_cnn()
    }

    # Metrics
    metrics = {"MAPE": mean_absolute_percentage_error}

    ev = Evaluator(
        storage_dir="results",
        X=X,
        y=y,
        random_state=RANDOM_STATE,
        metrics=metrics,
        feature_selection=True,
    )
    ev.evaluate(clfs, result_name=f"scores_{name}")
    ev.process_ranks(result_name=f"ranks_{name}")
    st = StatisticTest(ev)
    st.process(table_name=f"{name}")


if __name__ == "__main__":
    experiment2(name="experiment_2_euro28", dataset="Euro28")
    experiment2(name="experiment_2_us26", dataset="US26")
