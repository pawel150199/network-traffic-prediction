import argparse
from pickletools import optimize
import warnings
from import_data import import_data
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from loggers import configureLogger


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    SEED = 1410
    logger = configureLogger()

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
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Result shape: {results.shape}")

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=SEED)

    scaler = MinMaxScaler()
    scaler.fit(X_train, y_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(random_state=SEED)
    pca.fit(X_train, y_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    print(X_train.shape)

    # Learning
    mlp = MLPRegressor(hidden_layer_sizes=10000, batch_size=30, activation="identity", solver="adam", random_state=SEED)
    model = mlp.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test).astype(int)

    mse = mean_absolute_error(y_test, y_pred)

    if args.verbose:
        logger.info(f"Mean square error: {mse}")

    # Visualizations
    plt.figure(figsize=(10,5))
    plt.plot(y_pred, color="blue", linewidth=0.8, linestyle="dotted", label="Predicted data")
    plt.plot(y_test, color="red", linewidth=0.6, label="Real data")
    plt.ylabel("avgActiveTransceivers")
    plt.title(f"Real vs Predicted Values - MAE {mse}")
    plt.grid()
    plt.legend()
    plt.savefig("img/real_vs_predicted.png")
    logger.info("Saved image in 'img/real_vs_predicted.png'")

    plt.figure(figsize=(10,5))
    plt.plot(mlp.loss_curve_, color="blue", linestyle="dotted", linewidth=0.8, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.legend()
    plt.savefig("img/loss_curve.png")
    logger.info("Saved image in 'img/loss_curve.png'")