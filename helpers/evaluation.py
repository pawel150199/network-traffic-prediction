import numpy as np
import warnings
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedKFold
from helpers.feature_selection import FeatureSelection


class Evaluator(object):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        random_state: int = None,
        metrics: list = [],
        storage_dir: str = None,
        n_splits: int = 5,
        n_repeats: int = 2,
        feature_selection=False,
    ):
        """Class is used for evaluate experiment"""

        self.random_state = random_state
        self.metrics = metrics
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.storage_dir = storage_dir
        self.X = X
        self.y = y
        self.feature_selection = feature_selection

        if self.storage_dir is not None:
            return
        else:
            raise ValueError("Directory cannot be None!")

    def evaluate(self, clfs: list, result_name: str):
        """Evaluation function"""

        warnings.filterwarnings("ignore")

        self.clfs = clfs
        param_num = 4
        rskf = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        # n_splits x n_repeats x clfs x metrics
        self.scores = np.zeros(
            (
                param_num,
                self.n_splits * self.n_repeats,
                len(self.clfs),
                len(self.metrics),
            )
        )

        for param_id in range(4):
            X = self.X
            y = self.y[:, param_id]
            for fold_id, (train, test) in enumerate(rskf.split(X, y)):
                for clf_id, clf_name in enumerate(clfs):
                    X_test = X[test]
                    X_train = X[train]
                    if clf_name in ["CART", "KNN", "SVR", "RF", "LR"]:
                        X_test = X_test.reshape((len(test), 300))
                        X_train = X_train.reshape((len(train), 300))
                        if self.feature_selection:
                            fs = FeatureSelection(self.random_state)
                            fs.fit(X_train, y[train])
                            X_train = fs.transform(X_train)
                            X_test = fs.transform(X_test)
                    elif clf_name == "MLP":
                        X_test = X_test.reshape((len(test), 300))
                        X_train = X_train.reshape((len(train), 300))
                    clf = clfs[clf_name]
                    clf.fit(X_train, y[train])
                    y_pred = clf.predict(X_test)
                    for metric_id, metric_name in enumerate(self.metrics):
                        # PARAM X FOLD X CLASSIFIER X METRIC
                        self.scores[param_id, fold_id, clf_id, metric_id] = self.metrics[metric_name](y[test], y_pred)

        self.mean = np.mean(self.scores, axis=1)
        self.std = np.std(self.scores, axis=1)

        np.save(f"{self.storage_dir}/{result_name}", self.scores)
        np.save(f"{self.storage_dir}/{result_name}-mean", self.mean)
        np.save(f"{self.storage_dir}/{result_name}-std", self.std)

    def process_ranks(self, result_name):
        """Calculate global ranks"""

        self.mean_ranks = []
        self.ranks = []

        for m, _ in enumerate(self.metrics):
            scores_ = self.mean[:, :, m]

            # Ranks
            ranks = []
            for row in scores_:
                ranks.append(rankdata(row).tolist())

            ranks = np.array(ranks)
            self.ranks.append(ranks)
            mean_ranks = np.mean(ranks, axis=0)
            self.mean_ranks.append(mean_ranks)

        self.mean_ranks = np.array(self.mean_ranks)
        self.ranks = np.array(self.ranks)

        np.save(f"{self.storage_dir}/{result_name}", self.ranks)
