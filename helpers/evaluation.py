import numpy as np
import os
import warnings
from sklearn.base import clone
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

class Evaluator(object):
    def __init__(self, outputs: np.array,  X: np.array, y: np.array, random_state: int = None, metrics: list = [], storage_dir: str = None, n_splits: int = 5, n_repeats: int = 2):
        """Class is used for evaluate experiment"""

        self.random_state = random_state
        self.metrics = metrics
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.storage_dir = storage_dir
        self.outputs = outputs
        self.X = X
        self.y = y

        if self.storage_dir is not None:
            return
        else: 
            raise ValueError("Directory cannot be None!")
    
    def evaluate(self, clfs: list, result_name: str):
        """Evaluation function"""

        warnings.filterwarnings("ignore")

        self.clfs = clfs
        rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state = self.random_state)
        # n_splits x n_repeats x clfs x metrics
        self.scores = np.zeros((self.n_splits*self.n_repeats, len(self.clfs), len(self.metrics)))

        for fold_id, (train, test) in enumerate(rskf.split(self.X, self.y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clfs[clf_name]
                clf.fit(self.X[train], self.y[train])
                y_pred = clf.predict(self.X[test])
                for metric_id, metric_name in enumerate(self.metrics):
                # DATA X FOLD X CLASSIFIER X METRIC 
                    self.scores[fold_id, clf_id, metric_id] = self.metrics[metric_name](self.y[test],y_pred)
        
        self.mean = np.mean(self.scores, axis=1)
        self.std = np.std(self.scores, axis=1)

        np.save(f"results/{self.storage_dir}/{result_name}", self.scores)
        np.save(f"results/{self.storage_dir}/{result_name}-mean", self.mean)
        np.save(f"results/{self.storage_dir}/{result_name}-std", self.std)

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

        np.save(f"results/{self.storage_dir}/{result_name}", self.ranks)
