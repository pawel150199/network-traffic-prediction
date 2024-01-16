from distutils.log import error
import numpy as np
import warnings
import os
from tabulate import tabulate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind



def main():
    RANDOM_STATE = 1410
    alpha=0.05
    m_fmt="%3f"
    std_fmt=None
    nc="---"
    db_fmt="%s"
    tablefmt="plain"

    clfs = ["CART", "SVR", "KNN", "RF", "LR", "CART-FS", "SVR-FS", "KNN-FS", "RF-FS", "LR-FS"]    

    scores_1 = np.load("results/scores_part_simple_euro28.npy")
    scores_2 = np.load("results/scores_part_simple_feature_selection_euro28.npy")
    scores = np.concatenate((scores_1, scores_2), axis=2)
    

    mean_scores_1 = np.load("results/scores_part_simple_euro28-mean.npy")
    mean_scores_2 = np.load("results/scores_part_simple_feature_selection_euro28-mean.npy")
    mean_scores = np.concatenate((mean_scores_1, mean_scores_2), axis=1)
   
    std_1 = np.load("results/scores_part_simple_euro28-std.npy")
    std_2 = np.load("results/scores_part_simple_feature_selection_euro28-std.npy")
    std = np.concatenate((std_1, std_2), axis=1)

    np.save("results/scores_experiment_1_euro28", scores)
    np.save("results/scores_experiment_1_euro28-mean", mean_scores)
    np.save("results/scores_experiment_1_euro28-std", std)

    parameters = [
        "highestSlot",
        "avgHighestSlot",
        "sumOfSlots",
        "avgActiveTransceivers",
    ]

    n_clfs = len(clfs)
    t = []
    # Generate table
    for param_idx, param_name in enumerate(parameters):
        # Mean value
        t.append([db_fmt % param_name] + ["%.3f" % v for v in mean_scores[param_idx, :]])
        # If std_fmt is not None, std will appear in tables
        if std_fmt:
            t.append([std_fmt % v for v in std[param_idx, :]])
        # Calculate T and P for T-student test
        T, p = np.array(
            [[ttest_ind(scores[param_idx, :, i],
                scores[param_idx, :, j], random_state=RANDOM_STATE)
            for i in range(len(clfs))]
            for j in range(len(clfs))]
        ).swapaxes(0, 2)
        T = -T
        _ = np.where((p < alpha) * (T > 0))
        conclusions = [list(1 + _[1][_[0] == i])
                    for i in range(n_clfs)]
                
        t.append([''] + [", ".join(["%i" % i for i in c])
                                if len(c) > 0 and len(c) < len(clfs) - 1 else ("all" if len(c) == len(clfs)-1 else nc)
                                for c in conclusions])
    # Show outputs
    headers = ["Parameter"]
    for i in clfs:
        headers.append(i)

    print(tabulate(t, headers, tablefmt="grid"))
    # Save outputs as .tex extension
    with open("tables/%s.txt" % ("experiment_1_euro28"), "w") as f:
        f.write(tabulate(t, headers, tablefmt="latex"))

if __name__ == "__main__":
    main()