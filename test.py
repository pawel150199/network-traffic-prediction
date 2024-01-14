import numpy as np
from tabulate import tabulate

scores = np.load("/Users/pawelpolski/Desktop/network-traffic-prediction/results/scores_main_evaluation_euro28-mean.npy")
headers = ["CART", "KNN", "SVR", "RF", "MLP", "LR"]
row_names = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]

print(scores.shape)
print(scores)

tab = tabulate(scores, headers=headers, floatfmt="4f", tablefmt="grid")

print(tab)