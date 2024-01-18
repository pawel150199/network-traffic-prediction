import numpy as np
from tabulate import tabulate

scores_1 = np.load("results/no-feature-selection-time_us26-mean.npy")
scores_2 = np.load("results/time_us26-mean.npy")

scores = np.concatenate((scores_1, scores_2), axis=1)

headers = ["CART", "KNN", "SVR", "RF", "LR", "CART-FS", "KNN-FS", "SVR-FS", "RF-FS", "LR-FS"]
row_names = ["highestSlot", "avgHighestSlot", "sumOfSlots", "avgActiveTransceivers"]

print(scores_1.shape)
print(scores.shape)
print(scores)

tab = tabulate(scores, headers=headers, floatfmt="4f", tablefmt="grid")

with open("tables/%s.txt" % ("time_us26"), "w") as f:
    f.write(tabulate(scores, headers=headers, floatfmt="4f", tablefmt="latex"))


print(tab)