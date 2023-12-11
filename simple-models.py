import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import warnings

from helpers.import_data import import_data

warnings.filterwarnings("ignore")

data, labels = import_data("Euro28")
data = data.reshape(100,300)

print(data.shape)

results = labels[:, 2]

X_train, X_test, y_train, y_test = train_test_split(data, results, test_size=0.2, random_state=1410)
