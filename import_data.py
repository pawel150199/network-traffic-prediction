import os
import numpy as np

def import_data(directory: str) -> None:
    data = []
    results = []

    for i in os.listdir(directory):
        d = np.genfromtxt(f"{directory}/{i}/requests.csv", delimiter=',', skip_header=1, dtype=float)
        data.append(d[:, 1:])
        results.append(np.genfromtxt(f"{directory}/{i}/results.txt", dtype=float, usecols=(1,)))

    return np.array(data), np.array(results)