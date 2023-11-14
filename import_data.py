import os
import numpy as np

def import_data(directory: str) -> None:
    """
    Load all data to learning from specific directory

    Args:
        directory (str): directory where all data are stored

    Returns:
        np.array: data from simulations in readable form shape (100,100,3) where we have 100 simulations, 100 requests and 3 features (input node, output node, bitrate
        np.array: labels to predictions from simulation in readable form with shape (100,4) where we have 100 simulations and 4 data to predictions (highestSlot, avgHighestSlot, sumOfSlots, avgActiveTransceivers)
    """
    
    data = []
    results = []

    for i in os.listdir(directory):
        d = np.genfromtxt(f"{directory}/{i}/requests.csv", delimiter=',', skip_header=1, dtype=float)
        data.append(d[:, 1:])
        results.append(np.genfromtxt(f"{directory}/{i}/results.txt", dtype=float, usecols=(1,)))

    return np.array(data), np.array(results)