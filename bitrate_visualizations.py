import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Print single, random bitrate
data = pd.read_csv("Euro28/request-set_1/requests.csv")

plt.figure(figsize=(14,9))
plt.plot(data["bitrate"], color="blue", linewidth=0.6)
plt.xlabel("Request Sequence")
plt.ylabel("Bitrate")
plt.title("Bitrate")
plt.tight_layout()
plt.savefig("img/single_bitrate.png")

# Print all bitrates

plt.figure(figsize=(14,9))

for i in os.listdir("Euro28"):
    data = pd.read_csv(f"Euro28/{i}/requests.csv")
    plt.plot(data["bitrate"], linewidth=0.6, label=i)
    plt.xlabel("Request Sequence")
    plt.ylabel("Bitrate")
    plt.title("Bitrate")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)
plt.tight_layout()
plt.savefig("img/all_bitrates.png")