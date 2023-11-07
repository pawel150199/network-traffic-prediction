import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load your data
data = pd.read_csv("Euro28/request-set_1/requests.csv")

# Create a weighted graph
G = nx.Graph()

source = np.array(data["source"])
destination = np.array(data["destination"])
weight = np.array(data["bitrate"])

# Normalize the weight values
normalized_weights = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))

for i in range(len(data)):
    # Add weighted edges
    G.add_edge(source[i], destination[i], weight=normalized_weights[i])

# Set node attributes for visualization
node_size = 200  # Adjust the size of the nodes
node_color = 'lightblue'

# Create a colormap to represent edge weights
cmap = plt.get_cmap('viridis')

# Customize the layout
pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility

# Draw the graph with edge colors
edge_colors = [cmap(w) for w in normalized_weights]
nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color, font_size=8, font_color='black',
        width=2, edge_color=edge_colors)

# Add labels to the nodes
node_labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

# Create a color bar for edge colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=weight.min(), vmax=weight.max()))
sm.set_array([])
plt.colorbar(sm, label="Edge Weights")

plt.tight_layout()
plt.axis('off')
plt.savefig("img/graph.png")