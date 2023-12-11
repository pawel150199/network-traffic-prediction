import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

def visualization(path_to_data: str, save_path: str) -> None:
    
    """
    Draw graph from raw csv data

    Args:
        path_to_data (str): Path to csv file which represents network graph
        save_path (str): Path where save graph image

    Usage::

    from visualize_graph import visualization

    path_to_data = "data"
    save_path = "graph.png"
    
    visualization(path_to_data=path_to_data, save_path=save_path)
    """    

    warnings.filterwarnings("ignore")

    # Load your data
    data = pd.read_csv(path_to_data)

    # Create a weighted graph
    G = nx.Graph()

    source = np.array(data["source"])
    destination = np.array(data["destination"])
    weight = np.array(data["bitrate"])

    # Normalize the weight values
    normalized_weights = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))

    for i in range(len(data)):
        G.add_edge(source[i], destination[i], weight=normalized_weights[i])

    # Set node attributes for visualization
    node_size = 200
    node_color = 'lightblue'

    # Create a colormap to represent edge weights
    cmap = plt.get_cmap('viridis')

    # Customize the layout
    pos = nx.spring_layout(G, seed=42)

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
    plt.colorbar(sm, label="Edges Weight (Bitrate in Mbps)")
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(save_path)
    