# this file turns point cloud data into kNN graph structures for input into GNN models
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import math 

# finds the k value for kNN graph 
# for N nodes k = log(n)
def find_k_value(point_cloud):
    n = len(point_cloud)
    return int(math.log(n,10))

# kneightbors_graph returns a matrix with data on which points are neighbors
# The function returns a tensor representing the edge_index, which tells you how the points are connected in the k-NN graph.
def create_knn_graph(point_cloud):
    k = find_k_value(point_cloud)
    adjacency_matrix = kneighbors_graph(point_cloud, k, mode='connectivity', include_self=False).toarray()
    edge_index = np.array(np.nonzero(adjacency_matrix))
    return torch.tensor(edge_index, dtype=torch.long)

# Data from PyTorch Geometric holds graph data (in this case our data)
# x is the coordinates of each node
# edge_index is the connections between each node
def prepare_graph_data(point_cloud):
    edge_index = create_knn_graph(point_cloud)
    x = torch.tensor(point_cloud, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

####################################
# visualization of the graph structure
def plot_point_cloud_with_edges(graph_data):
    # Extract the point cloud and edge_index
    points = graph_data.x.numpy()
    edge_index = graph_data.edge_index.numpy()

    # Create a NetworkX graph from edge_index
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    # Plot the point cloud in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=20)

    # Draw edges in the k-NN graph
    for edge in G.edges():
        p1, p2 = points[edge[0]], points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='b', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("Hela Pitstop Point Cloud")
    plt.show(block=True)


if __name__ == "__main__":
    from preprocessing import preprocess_data
    
    # Example usage
    file_path1 = "data/class_1_real/point_cloud_1.csv" #DMSO
    point_cloud1 = preprocess_data(file_path1)

    file_path2 = "data/class_2_real/point_cloud_2.csv" #Pitstop
    point_cloud2 = preprocess_data(file_path2)
    
    graph_data1 = prepare_graph_data(point_cloud1)
    print(graph_data1)

    # Visualize the point cloud and k-NN graph
    plot_point_cloud_with_edges(graph_data1)

    graph_data2 = prepare_graph_data(point_cloud2)
    print(graph_data2)

    # Visualize the point cloud and k-NN graph
    plot_point_cloud_with_edges(graph_data2)