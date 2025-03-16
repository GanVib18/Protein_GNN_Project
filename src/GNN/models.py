# Core libraries
import numpy as np  
import pandas as pd  
from matplotlib.pylab import plt  
import itertools 

# PyTorch Geometric modules
from torch_geometric.loader import DataLoader 
import torch  
from torch.nn import Linear  
import torch.nn.functional as F  
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool  
from torch_geometric.nn import global_max_pool

# General PyTorch components
from torch import Tensor 
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, BatchNorm1d, ReLU, Parameter
from torch_geometric.nn import MessagePassing, global_max_pool
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear, BatchNorm1d, ReLU

# Graph Convolutional Network components
from torch_geometric.nn import GCNConv, global_max_pool 

# Custom utility modules
from graph_construction import prepare_graph_data
from preprocessing import preprocess_data 

# Scikit-learn modules for evaluation and data splitting
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold  
from sklearn.metrics import ( 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Number of features and classes in the dataset
num_features = 3
num_classes = 2

# Original code taken from
# https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/point_cloud.html

# Building a PointNet Layer, which will be used to define our network
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='max')  # Aggregation type: 'max' for maximum pooling

        # Multi-layer perceptron (MLP) for learning transformations
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),  # Input includes node features + relative positions
            ReLU(),  # Activation function
            Linear(out_channels, out_channels),  # Output layer
        )
        
    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass to propagate messages.
        h: Node features
        pos: Node positions
        edge_index: Connectivity information (edges)
        """
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:
        """
        Message defines how information is aggregated.
        h_j: Features of source nodes
        pos_j, pos_i: Positions of source and target nodes
        """
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)  # Combine features and relative positions
        return self.mlp(edge_feat)  # Pass through MLP

# PointNet model
class PointNet(torch.nn.Module):
    def __init__(self, hidden_dim=32, num_classes=2):
        super().__init__()
        self.conv1 = PointNetLayer(3, hidden_dim)  # First PointNet layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.conv2 = PointNetLayer(hidden_dim, hidden_dim * 2)  # Second PointNet layer
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 2)  # Batch normalization
        self.classifier = Linear(hidden_dim * 2, num_classes)  # Fully connected classification layer

    def forward(self, pos, edge_index, batch):
        """
        Forward pass through PointNet.
        pos: Node positions
        edge_index: Graph edges
        batch: Batch information for pooling
        """
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = self.bn1(h).relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = self.bn2(h).relu()
        h = global_max_pool(h, batch)  # Global pooling for graph-level output
        return self.classifier(h)

class GCNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # Aggregation type: 'add' for summation
        self.linear = Linear(in_channels, out_channels)  # Linear transformation

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GCN layer.
        h: Node features
        edge_index: Edge indices
        """
        # Add self-loops to ensure each node contributes to its own representation
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        # Compute normalization (degree-based)
        row, col = edge_index
        deg = degree(row, h.size(0), dtype=h.dtype)  # Degree of each node
        deg_inv_sqrt = deg.pow(-0.5)  # Inverse square root of degrees
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # Normalization factor

        # Propagate messages and apply normalization
        return self.propagate(edge_index, h=h, norm=norm)

    def message(self, h_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Message function for message passing.
        h_j: Features of source nodes
        norm: Normalization factor
        """
        return norm.view(-1, 1) * h_j  # Scale features by normalization factor

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        Update function to apply the linear transformation.
        aggr_out: Aggregated features
        """
        return self.linear(aggr_out)  # Apply linear transformation

class GCNNet(torch.nn.Module):
    def __init__(self, hidden_dim=32, num_classes=2):
        super().__init__()
        self.conv1 = GCNLayer(3, hidden_dim)  # First GCN layer
        self.bn1 = BatchNorm1d(hidden_dim)  # Batch normalization
        self.conv2 = GCNLayer(hidden_dim, hidden_dim * 2)  # Second GCN layer
        self.bn2 = BatchNorm1d(hidden_dim * 2)  # Batch normalization
        self.classifier = Linear(hidden_dim * 2, num_classes)  # Fully connected classification layer

    def forward(self, pos, edge_index, batch):
        """
        Forward pass for GCN network.
        pos: Node positions (used as initial features)
        edge_index: Graph edges
        batch: Batch information for pooling
        """
        h = self.conv1(h=pos, edge_index=edge_index)  # First GCN layer
        h = self.bn1(h).relu()  # Apply batch norm and ReLU activation
        h = self.conv2(h=h, edge_index=edge_index)  # Second GCN layer
        h = self.bn2(h).relu()  # Apply batch norm and ReLU activation
        h = global_max_pool(h, batch)  # Global pooling for graph-level representation
        return self.classifier(h)  # Classification layer

class GatedGraphLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # Aggregation type: 'add'
        self.linear = Linear(in_channels, out_channels)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Add self-loops to ensure each node contributes to its own representation
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        # Compute normalization (degree-based)
        row, col = edge_index
        deg = degree(row, h.size(0), dtype=h.dtype)  # Degree of each node
        deg_inv_sqrt = deg.pow(-0.5)  # Inverse square root of degrees
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # Normalization factor

        # Propagate messages and apply normalization
        return self.propagate(edge_index, h=h, norm=norm)

    def message(self, h_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * h_j  # Scale features by normalization factor

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.linear(aggr_out)  # Apply linear transformation

class GGSNN(torch.nn.Module):
    def __init__(self, hidden_dim=32, num_classes=2):
        super(GGSNN, self).__init__()

        self.conv1 = GatedGraphLayer(3, hidden_dim)  # First GGS-NN Layer
        self.conv2 = GatedGraphLayer(hidden_dim, hidden_dim * 2)  # Second GGS-NN Layer
        self.classifier = Linear(hidden_dim * 2, num_classes)  # Fully connected layer for classification

    def forward(self, pos, edge_index, batch):
        # Forward pass through the network
        h = self.conv1(pos, edge_index)  # Apply first GGS-NN layer
        h = torch.relu(h)  # Apply ReLU activation
        h = self.conv2(h, edge_index)  # Apply second GGS-NN layer
        h = torch.relu(h)  # Apply ReLU activation

        h = global_max_pool(h, batch)  # Global pooling to aggregate the graph-level features
        return self.classifier(h)  # Classification output