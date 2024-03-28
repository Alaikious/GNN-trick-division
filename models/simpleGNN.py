import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adjacency_matrix, feature_matrix):
        # GCN layer operation
        output = torch.matmul(adjacency_matrix, feature_matrix)
        output = self.linear(output)
        output = F.relu(output)
        return output


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GCNLayer(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(GCNLayer(hidden_dim, output_dim))

    def forward(self, adjacency_matrix, feature_matrix):
        # GCN model forward pass
        h = feature_matrix
        for layer in self.layers:
            h = layer(adjacency_matrix, h)
        return h


"""
num_layers = 3
input_dim = 10
hidden_dim = 64
output_dim = 2

# Instantiate GCN model
gcn_model = GCN(num_layers, input_dim, hidden_dim, output_dim)

# Example input data (adjacency matrix and feature matrix)
adjacency_matrix = torch.randn(10, 10)  # Example adjacency matrix
feature_matrix = torch.randn(10, input_dim)  # Example feature matrix

# Forward pass
output = gcn_model(adjacency_matrix, feature_matrix)
print(output.shape)  # Example output shape
"""
