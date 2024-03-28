import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from models.mlp import MLP


class RandomForestMLP(nn.Module):
    def __init__(self, num_trees, num_layers, input_dim, hidden_dim, output_dim):
        super(RandomForestMLP, self).__init__()

        self.num_trees = num_trees
        self.mlp_models = nn.ModuleList([MLP(num_layers, input_dim, hidden_dim, output_dim) for _ in range(num_trees)])

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.mlp_models])
        return torch.mean(outputs, dim=0)


"""
# Define model parameters
num_trees = 5
num_layers = 2
input_dim = 10
hidden_dim = 64
output_dim = 2

# Instantiate the random forest of MLPs
random_forest = RandomForestMLP(num_trees, num_layers, input_dim, hidden_dim, output_dim)

# Example input data
input_data = torch.randn(10, input_dim)  # Example input data

# Forward pass
output = random_forest(input_data)
print(output.shape)  # Example output shape
"""
