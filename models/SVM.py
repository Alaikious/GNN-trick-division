import torch
import torch.nn as nn
import torch.nn.functional as F


class SVM(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
        '''

        super(SVM, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, 1)  # Output dimension fixed to 1 for SVM
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, 1))  # Output dimension fixed to 1 for SVM

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


"""
# Define model parameters
num_layers = 3
input_dim = 10
hidden_dim = 64

# Instantiate SVM model
svm_model = SVM(num_layers, input_dim, hidden_dim)

# Example input data
input_data = torch.randn(10, input_dim)  # Example input data

# Forward pass
output = svm_model(input_data)
print(output.shape)  # Example output shape
"""
