import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, input, adj):
        h = self.W(input)
        N = h.size(0)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = F.leaky_relu(self.a(a_input).squeeze(2), negative_slope=self.alpha)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)


class GAT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, num_heads, dropout=0.5):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_layers = nn.ModuleList([GraphAttentionLayer(input_dim if i == 0 else hidden_dim * num_heads,
                                                                   hidden_dim, dropout=dropout) for i in
                                               range(num_layers)])
        self.out_att = GraphAttentionLayer(hidden_dim * num_heads, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.dropout(x)
        for layer in self.attention_layers:
            x = layer(x, adj)
        x = F.elu(torch.sum(x, dim=1))
        x = F.dropout(x, self.dropout.p)
        x = F.log_softmax(self.out_att(x, adj), dim=1)
        return x


"""
# Define model parameters
num_layers = 2
input_dim = 10
hidden_dim = 64
output_dim = 2
num_heads = 2

# Instantiate GAT model
gat_model = GAT(num_layers, input_dim, hidden_dim, output_dim, num_heads)

# Example input data
input_data = torch.randn(10, input_dim)  # Example input data
adjacency_matrix = torch.randn(10, 10)  # Example adjacency matrix

# Forward pass
output = gat_model(input_data, adjacency_matrix)
print(output.shape)  # Example output shape
"""
