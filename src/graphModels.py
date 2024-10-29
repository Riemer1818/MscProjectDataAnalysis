# graphModels.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

# =========================
# GRAPH CONVOLUTIONAL NETWORK
# =========================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))  # Apply sigmoid activation for output between 0 and 1

# =========================
# GRAPH ATTENTION NETWORK (GAT)
# =========================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, heads=1):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)

# =========================
# GRAPH ATTENTION NETWORK (GAT) WITH DROPOUT
# =========================
class GATWithDropout(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, heads=1, dropout=0.5):
        super(GATWithDropout, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)


#####################
# GAT WITH BATCH NORM
#####################
class GATWithBatchNorm(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, heads=1, dropout=0.5):
        super(GATWithBatchNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)

#####################
# GCN WITH BATCH NORM
#####################

class ModifiedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1):
        super(ModifiedGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)  # Linear projection layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Graph Convolutional Layers with Batch Norm
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)  # Reduce dimensionality

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            residual = x  # Save for residual connection
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = x + residual  # Residual connection

        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))
