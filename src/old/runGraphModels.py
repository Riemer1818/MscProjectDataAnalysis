import argparse
import os
import json
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv
import matplotlib.pyplot as plt
import time


# # =========================
# # SYSTEM ARGUMENT PARSING
# # =========================

# def parse_arguments():
#     """Parse system arguments for training configuration."""
#     print("Parsing system arguments...")
#     parser = argparse.ArgumentParser(description='Train GCN with different configurations.')
    
#     parser.add_argument('--num_layers_options', nargs='+', type=int, default=[10, 20, 30],
#                         help='List of options for the number of layers.')
#     parser.add_argument('--hidden_channels_options', nargs='+', type=int, default=[16, 32, 64],
#                         help='List of options for hidden channels.')
#     parser.add_argument('--num_epochs', type=int, default=20,
#                         help='Number of training epochs.')
#     parser.add_argument('--learning_rate', type=float, default=0.001,
#                         help='Learning rate for the optimizer.')
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='Batch size for training.')
    
#     return parser.parse_args()


# # =========================
# # DATA LOADING FUNCTIONS
# # =========================

def load_and_create_edge_index(interaction_file):
    print(f"Loading interaction data from {interaction_file}")
    interaction_data = pd.read_csv(interaction_file, compression='gzip', sep=' ')
    interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
    interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')
    nodes = pd.concat([interaction_data['protein1'], interaction_data['protein2']]).unique()
    gene_to_index = {gene: idx for idx, gene in enumerate(nodes)}
    edges = [[gene_to_index[row['protein1']], gene_to_index[row['protein2']]] for _, row in interaction_data.iterrows()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(nodes)
    num_edges = edge_index.shape[1]
    edge_density = num_edges / (num_nodes * (num_nodes - 1))

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Edge density: {edge_density:.4f}")

    return edge_index, num_nodes

def get_num_samples_and_output_channels(file_path):
    print(f"Reading HDF5 file from {file_path}")
    with h5py.File(file_path, 'r') as f:
        num_samples = len(f.keys())
        sample_group = f['sample_1']
        output_channels = sample_group['Y'].shape[0]
    print(f"Number of samples: {num_samples}")
    print(f"Output channels: {output_channels}")
    return num_samples, output_channels

def load_sample_from_h5(file_path, sample_idx, edge_index):
    # print(f"Loading sample {sample_idx} from {file_path}")
    with h5py.File(file_path, 'r') as f:
        sample_group = f[f'sample_{sample_idx}']
        node_features = torch.tensor(sample_group['X'][:], dtype=torch.float32)
        motif_matrix = torch.tensor(sample_group['Y'][:], dtype=torch.float32)
    # print(f"Sample {sample_idx} loaded with {node_features.shape[0]} nodes and {node_features.shape[1]} features per node")
    return Data(x=node_features, edge_index=edge_index, y=motif_matrix)

def compute_normalization_stats(data_list):
    all_node_features = torch.cat([data.x for data in data_list], dim=0)  # Concatenate all features from all samples
    mean = all_node_features.mean(dim=0)
    std = all_node_features.std(dim=0)
    return mean, std

def normalize_node_features(data_list, mean, std):
    for data in data_list:
        data.x = (data.x - mean) / std  # Standardization (mean=0, std=1)
    return data_list

def normalize_motif_matrix(data_list, y_mean, y_std):
    for data in data_list:
        data.y = (data.y - y_mean) / y_std
    return data_list

# =========================
# GRAPH CONVOLUTIONAL NETWORK
# =========================


# =========================
# TRAINING AND EVALUATION
# =========================

def train_and_evaluate_model(model, dataloader, val_dataloader, num_epochs, learning_rate, T_max, device, checkpoint_dir, criterion):
    # Print out the model's architecture to verify its structure
    print(f"Using model: {model.__class__.__name__}")

    # Move the model to the device (e.g., GPU or CPU)
    model = model.to(device)

    # Setup optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    # criterion = torch.nn.MSELoss()

    checkpoint_dir = os.path.join(checkpoint_dir, model.__class__.__name__, criterion.__class__.__name__)
    # Prepare checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Track training and validation losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch

        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch)
            target = batch.y.view(batch.num_graphs, -1) if batch.y.dim() == 1 else batch.y
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_loss / len(dataloader)
        train_losses.append(avg_train_loss)

        if val_dataloader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for j, batch in enumerate(val_dataloader):
                    batch = batch.to(device)
                    output = model(batch)
                    target = batch.y.view(batch.num_graphs, -1) if batch.y.dim() == 1 else batch.y
                    val_loss += criterion(output, target).item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')

        end_time = time.time()  # End timing the epoch
        epoch_time = end_time - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Time Taken: {epoch_time:.2f} seconds')

        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Save the training and validation losses to a JSON file
    losses = {'train_losses': train_losses, 'val_losses': val_losses}
    losses_path = os.path.join(checkpoint_dir, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses, f)

    return train_losses, val_losses


# # =========================
# # MAIN EXECUTION
# # =========================

# args = parse_arguments()
# print("parsed arguments:", args)


batch_size = 16
INTERACTION_DATA_FILE = '9606.protein.physical.links.v12.0.txt.gz'
HDF5_GRAPH_FILE = "protein.physical.links.full_links.h5"
num_layers = 1
hidden_channels = 1
num_epochs = 15
learning_rate = 0.001
checkpoint_dir = 'checkpoints_diffModels'
T_max = 10
in_channels = 1

# Load data
edge_index, num_nodes = load_and_create_edge_index(INTERACTION_DATA_FILE)
num_samples, out_channels = get_num_samples_and_output_channels(HDF5_GRAPH_FILE)
data_list = [load_sample_from_h5(HDF5_GRAPH_FILE, idx, edge_index) for idx in range(1, num_samples + 1)]

# Compute normalization stats for node features X
mean_x, std_x = compute_normalization_stats(data_list)

# If needed, you can compute normalization stats for motif matrix Y in a similar way
all_motif_matrices = torch.cat([data.y for data in data_list], dim=0)
mean_y = all_motif_matrices.mean(dim=0)
std_y = all_motif_matrices.std(dim=0)

# Normalize both X and Y
data_list = normalize_node_features(data_list, mean_x, std_x)
data_list = normalize_motif_matrix(data_list, mean_y, std_y)

import random
from torch.utils.data import Subset

# ============================
# SPLIT DATASET AND USE SUBSET
# ============================

# Split dataset into training and validation
train_data, val_data = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)

# Define the proportion of the dataset you want to use for exploration (e.g., 10%)
train_subset_size = int(0.1 * len(train_data))  # Use 10% of the training data
val_subset_size = int(0.1 * len(val_data))      # Use 10% of the validation data

# Randomly sample a subset from train_data and val_data
train_indices = random.sample(range(len(train_data)), train_subset_size)
val_indices = random.sample(range(len(val_data)), val_subset_size)

# Use the Subset class to create smaller datasets
train_subset = Subset(train_data, train_indices)
val_subset = Subset(val_data, val_indices)

# Create DataLoader using the smaller subset of data
dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Print the number of samples used for training and validation
print(f"Training with {len(train_subset)} samples and validating with {len(val_subset)} samples")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = data_list[0].x.shape[1]
print(f"Number of input channels: {in_channels}")
print(f"Using device: {device}")

# # GAT model
# print("Training GAT model...")
# gat_model = GAT(in_channels, hidden_channels, out_channels, num_layers=num_layers, heads=4)
# train_losses, val_losses = train_and_evaluate_model(
#     model=gat_model,
#     dataloader=dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     T_max=T_max,
#     device=device,
#     checkpoint_dir=checkpoint_dir
# )

# # GIN model
# print("Training GIN model...")
# gin_model = GIN(in_channels, hidden_channels, out_channels, num_layers=num_layers)
# train_losses, val_losses = train_and_evaluate_model(
#     model=gin_model,
#     dataloader=dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     T_max=T_max,
#     device=device,
#     checkpoint_dir=checkpoint_dir
# )

# # Create a GCN model
# print("Training GCN model...")
# gcn_model = GCN(in_channels, hidden_channels, out_channels, num_layers)
# train_losses, val_losses = train_and_evaluate_model(
#     model=gcn_model,
#     dataloader=dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     T_max=T_max,
#     device=device,
#     checkpoint_dir=checkpoint_dir
# )

# GAT with dropout
# print("Training GAT with dropout model...")
# gat_dropout_model = GATWithDropout(in_channels, hidden_channels, out_channels, num_layers=num_layers, heads=4, dropout=0.5)
# train_losses, val_losses = train_and_evaluate_model(
#     model=gat_dropout_model,
#     dataloader=dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     T_max=T_max,
#     device=device,
#     checkpoint_dir=checkpoint_dir
# )


# print("Training GAT with batch norm model...")
# gat_batch_norm_modlel = GATWithBatchNorm(in_channels, hidden_channels, out_channels, num_layers=num_layers, heads=4, dropout=0.5)
# train_losses, val_losses = train_and_evaluate_model(
#     model=gat_batch_norm_modlel,
#     dataloader=dataloader,
#     val_dataloader=val_dataloader,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     T_max=T_max,
#     device=device,
#     checkpoint_dir=checkpoint_dir
# )

# Create a GCN model
print("Training GCN model with BCE loss...")
gcn_model = GCN(in_channels, hidden_channels, out_channels, num_layers)
train_losses, val_losses = train_and_evaluate_model(
    model=gcn_model,
    dataloader=dataloader,
    val_dataloader=val_dataloader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    T_max=T_max,
    device=device,
    checkpoint_dir=checkpoint_dir,
    criterion=torch.nn.BCELoss()
)

# Create a GCN model
print("Training GCN model with L1 loss...")
gcn_model = GCN(in_channels, hidden_channels, out_channels, num_layers)
train_losses, val_losses = train_and_evaluate_model(
    model=gcn_model,
    dataloader=dataloader,
    val_dataloader=val_dataloader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    T_max=T_max,
    device=device,
    checkpoint_dir=checkpoint_dir,
    criterion=torch.nn.L1Loss()
)

