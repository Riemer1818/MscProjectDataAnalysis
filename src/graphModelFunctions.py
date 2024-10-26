# graphModelFunctions.py

import argparse
import os
import json
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import time
from graphModels import GCN, GAT, GATWithDropout, GATWithBatchNorm, GIN

# =========================
# ARGUMENT PARSING FUNCTIONS
# =========================

def parse_arguments():
    """Parse system arguments for training configuration."""
    print("Parsing system arguments...")
    parser = argparse.ArgumentParser(description='Train GCN with different configurations.')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model to use for training (GCN, GAT, GATWithDropout, GATWithBatchNorm, GIN).')
    parser.add_argument('--num_layers', nargs='+', type=int, default=10,
                        help='Number of layers.')
    parser.add_argument('--hidden_channels', nargs='+', type=int, default=8,
                        help='hHidden channels.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    
    return parser.parse_args()


# ============================
# MODEL INITIALIZATION
# ============================

def initialize_model(model_name, in_channels, hidden_channels, out_channels, num_layers, heads=1, dropout=0.5):
    
    """Initialize model based on model name."""
    if model_name == "GCN":
        return GCN(in_channels, hidden_channels, out_channels, num_layers)
    elif model_name == "GAT":
        return GAT(in_channels, hidden_channels, out_channels, num_layers, heads=1)
    elif model_name == "GATWithDropout":
        return GATWithDropout(in_channels, hidden_channels, out_channels, num_layers, heads, dropout)
    elif model_name == "GATWithBatchNorm":
        return GATWithBatchNorm(in_channels, hidden_channels, out_channels, num_layers, heads, dropout)
    elif model_name == "GIN":
        return GIN(in_channels, hidden_channels, out_channels, num_layers)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def setup_optimizer_and_scheduler(model, learning_rate=0.001, T_max=10):
    """Set up optimizer and scheduler for further training if needed."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return optimizer, scheduler

# =========================
# DATA LOADING FUNCTIONS
# =========================

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
    with h5py.File(file_path, 'r') as f:
        sample_group = f[f'sample_{sample_idx}']
        node_features = torch.tensor(sample_group['X'][:], dtype=torch.float32)
        motif_matrix = torch.tensor(sample_group['Y'][:], dtype=torch.float32)
    return Data(x=node_features, edge_index=edge_index, y=motif_matrix)

def compute_normalization_stats(data_list):
    all_node_features = torch.cat([data.x for data in data_list], dim=0)
    mean = all_node_features.mean(dim=0)
    std = all_node_features.std(dim=0)
    return mean, std

def normalize_node_features(data_list, mean, std):
    for data in data_list:
        data.x = (data.x - mean) / std
        # Replace NaN and Inf values with 0
        data.x[torch.isnan(data.x)] = 0
        data.x[torch.isinf(data.x)] = 0
    return data_list

def normalize_motif_matrix(data_list):
    """Normalize target matrix by scaling values to range [0, 1].

    Returns:
        data_list (list): List of normalized data samples.
        baseline (float): Baseline value representing "no change" (fixed at 0.5).
    """
    baseline = 0.5  # Fixed baseline for [0, 1] range scaling
    for data in data_list:
        data.y = (data.y - data.y.min()) / (data.y.max() - data.y.min())
        # Replace NaN and Inf values with 0
        data.y[torch.isnan(data.y)] = 0
        data.y[torch.isinf(data.y)] = 0
    return data_list, baseline


# =========================
# TRAINING AND EVALUATION
# =========================

def train_and_evaluate_model(model, dataloader, val_dataloader, num_epochs, learning_rate, T_max, device, checkpoint_dir, criterion):
    print(f"Using model: {model.__class__.__name__}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    checkpoint_dir = os.path.join(checkpoint_dir, model.__class__.__name__, criterion.__class__.__name__)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch)
            
            # Flatten target values if necessary
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
                for batch in val_dataloader:
                    batch = batch.to(device)
                    output = model(batch)
                    
                    # Flatten target values if necessary
                    target = batch.y.view(batch.num_graphs, -1) if batch.y.dim() == 1 else batch.y
                    val_loss += criterion(output, target).item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')

        end_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Time Taken: {end_time - start_time:.2f} seconds')
        
        # Calculate and print top-k accuracy
        
        accuracy = evaluate_top_k_accuracy(output, target, k=10, baseline=0.5)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Top-10 Accuracy: {accuracy:.4f}")

        # Initialize the checkpoint data dictionary
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # Include model architecture parameters in the checkpoint
            'model_config': {
                'in_channels': model.in_channels,
                'hidden_channels': model.hidden_channels,
                'out_channels': model.out_channels,
                'num_layers': model.num_layers,
            }
        }

        if hasattr(model, 'heads'):
            checkpoint_data['model_config']['heads'] = model.heads
        if hasattr(model, 'dropout'):
            checkpoint_data['model_config']['dropout'] = model.dropout

            # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    losses_path = os.path.join(checkpoint_dir, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'accuracies' : accuracies}, f)

    return train_losses, val_losses, accuracies


def evaluate_top_k_accuracy(model_output, target, k=10, baseline=1.0):
    """
    Evaluate whether the model's top-k predictions match the top-k values in the target
    based on the distance from a given baseline (default is 1 for no normalization).
    
    Args:
        model_output (torch.Tensor): The output predictions from the model (e.g., shape [541]).
        target (torch.Tensor): The true motif matrix target values (e.g., shape [541]).
        k (int): The number of top values to compare (default is 10).
        baseline (float): The baseline (centroid) value indicating "no change" (default 1.0).
        
    Returns:
        accuracy (float): Proportion of top-k matches between the model output and target.
    """
    # Ensure model_output and target are flattened to 1D
    model_output = model_output.flatten()
    target = target.flatten()

    # Calculate the absolute distance from the baseline for both model output and target
    model_distance_from_baseline = torch.abs(model_output - baseline)
    target_distance_from_baseline = torch.abs(target - baseline)
    
    # Get indices of the top-k furthest values in the model output and target based on distance from baseline
    _, top_k_pred_indices = torch.topk(model_distance_from_baseline, k)
    _, top_k_target_indices = torch.topk(target_distance_from_baseline, k)
    
    # Convert indices to sets to find matches
    top_k_pred_indices = set(top_k_pred_indices.tolist())
    top_k_target_indices = set(top_k_target_indices.tolist())
    
    # Calculate the intersection of the indices to see how many match
    matches = len(top_k_pred_indices & top_k_target_indices)
    accuracy = matches / k  # Calculate the accuracy as a proportion of k
    
    print(f"Top-{k} accuracy based on distance from {baseline}: {accuracy * 100:.2f}%")
    return accuracy


def load_model_from_checkpoint(checkpoint_dir, checkpoint_file, model, optimizer=None, scheduler=None):
    """
    Load model and optimizer states from a checkpoint file.
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with checkpoint '{checkpoint_path}'")
    return model, optimizer, scheduler

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
