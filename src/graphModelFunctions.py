# graphModelFunctions.py

import argparse
import os
import json
import h5py
import torch
from torch_geometric.data import Data
import pandas as pd
import time
from graphModels import GCN, GAT, GATWithDropout, GATWithBatchNorm, ModifiedGCN

# =========================
# ARGUMENT PARSING FUNCTIONS
# =========================

def parse_arguments():
    """Parse system arguments for training configuration."""
    print("Parsing system arguments...")
    parser = argparse.ArgumentParser(description='Train GCN with different configurations.')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model to use for training (GCN, GAT, GATWithDropout, GATWithBatchNorm, ModifiedGCN).')
    parser.add_argument('--num_layers', nargs='+', type=int, default=2,
                        help='Number of layers.')
    parser.add_argument('--hidden_channels', nargs='+', type=int, default=128,
                        help='Hidden channels.')
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
    elif model_name == "ModifiedGCN":
        return ModifiedGCN(in_channels, hidden_channels, out_channels, num_layers)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

# =========================
# DATA LOADING FUNCTIONS
# =========================

def load_full_interaction_graph(interaction_data_file):
    # Load interaction data
    interaction_data = pd.read_csv(interaction_data_file, compression='gzip', sep=' ')
    interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
    interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

    # Create a unique mapping for nodes
    nodes = pd.concat([interaction_data['protein1'], interaction_data['protein2']]).unique()
    gene_to_index = {gene: idx for idx, gene in enumerate(nodes)}

    # Convert edges to index-based format
    edges = [
        [gene_to_index[row['protein1']], gene_to_index[row['protein2']]]
        for _, row in interaction_data.iterrows()
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to('cuda')

    return edge_index, gene_to_index


def get_num_samples_and_output_channels(file_path):
    with h5py.File(file_path, 'r') as f:
        if len(f) == 0:
            raise ValueError(f"The HDF5 file at {file_path} contains no groups.")
        
        num_samples = len(f)
        first_sample = list(f.keys())[0]
        if 'Y' not in f[first_sample]:
            raise KeyError(f"The group '{first_sample}' does not contain a 'Y' dataset.")
        
        output_channels = f[first_sample]['Y'].shape[0]
    return num_samples, output_channels

def subset_edge_index_for_sample(full_edge_index, sample_nodes):
    # Move full_edge_index to CPU if it's on the GPU
    full_edge_index = full_edge_index.cpu()

    # Create a mask for edges where both nodes are in `sample_nodes`
    node_mask = torch.zeros(full_edge_index.max() + 1, dtype=torch.bool)
    node_mask[sample_nodes] = True
    mask = node_mask[full_edge_index[0]] & node_mask[full_edge_index[1]]
    return full_edge_index[:, mask].to('cuda')  # Move the subsetted edge_index to GPU

def load_sample_from_h5(file_path, sample_idx, full_edge_index, gene_to_index):
    with h5py.File(file_path, 'r') as f:
        sample_group = f[f'sample_{sample_idx}']
        node_features = torch.tensor(sample_group['X'][:], dtype=torch.float32).to('cuda')
        motif_matrix = torch.tensor(sample_group['Y'][:], dtype=torch.float32).to('cuda')

        # Determine which nodes are present in this sample based on `X` shape
        num_nodes_in_sample = node_features.shape[0]
        sample_nodes = list(gene_to_index.values())[:num_nodes_in_sample]
        sample_nodes = torch.tensor(sample_nodes, dtype=torch.long)  # Keep sample_nodes on CPU

        # Subset the full edge_index based on sample_nodes
        edge_index = subset_edge_index_for_sample(full_edge_index, sample_nodes)

    return Data(x=node_features, edge_index=edge_index, y=motif_matrix)


# =========================
# NORMALIZATION FUNCTIONS
# =========================

def compute_normalization_stats(data_list):
    all_node_features = torch.cat([data.x for data in data_list], dim=0)
    mean = all_node_features.mean(dim=0)
    std = all_node_features.std(dim=0)
    return mean, std

def normalize_node_features(data_list, mean, std):
    for data in data_list:
        data.x = (data.x - mean) / std
        data.x[torch.isnan(data.x)] = 0
        data.x[torch.isinf(data.x)] = 0
    return data_list

def normalize_motif_matrix(data_list):
    baseline = 0.5  # Fixed baseline for [0, 1] range scaling
    for data in data_list:
        data.y = (data.y - data.y.min()) / (data.y.max() - data.y.min())
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
            target = batch.y

            if criterion.__class__.__name__ == "CrossEntropyLoss":
                output = output.view(-1, output.size(-1))
                target = target.view(-1).long()
            elif criterion.__class__.__name__ == "MSELoss":
                target = target.view(output.shape)

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
                    target = batch.y

                    if criterion.__class__.__name__ == "CrossEntropyLoss":
                        output = output.view(-1, output.size(-1))
                        target = target.view(-1).long()
                    elif criterion.__class__.__name__ == "MSELoss":
                        target = target.view(output.shape)
                    
                    val_loss += criterion(output, target).item()
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')

        end_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Time Taken: {end_time - start_time:.2f} seconds')
        
        accuracy = evaluate_top_k_accuracy(output, target, k=10, baseline=0.5)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Top-10 Accuracy: {accuracy:.4f}")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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

        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    losses_path = os.path.join(checkpoint_dir, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'accuracies' : accuracies}, f)

    return train_losses, val_losses, accuracies

def evaluate_top_k_accuracy(model_output, target, k=10, baseline=1.0):
    model_output = model_output.flatten()
    target = target.flatten()

    model_distance_from_baseline = torch.abs(model_output - baseline)
    target_distance_from_baseline = torch.abs(target - baseline)
    
    _, top_k_pred_indices = torch.topk(model_distance_from_baseline, k)
    _, top_k_target_indices = torch.topk(target_distance_from_baseline, k)
    
    top_k_pred_indices = set(top_k_pred_indices.tolist())
    top_k_target_indices = set(top_k_target_indices.tolist())
    
    matches = len(top_k_pred_indices & top_k_target_indices)
    accuracy = matches / k
    
    print(f"Top-{k} accuracy based on distance from {baseline}: {accuracy * 100:.2f}%")
    return accuracy

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
