import h5py
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os
import json
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
import argparse
# =========================
# File Paths and Constants
# =========================

# HDF5_GRAPH_FILE = 'combined_graph_data_full_links.h5'
# INTERACTION_DATA_FILE = '9606.protein.physical.links.v12.0.txt.gz'
# ENSEMBL_GENE_FILE = 'ensembl_protein_ids_cleaned.csv'


HDF5_GRAPH_FILE = '100_0001_50_protein.physical.links.h5'
INTERACTION_DATA_FILE = '9606.protein.physical.links.v12.0.txt.gz'
ENSEMBL_GENE_FILE = 'ensembl_protein_ids_cleaned.csv'


def parse_arguments():
    """Parse system arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train GCN with different configurations.')
    
    parser.add_argument('--num_layers_options', nargs='+', type=int, default=[10, 20, 30],
                        help='List of options for the number of layers.')
    parser.add_argument('--hidden_channels_options', nargs='+', type=int, default=[16, 32, 64],
                        help='List of options for hidden channels.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')

    return parser.parse_args()
# =========================
# Data Loading Functions
# =========================

def load_and_create_edge_index(interaction_file):
    """Load interaction data and create an edge index."""
    interaction_data = pd.read_csv(interaction_file, compression='gzip', sep=' ')
    interaction_data.replace({'9606.': ''}, regex=True, inplace=True)

    nodes = pd.concat([interaction_data['protein1'], interaction_data['protein2']]).unique()
    gene_to_index = {gene: idx for idx, gene in enumerate(nodes)}
    edges = [[gene_to_index[row['protein1']], gene_to_index[row['protein2']]] for _, row in interaction_data.iterrows()]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Print stats
    print_stats(len(nodes), edge_index.shape[1])
    
    return edge_index, len(nodes)

def print_stats(num_nodes, num_edges):
    edge_density = num_edges / (num_nodes * (num_nodes - 1))
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Edge density: {edge_density:.4f}")

def get_num_samples_and_output_channels(file_path):
    """Retrieve the number of samples and output channels from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        sample_group = f['sample_1']
        return len(f.keys()), sample_group['Y'].shape[0]

def load_sample_from_h5(file_path, sample_idx, edge_index):
    """Load a sample from HDF5 and return a Data object."""
    with h5py.File(file_path, 'r') as f:
        sample_group = f[f'sample_{sample_idx}']
        node_features = torch.tensor(sample_group['X'][:], dtype=torch.float32)
        motif_matrix = torch.tensor(sample_group['Y'][:], dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, y=motif_matrix)

# =========================
# Graph Neural Network
# =========================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print(f"x shape: {x.shape}")
        print(f"edge_index shape: {edge_index.shape}")
        print(f"batch shape: {batch.shape}")
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        return self.fc(x)

# =========================
# Training and Evaluation
# =========================

def train_and_evaluate_model(model_params, dataloader, val_dataloader=None, num_epochs=10, learning_rate=0.01):
    model = GCN(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    save_dir = f'checkpoints_layers_{model_params["num_layers"]}_hidden_{model_params["hidden_channels"]}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = run_epoch(model, dataloader, criterion, optimizer, is_train=True)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

        if val_dataloader:
            avg_val_loss = run_epoch(model, val_dataloader, criterion, is_train=False)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved at {checkpoint_path}')

    save_losses(train_losses, val_losses, save_dir)
    return train_losses, val_losses

def run_epoch(model, dataloader, criterion, optimizer=None, is_train=True):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        batch_size = batch.num_graphs
        batch.y = batch.y.view(batch_size, -1)
        output = model(batch)

        loss = criterion(output, batch.y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def save_losses(train_losses, val_losses, save_dir):
    losses = {'train_losses': train_losses, 'val_losses': val_losses}
    with open(os.path.join(save_dir, 'losses.json'), 'w') as f:
        json.dump(losses, f)


# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    args = parse_arguments()

    # Load edge index and node information
    edge_index, num_nodes = load_and_create_edge_index(INTERACTION_DATA_FILE)
    NUM_SAMPLES, out_channels = get_num_samples_and_output_channels(HDF5_GRAPH_FILE)

    # Load data and split
    data_list = [load_sample_from_h5(HDF5_GRAPH_FILE, idx, edge_index) for idx in range(1, NUM_SAMPLES + 1)]
    train_data, val_data = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)
    dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    in_channels = data_list[0].x.shape[0]  # Number of features per node

    # Training loop
    for num_layers in args.num_layers_options:
        for hidden_channels in args.hidden_channels_options:
            print(f"\nTraining with {num_layers} layers and {hidden_channels} hidden units...")
            model_params = {
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'out_channels': out_channels,
                'num_layers': num_layers
            }
            train_losses, val_losses = train_and_evaluate_model(
                model_params, dataloader, val_dataloader, num_epochs=args.num_epochs, learning_rate=args.learning_rate)

            plt.plot(train_losses, label=f'Train (Layers={num_layers}, Hidden={hidden_channels})')
            if val_losses:
                plt.plot(val_losses, label=f'Val (Layers={num_layers}, Hidden={hidden_channels})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Time for Different Architectures')
    plt.legend()
    plt.show()