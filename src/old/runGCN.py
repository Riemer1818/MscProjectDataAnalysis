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
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, global_add_pool
import matplotlib.pyplot as plt
import time

# # =========================
# # SYSTEM ARGUMENT PARSING
# # =========================

def parse_arguments():
    """Parse system arguments for training configuration."""
    print("Parsing system arguments...")
    parser = argparse.ArgumentParser(description='Train GCN with different configurations.')
    
    parser.add_argument('--num_layers_options', nargs='+', type=int, default=[10, 20, 30],
                        help='List of options for the number of layers.')
    parser.add_argument('--hidden_channels_options', nargs='+', type=int, default=[16, 32, 64],
                        help='List of options for hidden channels.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    
    return parser.parse_args()


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


# # =========================
# # GRAPH CONVOLUTIONAL NETWORK
# # =========================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1):
        super(GCN, self).__init__()
        # print(f"Initializing GCN with {num_layers} layers and {hidden_channels} hidden channels")
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            # print(f"Layer {i+1}: x has shape {x.shape}")
        x = global_mean_pool(x, batch)
        # print(f"After global mean pool: x has shape {x.shape}")
        return self.fc(x)


# # =========================
# # TRAINING AND EVALUATION
# # =========================

def train_and_evaluate_model(in_channels, hidden_channels, out_channels, num_layers, dataloader, val_dataloader, num_epochs, learning_rate, T_max, device, checkpoint_dir):
    print(f"Initializing model with {num_layers} layers and {hidden_channels} hidden channels")
    
    # Create the model
    model = GCN(in_channels, hidden_channels, out_channels, num_layers).to(device)
    
    # Setup optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    criterion = torch.nn.MSELoss()
    
    # Prepare checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine if a checkpoint exists based on epoch-based naming
    last_checkpoint = None
    last_epoch = 0

    # Look for checkpoints in the directory to resume from
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
    if checkpoint_files:
        checkpoint_epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
        last_epoch = max(checkpoint_epochs)
        last_checkpoint = os.path.join(checkpoint_dir, f'model_epoch_{last_epoch}.pt')
        print(f"Found checkpoint at epoch {last_epoch}. Loading from {last_checkpoint}...")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded. Resuming training from epoch {last_epoch + 1}")

    # Track training and validation losses
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(last_epoch, num_epochs):
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

    losses = {'train_losses': train_losses, 'val_losses': val_losses}
    losses_path = os.path.join(checkpoint_dir, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses, f)
    
    return train_losses, val_losses

# # =========================
# # MAIN EXECUTION
# # =========================

args = parse_arguments()
print("parsed arguments:", args)

INTERACTION_DATA_FILE = '9606.protein.physical.links.v12.0.txt.gz'
HDF5_GRAPH_FILE = "protein.physical.links.full_links.h5"

# Load data
edge_index, num_nodes = load_and_create_edge_index(INTERACTION_DATA_FILE)
num_samples, out_channels = get_num_samples_and_output_channels(HDF5_GRAPH_FILE)
data_list = [load_sample_from_h5(HDF5_GRAPH_FILE, idx, edge_index) for idx in range(1, num_samples + 1)]

# Split dataset
train_data, val_data = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)
dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

print(f"Training with {len(train_data)} samples and validating with {len(val_data)} samples")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = data_list[0].x.shape[1]
print(f"Number of input channels: {in_channels}")
print(f"Using device: {device}")

num_layers_options = [10]
hidden_channels_options = [16]

# Training loop for different configurations
for num_layers in num_layers_options:
    for hidden_channels in hidden_channels_options:
        print(f"\nTraining with {num_layers} layers and {hidden_channels} hidden units...")
        train_losses, val_losses = train_and_evaluate_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            T_max=10,
            device=device,
            checkpoint_dir=f'checkpoints_{num_layers}_{hidden_channels}'
        )
        # Plotting
        plt.plot(train_losses, label=f'Train Loss (Layers={num_layers}, Hidden={hidden_channels})')
        if val_losses:
            plt.plot(val_losses, label=f'Val Loss (Layers={num_layers}, Hidden={hidden_channels})')
