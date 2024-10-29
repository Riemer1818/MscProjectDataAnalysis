# neuralNetworkFunctions.py

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import json
import argparse
import torch.optim as optim
import torch.nn as nn

# =========================
# ARGUMENT PARSING FUNCTIONS
# =========================

import argparse

import argparse

def parse_arguments():
    """Parse system arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train different models.')
    parser.add_argument('--model', type=str, default='BaselineNN',
                        help='Model to use for training (BaselineNN, DropoutNN, SkipConnectionNN, DropoutSkipConnectionNN).')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to resume training from a checkpoint (optional).')
    parser.add_argument('--loss_function', type=str, default='BCE',
                        choices=['BCE', 'MSE', "CrossEntropy"],
                        help='Loss function to use (BCE for Binary Cross-Entropy, MSE for Mean Squared Error).')
    return parser.parse_args()


# ============================
# DATASET INITIALIZATION
# ============================

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class H5Dataset(Dataset):
    def __init__(self, gene_matrix_path, motif_matrix_path, normalize_motif=True, transform=None):
        super().__init__()
        self.gene_matrix_path = gene_matrix_path
        self.motif_matrix_path = motif_matrix_path
        self.transform = transform
        self.normalize_motif = normalize_motif

        with h5py.File(self.gene_matrix_path, 'r') as f:
            self.gene_matrix = f[list(f.keys())[0]][:].T  # Transpose the matrix
        with h5py.File(self.motif_matrix_path, 'r') as f:
            self.motif_matrix = f[list(f.keys())[0]][:].T  # Transpose the matrix

        # Calculate mean and standard deviation for normalization
        self.gene_mean = self.gene_matrix.mean(axis=0)
        self.gene_std = self.gene_matrix.std(axis=0)

        # Normalize the motif matrix if specified
        if self.normalize_motif:
            self._normalize_motif_matrix()

    def _normalize_motif_matrix(self):
        """Normalize the motif matrix to the range [0, 1]."""
        for i in range(self.motif_matrix.shape[1]):
            data_min = self.motif_matrix[:, i].min()
            data_max = self.motif_matrix[:, i].max()
            if data_max > data_min:  # Avoid division by zero
                self.motif_matrix[:, i] = (self.motif_matrix[:, i] - data_min) / (data_max - data_min)
            else:
                self.motif_matrix[:, i] = 0  # If max equals min, set all values to zero

            # Convert to a PyTorch tensor before replacing NaN and Inf values
            motif_tensor = torch.tensor(self.motif_matrix[:, i], dtype=torch.float32)
            motif_tensor[torch.isnan(motif_tensor)] = 0
            motif_tensor[torch.isinf(motif_tensor)] = 0

            # Update the motif matrix with the processed tensor values
            self.motif_matrix[:, i] = motif_tensor.numpy()

    def __len__(self):
        return self.gene_matrix.shape[0]  # Should be 2996 after transposition

    def __getitem__(self, idx):
        # Normalize gene features
        gene_features = (self.gene_matrix[idx, :] - self.gene_mean) / (self.gene_std + 1e-8)
        motif_features = self.motif_matrix[idx, :]

        # Convert to PyTorch tensors
        gene_features = torch.tensor(gene_features, dtype=torch.float32)
        motif_features = torch.tensor(motif_features, dtype=torch.float32)

        return gene_features, motif_features


def create_dataloader(gene_matrix_path, motif_matrix_path, batch_size, shuffle=True, normalize_motif=True):
    dataset = H5Dataset(gene_matrix_path, motif_matrix_path, normalize_motif=normalize_motif)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ============================
# MODEL TRAINING
# ============================

def train_model(model, train_loader, val_loader, epochs, learning_rate, device, model_name, output_dir, criterion, checkpoint_dir='checkpoints'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Example scheduler, you can customize

    model = model.to(device)
    
    train_losses, val_losses, accuracies= [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for gene_batch, motif_batch in train_loader:
            gene_batch, motif_batch = gene_batch.to(device), motif_batch.to(device)
            optimizer.zero_grad()
            output = model(gene_batch)
            loss = criterion(output, motif_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        accuracy = evaluate_top_k_accuracy(output, motif_batch, k=10, baseline=0.5)
        
        accuracies.append(accuracy)

        # Update the learning rate
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Initialize the checkpoint data dictionary
            checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config': {
                'in_channels': model.in_channels if hasattr(model, 'in_channels') else None,
                'hidden_channels': model.hidden_channels if hasattr(model, 'hidden_channels') else None,
                'out_channels': model.out_channels if hasattr(model, 'out_channels') else None,
                'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
            }
            }

            # Include optional model attributes if they exist
            if hasattr(model, 'heads'):
                checkpoint_data['model_config']['heads'] = model.heads
            if hasattr(model, 'dropout'):
                checkpoint_data['model_config']['dropout'] = model.dropout

            # Save checkpoint
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pt')
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save losses to JSON
        save_losses_to_json(train_losses, val_losses, model_name, output_dir)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for gene_batch, motif_batch in val_loader:
            gene_batch, motif_batch = gene_batch.to(device), motif_batch.to(device)
            output = model(gene_batch)
            loss = criterion(output, motif_batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def save_losses_to_json(train_losses, val_losses, model_name, output_dir):
    losses = {'train_losses': train_losses, 'val_losses': val_losses}
    os.makedirs(output_dir, exist_ok=True)
    losses_path = os.path.join(output_dir, f'{model_name}_losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses, f)
    print(f"Losses saved to {losses_path}")

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
