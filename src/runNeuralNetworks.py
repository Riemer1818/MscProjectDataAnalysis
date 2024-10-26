import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.nn as nn
import torch.optim as optim
import os
import json

class H5Dataset(Dataset):
    def __init__(self, gene_matrix_path, motif_matrix_path, transform=None):
        super().__init__()
        self.gene_matrix_path = gene_matrix_path
        self.motif_matrix_path = motif_matrix_path
        self.transform = transform

        # Load the data from HDF5 files
        with h5py.File(self.gene_matrix_path, 'r') as f:
            self.gene_matrix = f[list(f.keys())[0]][:]  # Load the gene matrix dataset
            print(f"Gene matrix shape: {self.gene_matrix.shape}")

        with h5py.File(self.motif_matrix_path, 'r') as f:
            self.motif_matrix = f[list(f.keys())[0]][:]  # Load the motif matrix dataset
            print(f"Motif matrix shape: {self.motif_matrix.shape}")

        # Calculate mean and standard deviation along the correct axis (axis=1 for features)
        self.gene_mean = self.gene_matrix.mean(axis=1)  # Shape (4500,)
        self.gene_std = self.gene_matrix.std(axis=1)    # Shape (4500,)
        self.motif_mean = self.motif_matrix.mean(axis=1)  # Shape (541,)
        self.motif_std = self.motif_matrix.std(axis=1)    # Shape (541,)


    def __len__(self):
        # The number of samples is the first dimension of the gene matrix
        return self.gene_matrix.shape[0]  # 47936 samples

    def __getitem__(self, idx):
        # Retrieve the feature vectors for the idx-th sample along axis=1
        gene_features = self.gene_matrix[:, idx]  # Shape (4500,) - Features for the idx-th sample
        motif_features = self.motif_matrix[:, idx]  # Shape (541,) - Features for the idx-th sample

        # Normalize the gene features across samples (subtract mean and divide by std for each feature)
        gene_features = (gene_features - self.gene_mean) / (self.gene_std + 1e-8)  # Shape (4500,)
        
        # Normalize the motif features similarly, but note the shape is different (541 instead of 4500)
        motif_features = (motif_features - self.motif_mean) / (self.motif_std + 1e-8)  # Shape (541,)

        # Convert to PyTorch tensors
        gene_features = torch.tensor(gene_features, dtype=torch.float32)
        motif_features = torch.tensor(motif_features, dtype=torch.float32)

        return gene_features, motif_features



# Define a function to create a DataLoader with normalization
def create_dataloader(gene_matrix_path, motif_matrix_path, batch_size, shuffle=True):
    dataset = H5Dataset(gene_matrix_path, motif_matrix_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Define a base class for the models
class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

# Define the baseline model (simple fully connected neural network)
class BaselineNN(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(BaselineNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a model with dropout
class DropoutNN(BaseModel):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(DropoutNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a model with skip connections
class SkipConnectionNN(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(SkipConnectionNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Output layer
        
        # Add a linear layer to project the input to the same dimension (64)
        self.residual_projection = nn.Linear(input_dim, 64)

    def forward(self, x):
        # Project the residual to match the output of fc2
        residual = self.residual_projection(x)  # Shape (batch_size, 64)
        
        # Forward pass through the main branch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Add the skip connection
        x = x + residual  # Both tensors now have shape (batch_size, 64)
        
        # Final output layer
        x = self.fc3(x)  # Shape (batch_size, output_dim), where output_dim = 541
        return x

class DropoutSkipConnectionNN(BaseModel):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(DropoutSkipConnectionNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Final output layer
        
        # Add a projection layer to make residual compatible with the main branch (64 features)
        self.residual_projection = nn.Linear(input_dim, 64)

    def forward(self, x):
        # Project the input for the skip connection
        residual = self.residual_projection(x)  # Shape (batch_size, 64)
        
        # Forward pass through the main branch
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        
        # Add the skip connection
        x = x + residual  # Both tensors now have shape (batch_size, 64)
        
        # Final output layer
        x = self.fc3(x)  # Shape (batch_size, output_dim)
        return x

# Training function with loss tracking and saving
def train_model(model, train_loader, val_loader, epochs, learning_rate, device, model_name, output_dir):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model = model.to(device)
    
    train_losses = []
    val_losses = []
    
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
        
        # Validation loop
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save losses to JSON
    save_losses_to_json(train_losses, val_losses, model_name, output_dir)

# Evaluation function
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

# Function to save losses to a JSON file
def save_losses_to_json(train_losses, val_losses, model_name, output_dir):
    losses = {'train_losses': train_losses, 'val_losses': val_losses}
    os.makedirs(output_dir, exist_ok=True)
    losses_path = os.path.join(output_dir, f'{model_name}_losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses, f)
    print(f"Losses saved to {losses_path}")

# Main execution: define training parameters and run the training loop for each model
def main():
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    input_dim = 4500  # Number of input features (genes)
    output_dim = 541  # Number of output features (motifs)
    output_dir = 'model_losses'  # Directory to save the loss files

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset using the DataLoader defined earlier
    gene_matrix_path = 'final_gene_matrix.h5'
    motif_matrix_path = 'final_motif_matrix.h5'
    train_loader = create_dataloader(gene_matrix_path, motif_matrix_path, batch_size, shuffle=True)
    val_loader = create_dataloader(gene_matrix_path, motif_matrix_path, batch_size, shuffle=False)  # For simplicity using the same data, change for actual validation set

    # Test all models fairly by training them for the same number of epochs with the same parameters
    models = {
        'BaselineNN': BaselineNN(input_dim, output_dim),
        'DropoutNN': DropoutNN(input_dim, output_dim, dropout_prob=0.5),
        'SkipConnectionNN': SkipConnectionNN(input_dim, output_dim),
        'DropoutSkipConnectionNN': DropoutSkipConnectionNN(input_dim, output_dim, dropout_prob=0.5)
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        train_model(model, train_loader, val_loader, epochs, learning_rate, device, model_name, output_dir)
        print(f"Finished training {model_name}.\n")

if __name__ == '__main__':
    main()
