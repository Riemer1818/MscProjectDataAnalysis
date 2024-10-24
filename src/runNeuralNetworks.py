import torch
from torch.utils.data import Dataset, DataLoader
import h5py

# Define a custom dataset for loading and normalizing the HDF5 data
class H5Dataset(Dataset):
    def __init__(self, gene_matrix_path, motif_matrix_path, transform=None):
        super().__init__()
        self.gene_matrix_path = gene_matrix_path
        self.motif_matrix_path = motif_matrix_path
        self.transform = transform

        # Load the data from HDF5 files
        with h5py.File(self.gene_matrix_path, 'r') as f:
            self.gene_matrix = f[list(f.keys())[0]][:]  # Load the gene matrix dataset
        with h5py.File(self.motif_matrix_path, 'r') as f:
            self.motif_matrix = f[list(f.keys())[0]][:]  # Load the motif matrix dataset

        # Check that the number of output features matches in both matrices
        assert self.gene_matrix.shape[1] == self.motif_matrix.shape[1], \
            "Mismatch in number of output features between gene and motif matrices."

        # Calculate mean and standard deviation along the feature dimension (axis=0)
        self.gene_mean = self.gene_matrix.mean(axis=0)
        self.gene_std = self.gene_matrix.std(axis=0)
        self.motif_mean = self.motif_matrix.mean(axis=0)
        self.motif_std = self.motif_matrix.std(axis=0)

    def __len__(self):
        # The number of samples (output features) is the second dimension of the gene matrix
        return self.gene_matrix.shape[1]

    def __getitem__(self, idx):
        # Retrieve the feature vectors from both gene and motif matrices for a specific sample
        gene_features = self.gene_matrix[:, idx]
        motif_features = self.motif_matrix[:, idx]

        # Normalize the features
        gene_features = (gene_features - self.gene_mean) / (self.gene_std + 1e-8)  # Add a small epsilon to avoid division by zero
        motif_features = (motif_features - self.motif_mean) / (self.motif_std + 1e-8)

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
        self.fc3 = nn.Linear(input_dim, output_dim)  # Skip connection goes directly from input to output

    def forward(self, x):
        residual = self.fc3(x)  # Skip connection
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x + residual  # Add skip connection
        return x

# Define a model with both dropout and skip connections
class DropoutSkipConnectionNN(BaseModel):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(DropoutSkipConnectionNN, self).__init__(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(input_dim, output_dim)  # Skip connection

    def forward(self, x):
        residual = self.fc3(x)  # Skip connection
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = x + residual  # Add skip connection
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
    epochs = 15
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
