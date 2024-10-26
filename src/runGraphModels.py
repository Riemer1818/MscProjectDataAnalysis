# train_graph_model.py

import os
import random
import json
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from graphModelFunctions import (
    parse_arguments,
    load_and_create_edge_index,
    get_num_samples_and_output_channels,
    load_sample_from_h5,
    compute_normalization_stats,
    normalize_node_features,
    normalize_motif_matrix,
    train_and_evaluate_model,   
    initialize_model,
)

from graphModels import GCN, GAT, GATWithDropout, GATWithBatchNorm, GIN

if __name__ == "__main__":
    # ============================
    # CONFIGURATION AND SETUP
    # ============================

    # Parse arguments for dynamic configuration
    args = parse_arguments()

    model_name = args.model
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    hidden_channels = args.hidden_channels
    num_layers = args.num_layers
    
    checkpoint_dir = 'checkpoints_diffModels'
    interaction_data_file = '9606.protein.physical.links.v12.0.txt.gz'
    hdf5_graph_file = "protein.physical.links.full_links.h5"
    T_max = 10
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCEWithLogitsLoss()


    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================
    # DATA LOADING AND PREPROCESSING
    # ============================

    # Load data
    edge_index, num_nodes = load_and_create_edge_index(interaction_data_file)
    num_samples, out_channels = get_num_samples_and_output_channels(hdf5_graph_file)
    data_list = [load_sample_from_h5(hdf5_graph_file, idx, edge_index) for idx in range(1, num_samples + 1)]

    # Compute and normalize features
    mean_x, std_x = compute_normalization_stats(data_list)
    data_list = normalize_node_features(data_list, mean_x, std_x)

    # Normalize target labels to range [0, 1] with a fixed baseline of 0.5
    data_list, baseline = normalize_motif_matrix(data_list)

    # Split dataset into training and validation
    train_data, val_data = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)

    # Optionally use a subset for exploration
    train_subset_size = int(0.1 * len(train_data))
    val_subset_size = int(0.1 * len(val_data))
    train_indices = random.sample(range(len(train_data)), train_subset_size)
    val_indices = random.sample(range(len(val_data)), val_subset_size)
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(val_data, val_indices)

    # Create DataLoader for training and validation
    dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    print(f"Training with {len(train_subset)} samples and validating with {len(val_subset)} samples")

    # ============================
    # MODEL INITIALIZATION AND TRAINING
    # ============================

    # Initialize the model
    in_channels = data_list[0].x.shape[1]

    model = initialize_model(model_name, in_channels, hidden_channels, out_channels, num_layers).to(device)


    # Train and evaluate the model
    train_losses, val_losses, accuracies = train_and_evaluate_model(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        T_max=T_max,
        device=device,
        checkpoint_dir=checkpoint_dir,
        criterion=criterion
    )

    # ============================
    # SAVE TRAINING LOSSES
    # ============================

    losses = {'train_losses': train_losses, 'val_losses': val_losses, 'accuracies': accuracies}
    losses_path = os.path.join(checkpoint_dir, 'losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses, f)

    print("Training complete. Losses saved to", losses_path)
