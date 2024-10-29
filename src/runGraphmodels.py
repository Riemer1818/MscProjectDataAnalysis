# runGraphModels.py

import os
import random
import json
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader 

from graphModelFunctions import (
    parse_arguments,
    get_num_samples_and_output_channels,
    load_sample_from_h5,
    compute_normalization_stats,
    normalize_node_features,
    normalize_motif_matrix,
    train_and_evaluate_model,   
    initialize_model,
    load_full_interaction_graph  # Make sure this is defined in graphModelFunctions2
)

from graphModels import GCN, GAT, GATWithDropout, GATWithBatchNorm, ModifiedGCN

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
    
    checkpoint_dir = 'checkpoints_finalGraphs'
    hdf5_graph_file = "/home/msai/riemerpi001/data/filtered_seurats/MC3/processed_data/graphsFile.h5"
    interaction_data_file = '/home/msai/riemerpi001/9606.protein.physical.links.v12.0.txt.gz'
    
    # Load the full interaction graph into memory
    full_edge_index, gene_to_index = load_full_interaction_graph(interaction_data_file)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================
    # DATA LOADING AND PREPROCESSING
    # ============================

    # Get the number of samples and output channels from the HDF5 file
    num_samples, out_channels = get_num_samples_and_output_channels(hdf5_graph_file)
    
    # Load each sample's data with the full graph as a reference for subsetting
    data_list = [load_sample_from_h5(hdf5_graph_file, idx, full_edge_index, gene_to_index) for idx in range(1, num_samples + 1)]

    # Compute and normalize features
    mean_x, std_x = compute_normalization_stats(data_list)
    data_list = normalize_node_features(data_list, mean_x, std_x)

    # Normalize target labels to range [0, 1] with a fixed baseline of 0.5
    data_list, baseline = normalize_motif_matrix(data_list)

    # Split dataset into training and validation
    train_data, val_data = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=42)

    # Optionally use a subset for exploration
    train_subset_size = int(0.001 * len(train_data))
    val_subset_size = int(0.001 * len(val_data))
    train_indices = random.sample(range(len(train_data)), train_subset_size)
    val_indices = random.sample(range(len(val_data)), val_subset_size)
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(val_data, val_indices)

    # Create DataLoader for training and validation
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    print(f"Training with {len(dataloader)} samples and validating with {len(val_dataloader)} samples")
    print(f"Training with {len(dataloader)} batches and validating with {len(val_dataloader)} batches")

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
