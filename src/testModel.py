import os
import torch
from torch_geometric.data import DataLoader
from graphModelFunctions import (
    parse_arguments,
    load_and_create_edge_index,
    get_num_samples_and_output_channels,
    load_sample_from_h5,
    compute_normalization_stats,
    normalize_node_features,
    normalize_motif_matrix,
    evaluate_top_k_accuracy,
    load_model_from_checkpoint,
    initialize_model,
    setup_optimizer_and_scheduler,
)
from graphModels import GCN, GAT, GATWithDropout, GATWithBatchNorm, GIN

# ============================
# CONFIGURATION
# ============================

def construct_checkpoint_dir(model_name, base_dir='checkpoints_diffModels'):
    """Construct checkpoint directory based on model and criterion."""
    return os.path.join(base_dir, model_name)

# ============================
# DATA LOADING
# ============================

def prepare_data(interaction_file, hdf5_file, device):
    """Load and normalize data, returning a sample for inference."""
    # Load interaction data
    edge_index, _ = load_and_create_edge_index(interaction_file)
    
    # Load and normalize a sample from the dataset
    sample_data = load_sample_from_h5(hdf5_file, sample_idx=1, edge_index=edge_index)
    sample_data = sample_data.to(device)  # Move data to device
    
    # Normalize target values (motif matrix) for evaluation with baseline 0.5
    sample_data, _ = normalize_motif_matrix([sample_data])  # baseline is fixed at 0.5
    
    return sample_data[0]  # Return single sample from list

# ============================
# INFERENCE
# ============================

def run_inference(model, sample_data, k=10, baseline=0.5):
    """Run inference on a single sample and evaluate top-k accuracy."""
    model.eval()
    with torch.no_grad():
        output = model(sample_data)
    
    # Assuming `sample_data.y` contains the true target values
    if sample_data.y is not None:
        evaluate_top_k_accuracy(output, sample_data.y, k=k, baseline=baseline)
    else:
        print("No target values found in sample data for evaluation.")
    
    return output

# ============================
# LOAD MODEL CONFIGURATION
# ============================

def load_model_configuration_from_checkpoint(checkpoint_file):
    """Load the model configuration from a checkpoint file."""
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model_config = checkpoint['model_config']
    return model_config

# ============================
# MAIN EXECUTION
# ============================

def main():
    # Configuration and setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    model_name = "GCN"
    checkpoint_file = '/home/msai/riemerpi001/checkpoints_diffModels/GCN/MSELoss_V2/model_epoch_9.pt'

    # Load model configuration from checkpoint
    model_config = load_model_configuration_from_checkpoint(checkpoint_file)
    
    # Directory and model initialization
    checkpoint_dir = construct_checkpoint_dir(model_name)
    model = initialize_model(
        model_name, 
        model_config['in_channels'], 
        model_config['hidden_channels'], 
        model_config['out_channels'], 
        model_config['num_layers'],
        heads=model_config.get('heads', None),  # Use 'get' to avoid KeyError if 'heads' is not in config
        dropout=model_config.get('dropout', None)  # Same for dropout
    ).to(device)

    # Setup optimizer and scheduler if further training is needed
    optimizer, scheduler = setup_optimizer_and_scheduler(model)

    # Load model from checkpoint
    model, optimizer, scheduler = load_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Prepare data for inference
    interaction_file = '9606.protein.physical.links.v12.0.txt.gz'
    hdf5_file = "protein.physical.links.full_links.h5"
    sample_data = prepare_data(interaction_file, hdf5_file, device)
    
    # Run inference
    print(run_inference(model, sample_data))


if __name__ == "__main__":
    main()
