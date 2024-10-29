# runNeuralNetworks.py

import torch
from neuralNetworkFunctions import create_dataloader, train_model, parse_arguments
from neuralNetworks import BaselineNN, DropoutNN, SkipConnectionNN, DropoutSkipConnectionNN
import os
import os
import torch

def main():
    # Parse arguments
    args = parse_arguments()

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.num_epochs
    output_dim = 541  # Number of output features (motifs)
    checkpoint_dir = 'finalNNModels'  # Directory to save checkpoints
    # output_dir = os.path.join(checkpoint_dir, 'model_losses')  # Directory to save the loss files

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Select the criterion based on the specified loss function
    if args.loss_function == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss_function == 'MSE':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}. Choose either 'BCE' or 'MSE'.")

    output_dir = os.path.join(checkpoint_dir, args.loss_function)
    # Load the dataset using the DataLoader defined in neuralNetworkFunctions.py
    gene_matrix_path = '/home/msai/riemerpi001/data/filtered_seurats/MC3/processed_data/gene_matrix.h5'
    motif_matrix_path = '/home/msai/riemerpi001/data/filtered_seurats/MC3/processed_data/motif_matrix.h5'
    train_loader = create_dataloader(gene_matrix_path, motif_matrix_path, batch_size, shuffle=True)
    val_loader = create_dataloader(gene_matrix_path, motif_matrix_path, batch_size, shuffle=False)

    # Print the dimensions of the dataloaders
    train_samples, train_features = next(iter(train_loader))[0].shape
    val_samples, val_features = next(iter(val_loader))[0].shape
    print(f"Train Loader - Samples: {train_samples}, Features: {train_features}")
    print(f"Validation Loader - Samples: {val_samples}, Features: {val_features}")

    input_dim = train_features
    
    # Define available models
    models = {'BaselineNN': BaselineNN(input_dim, output_dim),
        'DropoutNN': DropoutNN(input_dim, output_dim, dropout_prob=0.5),
        'SkipConnectionNN': SkipConnectionNN(input_dim, output_dim),
        'DropoutSkipConnectionNN': DropoutSkipConnectionNN(input_dim, output_dim, dropout_prob=0.5)
    }

    # Check if the specified model is valid
    model_name = args.model
    if model_name not in models:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from {list(models.keys())}.")

    # Select the model based on the argument
    model = models[model_name]

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train the selected model
    print(f"Training {model_name} with {args.loss_function} loss function...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        model_name=model_name,
        output_dir=output_dir,
        criterion=criterion,
        checkpoint_dir=checkpoint_dir
    )
    print(f"Finished training {model_name}.\n")

if __name__ == '__main__':
    main()
