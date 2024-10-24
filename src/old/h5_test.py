import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np

# File paths
gene_matrix_file = 'final_gene_matrix.h5'
motif_matrix_file = 'final_motif_matrix.h5'
gene_matrix_threshold_file = gene_matrix_file # the threshold should not matter as it only affects the motif matrix
motif_matrix_threshold_file = 'final_motif_matrix_thresh_0.2.h5'

# PyTorch dtype
dtype = torch.float32

# Load data from HDF5 files
def load_data(gene_file, motif_file):
    with h5py.File(gene_file, 'r') as f:
        gene_matrix_np = f['gene_matrix'][:] # sample * genes
    with h5py.File(motif_file, 'r') as f:
        motif_matrix_np = f['motif_matrix'][:] # sample * TFs ([0-4])s  
    
    return gene_matrix_np, motif_matrix_np

# Convert numpy arrays to PyTorch tensors
def to_tensor(gene_matrix_np, motif_matrix_np):
    gene_matrix_tensor = torch.tensor(gene_matrix_np, dtype=dtype)
    motif_matrix_tensor = torch.tensor(motif_matrix_np, dtype=dtype)
    
    return gene_matrix_tensor, motif_matrix_tensor

# Display sample data
def display_samples(gene_matrix_tensor, motif_matrix_tensor, n=5):
    gene_matrix_sample = gene_matrix_tensor[:n]
    motif_matrix_sample = motif_matrix_tensor[:n]
    print("Gene matrix sample:\n", gene_matrix_sample)
    print("Motif matrix sample:\n", motif_matrix_sample)

def plot_motif_distribution(motif_matrix_tensor, output_file):
    # Exclude zeros for the plot
    non_zero_motif_matrix = motif_matrix_tensor[motif_matrix_tensor != 0]
    
    # Plot histogram of non-zero values
    if non_zero_motif_matrix.nelement() > 0:  # Check if there are non-zero elements
        plt.hist(non_zero_motif_matrix.flatten().numpy(), bins=50, alpha=0.7, color='blue')
        plt.title("Distribution of Non-Zero Motif Matrix Values")
    else:
        plt.title("No Non-Zero Values Found")
    
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig(output_file)

# Calculate and plot averages and standard deviations
def plot_averages_and_std(motif_matrix_np, output_file):
    column_averages = np.mean(motif_matrix_np, axis=1)
    column_std_devs = np.std(motif_matrix_np, axis=1)

    print("Average over the columns:\n", column_averages)
    print("Standard deviation over the columns:\n", column_std_devs)

    plt.figure(figsize=(10, 6))
    plt.plot(column_averages, label="Average", color='blue')
    plt.fill_between(range(len(column_averages)), 
                     column_averages - column_std_devs, 
                     column_averages + column_std_devs, 
                     color='lightblue', alpha=0.5, label="1 Standard Deviation")
    plt.title("Average Values and Standard Deviation Across TF Columns of Motif Matrix")
    plt.xlabel("TF Index")
    plt.ylabel("Average Value")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig(output_file)

# Print statistics of motif matrix
def print_stats(matrix_np, description="Matrix"):
    print(f"{description} shape: {matrix_np.shape}")
    print(f"Min value: {np.min(matrix_np)}")
    print(f"Max value: {np.max(matrix_np)}")
    print(f"Mean value: {np.mean(matrix_np)}")
    print(f"Standard deviation: {np.std(matrix_np)}")
    
# Main execution for original data
gene_matrix_np, motif_matrix_np = load_data(gene_matrix_file, motif_matrix_file)
gene_matrix_tensor, motif_matrix_tensor = to_tensor(gene_matrix_np, motif_matrix_np)
display_samples(gene_matrix_tensor, motif_matrix_tensor)
plot_motif_distribution(motif_matrix_tensor, 'motif_matrix_distribution.png')
plot_averages_and_std(motif_matrix_np, 'motif_matrix_column_averages_and_std.png')

print_stats(gene_matrix_np, description="Gene Matrix")
print_stats(motif_matrix_np, description="Motif Matrix")

# Main execution for threshold data
gene_matrix_np_thresh, motif_matrix_np_thresh = load_data(gene_matrix_threshold_file, motif_matrix_threshold_file)
gene_matrix_tensor_thresh, motif_matrix_tensor_thresh = to_tensor(gene_matrix_np_thresh, motif_matrix_np_thresh)
display_samples(gene_matrix_tensor_thresh, motif_matrix_tensor_thresh)
plot_motif_distribution(motif_matrix_tensor_thresh, 'motif_matrix_distribution_threshold.2.png')
plot_averages_and_std(motif_matrix_np_thresh, 'motif_matrix_column_averages_and_std_threshold.2.png')

print_stats(gene_matrix_np_thresh, description="Gene Matrix Threshold")
print_stats(motif_matrix_np_thresh, description="Motif Matrix Threshold")

