import h5py
import torch
import torch_geometric as tg
import pandas as pd
import os
import numpy as np
import gc
from torch_geometric.data import Data

# ======================
# CONFIGURATION
# ======================

# Set a threshold to remove weak edges after binning
THRESHOLD = 0.001  # Example threshold for binning, adjust as needed
NUM_BINS = 50   # Define the number of bins (adjust based on your needs)
# NUM_SAMPLES = 100

# File paths
HDF5_GENE_MATRIX_FILE = 'final_gene_matrix2.h5'  # HDF5 file with gene expression matrix. Threshold thing is not important. 
MOTIF_MATRIX_FILE = 'final_motif_matrix2.h5'     # HDF5 file with motif matrix
ENSEMBL_GENE_FILE = '~/data/ensembl_protein_ids_cleaned.csv'  # Ensembl gene mapping file
OUTPUT_HDF5_FILE = 'protein.physical.links.full_links2.h5'  # HDF5 file containing graph data (X and Y)
INTERACTION_DATA_FILE = '9606.protein.physical.links.v12.0.txt.gz'  # Interaction data file

USE_GPU = torch.cuda.is_available()  # Check if GPU is available

ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)
ensembl_gene_dict = ensembl_gene_df.groupby('hgnc_symbol')['ensembl_peptide_id'].apply(list).to_dict()
genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])

# Step 2: Load the interaction data and build the graph
interaction_data = pd.read_csv(INTERACTION_DATA_FILE, compression='gzip', sep=' ')
interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

# Define binning function for edge weights
def bin_weight(weight, num_bins=10):
    """Bin the weight into discrete bins."""
    return np.round(weight * num_bins) / num_bins

# Apply binning to edge weights and remove edges below the threshold
interaction_data['weight_binned'] = interaction_data['combined_score'].apply(lambda x: bin_weight(x / 1000, NUM_BINS))
binned_interactions = interaction_data[interaction_data['weight_binned'] >= THRESHOLD]

# Rebuild the edge list
# node_mapping = {gene: idx for idx, gene in enumerate(genes_of_interest)}

# Expand the node mapping to include all unique proteins in interaction data (protein1 and protein2)
all_proteins = set(interaction_data['protein1']).union(set(interaction_data['protein2']))

# Create a mapping that includes all proteins
node_mapping = {protein: idx for idx, protein in enumerate(all_proteins)}

# Print the number of nodes in the final node mapping
print(f"Total number of nodes in the expanded node mapping: {len(node_mapping)}")

edges = []
for _, row in binned_interactions.iterrows():
    if row['protein1'] in node_mapping and row['protein2'] in node_mapping:
        edges.append([node_mapping[row['protein1']], node_mapping[row['protein2']]])

# Create the final edge tensor
edges = torch.tensor(edges).t().contiguous()
graph_data = Data(edge_index=edges)

# Step 3: Load the gene expression matrix (X) and motif matrix (Y) as before
with h5py.File(HDF5_GENE_MATRIX_FILE, 'r') as f:
    gene_matrix_np = f['gene_matrix'][:].T  # X: Gene expression data

with h5py.File(MOTIF_MATRIX_FILE, 'r') as f:
    motif_matrix_np = f['motif_matrix'][:].T # Y: Motif matrix data

print(f"Gene matrix shape: {gene_matrix_np.shape}")
print(f"Motif matrix shape: {motif_matrix_np.shape}")

# Number of samples in the gene matrix (assuming samples are rows)
num_total_samples = gene_matrix_np.shape[0]

# Use GPU if available
device = torch.device('cuda' if USE_GPU else 'cpu')
graph_data = graph_data.to(device)

# Determine the number of samples to process
samples_to_process = num_total_samples  # Default: process all samples
if 'NUM_SAMPLES' in globals():  # Check if NUM_SAMPLES is defined
    samples_to_process = min(NUM_SAMPLES, num_total_samples)
    print(f"Processing {samples_to_process} out of {num_total_samples} samples as specified.")
else:
    print(f"Processing all {num_total_samples} samples.")

# Precompute gene index map to avoid repeated list conversion
gene_index_map = {hgnc_symbol: idx for idx, hgnc_symbol in enumerate(ensembl_gene_dict.keys())}


# Step 4: Process each sample and save X and Y into a single HDF5 file
with h5py.File(OUTPUT_HDF5_FILE, 'w') as hdf_out:
    for sample_idx in range(samples_to_process):
        print(f"Processing sample {sample_idx + 1}...")
        
        # Initialize node feature tensor for all nodes (based on expanded node_mapping)
        node_features = torch.zeros((len(node_mapping), 1), device=device)

        # Assign fold change values to the node features for the current sample
        for hgnc_symbol, ensembl_peptides in ensembl_gene_dict.items():
            # Get the gene index for the current HGNC symbol
            gene_idx = gene_index_map[hgnc_symbol]

            # Get fold change value from gene_matrix_np for the current sample and gene
            fold_change_value = gene_matrix_np[sample_idx, gene_idx]  # Corrected indexing: sample_idx as row, gene_idx as column

            # Assign this value to all corresponding ensembl_peptide_id nodes
            for ensembl_peptide_id in ensembl_peptides:
                if ensembl_peptide_id in node_mapping:
                    node_idx = node_mapping[ensembl_peptide_id]
                    node_features[node_idx] = fold_change_value  # Update only for nodes in node_mapping

        # Convert node features to NumPy array
        node_features_np = node_features.cpu().numpy()

        # Clear node_features tensor to free GPU memory
        del node_features
        torch.cuda.empty_cache()

        # Get the motif data (Y) for the current sample
        motif_data_np = motif_matrix_np[sample_idx, :]  # Y value for the sample

        # Save the data for this sample in the HDF5 file
        group = hdf_out.create_group(f'sample_{sample_idx + 1}')
        group.create_dataset('X', data=node_features_np)  # Save X (node features)
        group.create_dataset('Y', data=motif_data_np)     # Save Y (motif matrix)

        # Manually run garbage collection to free any lingering memory
        del node_features_np, motif_data_np
        gc.collect()

        print(f"Sample {sample_idx + 1} saved.")