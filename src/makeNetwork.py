import pandas as pd
import h5py
import torch
import torch_geometric as tg
import os
import numpy as np
from torch_geometric.data import Data

# ======================
# CONFIGURATION
# ======================

# File paths
ENSEMBL_GENE_FILE = '~/data/ensembl_protein_ids.csv'  # Ensembl gene mapping file (17K nodes)
INTERACTION_DATA_FILE = '/home/msai/riemerpi001/data/9606.protein.physical.links.v12.0.txt.gz'  # Interaction data file
HDF5_GENE_MATRIX_FILE = 'final_gene_matrix.h5'  # HDF5 file with gene expression matrix
VARIABLE_GENES_FILE = '~/data/all_variable_genes.csv'  # CSV with all variable genes (500 genes for example)
OUTPUT_DIR = '/home/msai/riemerpi001/network_graphs/'  # Directory to save updated graph data (optional)

# Other settings
NUM_SAMPLES = 1000  # Number of samples to process
USE_GPU = torch.cuda.is_available()  # Check if GPU is available

# ======================
# SCRIPT
# ======================

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load the Ensembl gene mapping file (17K nodes of interest)
ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)

# Create a dictionary mapping from hgnc_symbol to ensembl_peptide_id for fast lookup
ensembl_gene_dict = ensembl_gene_df.groupby('hgnc_symbol')['ensembl_peptide_id'].apply(list).to_dict()

# Create a set of genes of interest (17K nodes)
genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])
print(f"Number of genes of interest: {len(genes_of_interest)}")

# Step 2: Load the interaction data
interaction_data = pd.read_csv(INTERACTION_DATA_FILE, compression='gzip', sep=' ')

# Remove '9606.' prefix from protein identifiers
interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

# Filter interaction data to include only interactions between proteins in the 17K nodes of interest
filtered_interactions = interaction_data[
    interaction_data['protein1'].isin(genes_of_interest) & 
    interaction_data['protein2'].isin(genes_of_interest)
]

# Step 3: Build the fixed graph structure using PyTorch Geometric
node_mapping = {gene: idx for idx, gene in enumerate(genes_of_interest)}

# Prepare edges for PyTorch Geometric format (indices of nodes)
edges = []
for _, row in filtered_interactions.iterrows():
    if row['protein1'] in node_mapping and row['protein2'] in node_mapping:
        edges.append([node_mapping[row['protein1']], node_mapping[row['protein2']]])

edges = torch.tensor(edges).t().contiguous()

# Create PyTorch Geometric Data object
graph_data = Data(edge_index=edges)

# Step 4: Load the gene expression data (variable genes) and fold change matrix
geneDf = pd.read_csv(VARIABLE_GENES_FILE, index_col=1)

# Load fold change data from HDF5
with h5py.File(HDF5_GENE_MATRIX_FILE, 'r') as f:
    gene_matrix_np = f['gene_matrix'][:]  # Assuming rows are genes and columns are samples

# Get the number of samples from the matrix
num_total_samples = gene_matrix_np.shape[1]
print(f"Number of samples in the gene expression matrix: {num_total_samples}")

# Use GPU if available
device = torch.device('cuda' if USE_GPU else 'cpu')
graph_data = graph_data.to(device)

# Limit to NUM_SAMPLES for testing or set it to the full number of samples
samples_to_process = min(NUM_SAMPLES, num_total_samples)

# Process each sample sequentially
for sample_idx in range(samples_to_process):
    print(f"Processing sample {sample_idx + 1}...")

    # Initialize node feature tensor for fold change (size: [num_nodes, 1])
    node_features = torch.zeros((len(genes_of_interest), 1), device=device)

    # Update fold change data for the sample
    for hgnc_symbol in geneDf.index:
        if hgnc_symbol in ensembl_gene_dict:
            gene_index = geneDf.index.get_loc(hgnc_symbol)
            fold_change_value = gene_matrix_np[gene_index, sample_idx]  # Value for this sample

            # Update node features
            for ensembl_peptide_id in ensembl_gene_dict[hgnc_symbol]:
                if ensembl_peptide_id in node_mapping:
                    node_idx = node_mapping[ensembl_peptide_id]
                    node_features[node_idx] = fold_change_value

    # Update graph data with node features
    graph_data.x = node_features

    # Optionally save the updated graph data (PyTorch Geometric Data object)
    output_graph_file = os.path.join(OUTPUT_DIR, f"graph_sample_{sample_idx + 1}.pt")
    torch.save(graph_data, output_graph_file)

    print(f"Sample {sample_idx + 1} graph saved to {output_graph_file}")

# actually just make the new h5ad e.g file with the network in 1 file. 