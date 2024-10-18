import h5py
import torch
import torch_geometric as tg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx
from scipy.stats import zscore

# ======================
# CONFIGURATION
# ======================

# File paths
HDF5_GRAPH_FILE = 'combined_graph_data_full_links.h5'  # HDF5 file containing graph data (X and Y)
ENSEMBL_GENE_FILE = '~/data/ensembl_protein_ids_cleaned.csv'  # Ensembl gene mapping file
# INTERACTION_DATA_FILE = '/home/msai/riemerpi001/data/9606.protein.physical.links.v12.0.txt.gz'  # Interaction data file
INTERACTION_DATA_FILE = '/home/msai/riemerpi001/9606.protein.links.v12.0.txt.gz' # STRING -> 79mb

# ======================
# STEP 1: LOAD AND INSPECT UPDATED GRAPH DATA
# ======================

sample_idx = 1  # You can change this to inspect other samples
with h5py.File(HDF5_GRAPH_FILE, 'r') as f:
    sample_group = f[f'sample_{sample_idx}']
    updated_node_features = sample_group['X'][:]  # Node features (gene expression)
    motif_matrix = sample_group['Y'][:]   # Motif matrix

print(f"Sample {sample_idx} - Updated Node features shape: {updated_node_features.shape}")
print(f"Sample {sample_idx} - Motif matrix shape: {motif_matrix.shape}")

# ======================
# STEP 2: LOAD ORIGINAL GRAPH FROM INTERACTION DATA
# ======================

# Load interaction data
interaction_data = pd.read_csv(INTERACTION_DATA_FILE, compression='gzip', sep=' ')
interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

# Load the Ensembl gene mapping file
ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)
ensembl_gene_dict = ensembl_gene_df.groupby('hgnc_symbol')['ensembl_peptide_id'].apply(list).to_dict()

# Create a set of genes of interest (from the Ensembl mapping)
genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])

# Filter interaction data to include only interactions between proteins in the 17K nodes of interest
filtered_interactions = interaction_data[
    interaction_data['protein1'].isin(genes_of_interest) & 
    interaction_data['protein2'].isin(genes_of_interest)
]

# Map genes to node indices
node_mapping = {gene: idx for idx, gene in enumerate(genes_of_interest)}

# Prepare edges for PyTorch Geometric format (indices of nodes)
edges = []
for _, row in filtered_interactions.iterrows():
    if row['protein1'] in node_mapping and row['protein2'] in node_mapping:
        edges.append([node_mapping[row['protein1']], node_mapping[row['protein2']]])

# Convert edges to PyTorch tensor
edges = torch.tensor(edges).t().contiguous()

# Initialize node features as zeros (or some baseline feature values)
num_nodes = len(genes_of_interest)
original_node_features = torch.zeros((num_nodes, 1))  # All nodes initialized to zero

# Create the original graph as a PyTorch Geometric Data object
original_graph = tg.data.Data(x=original_node_features, edge_index=edges)

print(f"Original Graph - Node features shape: {original_node_features.shape}")
print(f"Original Graph - Number of edges: {edges.shape[1]}")

# ======================
# STEP 3: COMPARE ORIGINAL AND UPDATED GRAPHS
# ======================

def compare_graphs(graph_data_before, graph_data_after):
    # Convert both graphs to NetworkX for easy comparison and visualization
    g_before = to_networkx(graph_data_before, to_undirected=True)
    g_after = to_networkx(graph_data_after, to_undirected=True)

    # Check for any differences in edges
    edges_before = set(g_before.edges)
    edges_after = set(g_after.edges)
    edge_diff = edges_before.symmetric_difference(edges_after)

    if len(edge_diff) == 0:
        print("Graph structure (edges) is unchanged.")
    else:
        print(f"Graph structure has changed. {len(edge_diff)} edges are different.")
    
    # # Visualize the graph before and after for manual inspection (optional)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # nx.draw(g_before, node_size=50, alpha=0.8)
    # plt.title("Graph Before Update")

    # plt.subplot(1, 2, 2)
    # nx.draw(g_after, node_size=50, alpha=0.8)
    # plt.title("Graph After Update")

    # plt.show()

# Build the updated graph for comparison (this uses the updated node features from the HDF5 file)
updated_graph = tg.data.Data(x=torch.tensor(updated_node_features, dtype=torch.float32), edge_index=edges)

# Compare original and updated graphs
compare_graphs(original_graph, updated_graph)
