import h5py
import torch
import torch_geometric as tg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx
from networkx.algorithms.community import greedy_modularity_communities

# ======================
# CONFIGURATION
# ======================

# File paths
HDF5_GRAPH_FILE = 'combined_graph_data.h5'  # HDF5 file containing graph data (X and Y)
ENSEMBL_GENE_FILE = '~/data/ensembl_protein_ids_cleaned.csv'  # Ensembl gene mapping file
INTERACTION_DATA_FILE = '/home/msai/riemerpi001/data/9606.protein.physical.links.v12.0.txt.gz'  # Interaction data file

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
# STEP 3: PLOT A SMALL PIECE OF THE GRAPH WITH CLUSTERING
# ======================

# Convert to NetworkX for visualization
g = to_networkx(original_graph, to_undirected=True)

# Take a small subgraph (e.g., 50 nodes)
small_subgraph = g.subgraph(list(g.nodes)[:50])

# Apply a simple layout to the small subgraph
pos = nx.spring_layout(small_subgraph)  # Use spring layout for visualization

# Apply community detection (clustering) on the small subgraph
communities = list(greedy_modularity_communities(small_subgraph))

# Create a dictionary to map nodes to their community index
community_map = {}
for i, community in enumerate(communities):
    for node in community:
        community_map[node] = i

# Create a color array based on the community of each node
node_color = [community_map[node] for node in small_subgraph.nodes]

# Plot the small subgraph with community-based coloring
plt.figure(figsize=(10, 7))
nx.draw(small_subgraph, pos=pos, node_size=100, node_color=node_color, cmap=plt.cm.tab20, 
        edge_color='gray', with_labels=False, alpha=0.8)
plt.title("Small Subgraph Visualization with Communities (50 Nodes)")

# Save the plot as an image
plt.savefig("small_subgraph_with_communities.png")
plt.close()

print("Plot with communities saved successfully as 'small_subgraph_with_communities.png'.")

#### I NEED A WAY TO CHECK IF THE GRAPH IS CORRECTLY BUILT. I AM GETTING A FEELING THE MAJORITY OF THE GRAPH IS ORPHANED.

# import h5py
# import torch
# import torch_geometric as tg
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from torch_geometric.utils import to_networkx
# from sklearn.decomposition import PCA
# from networkx.algorithms.community import greedy_modularity_communities

# # ======================
# # CONFIGURATION
# # ======================

# # File paths
# HDF5_GRAPH_FILE = 'combined_graph_data.h5'  # HDF5 file containing graph data (X and Y)
# ENSEMBL_GENE_FILE = '~/data/ensembl_protein_ids_cleaned.csv'  # Ensembl gene mapping file
# INTERACTION_DATA_FILE = '/home/msai/riemerpi001/data/9606.protein.physical.links.v12.0.txt.gz'  # Interaction data file

# # ======================
# # STEP 1: LOAD AND INSPECT UPDATED GRAPH DATA
# # ======================

# sample_idx = 1  # You can change this to inspect other samples
# with h5py.File(HDF5_GRAPH_FILE, 'r') as f:
#     sample_group = f[f'sample_{sample_idx}']
#     updated_node_features = sample_group['X'][:]  # Node features (gene expression)
#     motif_matrix = sample_group['Y'][:]   # Motif matrix

# print(f"Sample {sample_idx} - Updated Node features shape: {updated_node_features.shape}")
# print(f"Sample {sample_idx} - Motif matrix shape: {motif_matrix.shape}")

# # ======================
# # STEP 2: LOAD ORIGINAL GRAPH FROM INTERACTION DATA
# # ======================

# # Load interaction data
# interaction_data = pd.read_csv(INTERACTION_DATA_FILE, compression='gzip', sep=' ')
# interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
# interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

# # Load the Ensembl gene mapping file
# ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)
# ensembl_gene_dict = ensembl_gene_df.groupby('hgnc_symbol')['ensembl_peptide_id'].apply(list).to_dict()

# # Create a set of genes of interest (from the Ensembl mapping)
# genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])

# # Filter interaction data to include only interactions between proteins in the 17K nodes of interest
# filtered_interactions = interaction_data[
#     interaction_data['protein1'].isin(genes_of_interest) & 
#     interaction_data['protein2'].isin(genes_of_interest)
# ]

# # Map genes to node indices
# node_mapping = {gene: idx for idx, gene in enumerate(genes_of_interest)}

# # Prepare edges for PyTorch Geometric format (indices of nodes)
# edges = []
# for _, row in filtered_interactions.iterrows():
#     if row['protein1'] in node_mapping and row['protein2'] in node_mapping:
#         edges.append([node_mapping[row['protein1']], node_mapping[row['protein2']]])

# # Convert edges to PyTorch tensor
# edges = torch.tensor(edges).t().contiguous()

# # Initialize node features as zeros (or some baseline feature values)
# num_nodes = len(genes_of_interest)
# original_node_features = torch.zeros((num_nodes, 1))  # All nodes initialized to zero

# # Create the original graph as a PyTorch Geometric Data object
# original_graph = tg.data.Data(x=original_node_features, edge_index=edges)

# print(f"Original Graph - Node features shape: {original_node_features.shape}")
# print(f"Original Graph - Number of edges: {edges.shape[1]}")

# # ======================
# # STEP 3: COMPARE ORIGINAL AND UPDATED GRAPHS
# # ======================

# # Visualize top changed nodes
# def visualize_top_changed_nodes(graph_data_before, graph_data_after, top_k=1000, save_as="graph_with_top_changes.png"):
#     # Calculate node feature changes
#     node_changes = torch.abs(graph_data_before.x - graph_data_after.x).squeeze().numpy()

#     # Sort nodes by the magnitude of change and select the top_k nodes
#     top_k_indices = np.argsort(-node_changes)[:top_k]

#     # Convert to NetworkX graph
#     g_after = to_networkx(graph_data_after, to_undirected=True)
#     pos = nx.random_layout(g_after)

#     # pos = nx.spring_layout(g_after)

#     # Create a subgraph with only the top_k changed nodes
#     top_k_nodes = list(top_k_indices)
#     subgraph_after = g_after.subgraph(top_k_nodes)

#     # Extract positions for the top_k nodes
#     top_k_pos = {node: pos[node] for node in top_k_nodes}
    
#     plt.figure(figsize=(10, 7))
    
#     # Extract node colors for the top_k nodes
#     node_color = [node_changes[n] for n in top_k_nodes]

#     # Draw the subgraph with colored nodes
#     nx.draw(subgraph_after, pos=top_k_pos, node_size=50, node_color=node_color, cmap=plt.cm.Reds, 
#             edge_color='gray', alpha=0.8, with_labels=False)

#     # Add colorbar
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
#     sm.set_array([])
#     plt.colorbar(sm, label='Node Change Intensity')

#     plt.title(f"Top {top_k} Nodes with Largest Changes in Features")
#     plt.savefig(save_as)
#     plt.close()

# # Visualize graph colored by communities using community detection
# def visualize_clustered_graph(graph_data_after, save_as="clustered_graph.png"):
#     g_after = to_networkx(graph_data_after, to_undirected=True)
    
#     # Apply a community detection algorithm to identify clusters (communities)
#     communities = list(greedy_modularity_communities(g_after))
    
#     # Create a color mapping for each community
#     community_color_map = {node: idx for idx, community in enumerate(communities) for node in community}
    
#     # Draw the graph with nodes colored by their community
#     # pos = nx.spring_layout(g_after)
#     pos = nx.random_layout(g_after)

    
#     node_color = [community_color_map[n] for n in g_after.nodes]

#     plt.figure(figsize=(10, 7))
#     nx.draw(g_after, pos=pos, node_size=50, node_color=node_color, cmap=plt.cm.tab20, 
#             edge_color='gray', alpha=0.8, with_labels=False)
    
#     plt.title("Graph Colored by Communities")
#     plt.savefig(save_as)
#     plt.close()

# # def compare_graphs(
    
# # ======================
# # STEP 4: BUILD UPDATED GRAPH AND VISUALIZE
# # ======================

# # Build the updated graph for comparison
# updated_graph = tg.data.Data(x=torch.tensor(updated_node_features, dtype=torch.float32), edge_index=edges)

# print("Comparing original and updated graphs...")

# # compare_graphs(original_graph, updated_graph)

# # Visualize the top 1000 changed nodes
# # visualize_top_changed_nodes(original_graph, updated_graph, top_k=1000, save_as="graph_with_top_changes.png")

# # Visualize graph with communities
# visualize_clustered_graph(updated_graph, save_as="clustered_graph.png")

# print("Plots saved successfully.")
