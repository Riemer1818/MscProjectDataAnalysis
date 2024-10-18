import pandas as pd
import h5py
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import os

# ======================
# CONFIGURATION
# ======================

# File paths
ENSEMBL_GENE_FILE = '~/data/ensembl_protein_ids.csv'  # Ensembl gene mapping file (17K nodes)
INTERACTION_DATA_FILE = '/home/msai/riemerpi001/data/9606.protein.physical.links.v12.0.txt.gz'  # Interaction data file
HDF5_GENE_MATRIX_FILE = 'final_gene_matrix.h5'  # HDF5 file with gene expression matrix
VARIABLE_GENES_FILE = '~/data/all_variable_genes.csv'  # CSV with all variable genes (500 genes for example)
OUTPUT_DIR = '/home/msai/riemerpi001/network_plots/'  # Directory to save output plots/graphs
SAMPLE_SUBSET = 10

# Other settings
DPI = 300  # Resolution of the output plot

# ======================
# SCRIPT
# ======================

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load the Ensembl gene mapping file (17K nodes of interest)
ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)

# Create a set of genes of interest (17K nodes)
genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])
print(f"Number of genes of interest: {len(genes_of_interest)}")

# Step 2: Load the interaction data from the STRING database
with gzip.open(INTERACTION_DATA_FILE, 'rt') as f:
    interaction_data = pd.read_csv(f, sep=' ')

# Remove '9606.' prefix from protein identifiers
interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

# Filter interaction data to include only interactions between proteins in the 17K nodes of interest
filtered_interactions = interaction_data[
    interaction_data['protein1'].isin(genes_of_interest) & 
    interaction_data['protein2'].isin(genes_of_interest)
]

# Step 3: Build the fixed graph structure from the filtered interactions
G_fixed = nx.Graph()

# Add edges to the graph
for index, row in filtered_interactions.iterrows():
    G_fixed.add_edge(row['protein1'], row['protein2'], weight=row['combined_score'])

# Step 4: Load the gene expression data (variable genes) and fold change matrix (multiple samples)
geneDf = pd.read_csv(VARIABLE_GENES_FILE, index_col=1)  # Assume this file has 500 variable genes

# Load fold change data from HDF5
with h5py.File(HDF5_GENE_MATRIX_FILE, 'r') as f:
    gene_matrix_np = f['gene_matrix'][:]  # Assuming rows are genes and columns are samples

# Get the number of samples from the matrix
num_samples = gene_matrix_np.shape[1]
print(f"Number of samples in the gene expression matrix: {num_samples}")

# Step 5: Iterate over each sample
# for sample_idx in range(num_samples):

for sample_idx in range(SAMPLE_SUBSET):
    print(f"Processing sample {sample_idx + 1} of {num_samples}...")

    # Copy the fixed graph structure to avoid modifying the original graph
    G_sample = G_fixed.copy()

    # Update fold change data for the sample
    for hgnc_symbol in geneDf.index:
        # Find matching proteins from the 17K gene list
        matching_proteins = ensembl_gene_df[ensembl_gene_df['hgnc_symbol'] == hgnc_symbol]

        # Get the fold change value for the current sample
        gene_index = geneDf.index.get_loc(hgnc_symbol)
        fold_change_value = gene_matrix_np[gene_index, sample_idx]  # Value for this sample

        # Update graph nodes with fold change data for each matching protein
        for ensembl_peptide_id in matching_proteins['ensembl_peptide_id']:
            if ensembl_peptide_id in G_sample:
                G_sample.nodes[ensembl_peptide_id]['fold_change'] = fold_change_value

    # Step 6: Visualize the graph for the current sample
    # Create a color map for the nodes based on fold change
    node_colors = []
    for node in G_sample.nodes:
        fold_change = G_sample.nodes[node].get('fold_change', 0)
        if fold_change > 0:
            node_colors.append('red')
        elif fold_change < 0:
            node_colors.append('blue')
        else:
            node_colors.append('gray')

    # Set the figure size for the network plot
    plt.figure(figsize=(20, 16))

    # Draw the network with updated attributes
    nx.draw(G_sample, node_color=node_colors, with_labels=False, node_size=50, edge_color='gray', alpha=0.6)

    # Save the plot for the current sample
    output_plot_file = os.path.join(OUTPUT_DIR, f"network_sample_{sample_idx + 1}.png")
    plt.savefig(output_plot_file, format='png', dpi=DPI)

    # Clear the plot to free memory
    plt.clf()

    print(f"Sample {sample_idx + 1} network plot saved to {output_plot_file}")

    # Optionally: You can also save the graph itself in a different format (e.g., GraphML, pickle, etc.)
    # nx.write_graphml(G_sample, os.path.join(OUTPUT_DIR, f"network_sample_{sample_idx + 1}.graphml"))
