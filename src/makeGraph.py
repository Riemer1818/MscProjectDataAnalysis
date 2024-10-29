import h5py
import torch
import torch_geometric as tg
import gc
import pandas as pd
import numpy as np
import argparse
from torch_geometric.data import Data

# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Process gene and motif data to create a graph dataset.")
#     parser.add_argument("--threshold", type=float, default=0.001, help="Threshold for binning weak edges.")
#     parser.add_argument("--num_bins", type=int, default=50, help="Number of bins for edge weight binning.")
#     parser.add_argument("--hdf5_gene_matrix_file", type=str, required=True, help="Path to HDF5 file with gene matrix.")
#     parser.add_argument("--motif_matrix_file", type=str, required=True, help="Path to HDF5 file with motif matrix.")
#     parser.add_argument("--ensembl_gene_file", type=str, required=True, help="Path to Ensembl gene mapping CSV file.")
#     parser.add_argument("--output_hdf5_file", type=str, required=True, help="Path to output HDF5 file.")
#     parser.add_argument("--interaction_data_file", type=str, required=True, help="Path to interaction data file (gzip).")
#     parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process.")
#     args = parser.parse_args()
#     return args

def main():
    # Parse command-line arguments
    # args = parse_arguments()

    # ======================
    # CONFIGURATION
    # ======================

    # Set parameters from arguments
    NUM_SAMPLES = None
    THRESHOLD = 0.5
    NUM_BINS = 50
    HDF5_GENE_MATRIX_FILE = "/home/msai/riemerpi001/data/filtered_seurats/MC3/processed_data/gene_matrix.h5"
    MOTIF_MATRIX_FILE = "/home/msai/riemerpi001/data/filtered_seurats/MC3/processed_data/motif_matrix.h5"
    ENSEMBL_GENE_FILE = "/home/msai/riemerpi001/data/filtered_seurats/MC3/ensembl_protein_ids_cleaned.csv"
    OUTPUT_HDF5_FILE = "/home/msai/riemerpi001/data/filtered_seurats/MC3/processed_data/graphsFile.h5"
    INTERACTION_DATA_FILE = "/home/msai/riemerpi001/9606.protein.physical.links.v12.0.txt.gz"

    # Step 1: Load gene data
    ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)
    ensembl_gene_dict = ensembl_gene_df.groupby('hgnc_symbol')['ensembl_peptide_id'].apply(list).to_dict()
    genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])

    # Step 2: Load interaction data and build the graph
    interaction_data = pd.read_csv(INTERACTION_DATA_FILE, compression='gzip', sep=' ')
    interaction_data['protein1'] = interaction_data['protein1'].str.replace('9606.', '')
    interaction_data['protein2'] = interaction_data['protein2'].str.replace('9606.', '')

    # Filter edges based on threshold and binning
    def bin_weight(weight, num_bins=10):
        """Bin the weight into discrete bins."""
        return np.round(weight * num_bins) / num_bins

    interaction_data['weight_binned'] = interaction_data['combined_score'].apply(lambda x: bin_weight(x / 1000, NUM_BINS))
    binned_interactions = interaction_data[interaction_data['weight_binned'] >= THRESHOLD]

    # Limit the number of edges per node to reduce graph density
    top_k_edges = binned_interactions.groupby('protein1').apply(lambda df: df.nlargest(20, 'weight_binned')).reset_index(drop=True)
    
    # Update node mapping to include only connected nodes
    all_proteins = set(top_k_edges['protein1']).union(set(top_k_edges['protein2']))
    node_mapping = {protein: idx for idx, protein in enumerate(all_proteins)}

    # Rebuild the edge list using pruned interactions
    edges = []
    for _, row in top_k_edges.iterrows():
        if row['protein1'] in node_mapping and row['protein2'] in node_mapping:
            edges.append([node_mapping[row['protein1']], node_mapping[row['protein2']]])

    # Create the final edge tensor as a sparse representation
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    graph_data = Data(edge_index=edges)

    # Load gene expression matrix and motif matrix with dimensionality reduction
    with h5py.File(HDF5_GENE_MATRIX_FILE, 'r') as f:
        gene_matrix_np = f['gene_matrix'][:].T

    with h5py.File(MOTIF_MATRIX_FILE, 'r') as f:
        motif_matrix_np = f['motif_matrix'][:].T

    # Apply dimensionality reduction (e.g., PCA)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=100)  # Reduce to 100 dimensions
    # gene_matrix_np = pca.fit_transform(gene_matrix_np)

    # Use GPU if available
    device = torch.device('cpu') # could actually be run on GPU 
    graph_data = graph_data.to(device)

    
    print(f"Gene matrix shape: {gene_matrix_np.shape}")
    print(f"Motif matrix shape: {motif_matrix_np.shape}")

    # Determine the number of samples to process
    num_total_samples = gene_matrix_np.shape[0]
    samples_to_process = num_total_samples if NUM_SAMPLES is None else min(NUM_SAMPLES, num_total_samples)
    
    # Precompute gene index map
    gene_index_map = {hgnc_symbol: idx for idx, hgnc_symbol in enumerate(ensembl_gene_dict.keys())}

    # Step 4: Process each sample and save X and Y into a single HDF5 file
    with h5py.File(OUTPUT_HDF5_FILE, 'w') as hdf_out:
        for sample_idx in range(samples_to_process):
            # print(f"Processing sample {sample_idx + 1}...")
            
            # Initialize node feature tensor
            node_features = torch.zeros((len(node_mapping), 1), device=device)

            # Assign fold change values to the node features for the current sample
            for hgnc_symbol, ensembl_peptides in ensembl_gene_dict.items():
                gene_idx = gene_index_map[hgnc_symbol]
                fold_change_value = gene_matrix_np[sample_idx, gene_idx]

                for ensembl_peptide_id in ensembl_peptides:
                    if ensembl_peptide_id in node_mapping:
                        node_idx = node_mapping[ensembl_peptide_id]
                        node_features[node_idx] = fold_change_value

            # Convert node features to NumPy array
            node_features_np = node_features.cpu().numpy()

            # Clear node_features tensor to free GPU memory
            del node_features
            torch.cuda.empty_cache()

            # Get the motif data for the current sample
            motif_data_np = motif_matrix_np[sample_idx, :]

            # Save data for this sample in the HDF5 file
            group = hdf_out.create_group(f'sample_{sample_idx + 1}')
            group.create_dataset('X', data=node_features_np)
            group.create_dataset('Y', data=motif_data_np)

            # Manually run garbage collection
            del node_features_np, motif_data_np
            gc.collect()

            if (sample_idx + 1) % 100 == 0:
                print(f"Sample {sample_idx + 1} saved.")

if __name__ == "__main__":
    main()
