import h5py
import torch
import torch_geometric as tg
import pandas as pd
import os
import numpy as np
import gc
import argparse
from torch_geometric.data import Data

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process gene and motif data to create a graph dataset.")
    parser.add_argument("--threshold", type=float, default=0.001, help="Threshold for binning weak edges.")
    parser.add_argument("--num_bins", type=int, default=50, help="Number of bins for edge weight binning.")
    parser.add_argument("--hdf5_gene_matrix_file", type=str, required=True, help="Path to HDF5 file with gene matrix.")
    parser.add_argument("--motif_matrix_file", type=str, required=True, help="Path to HDF5 file with motif matrix.")
    parser.add_argument("--ensembl_gene_file", type=str, required=True, help="Path to Ensembl gene mapping CSV file.")
    parser.add_argument("--output_hdf5_file", type=str, required=True, help="Path to output HDF5 file.")
    parser.add_argument("--interaction_data_file", type=str, required=True, help="Path to interaction data file (gzip).")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process.")
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # ======================
    # CONFIGURATION
    # ======================

    # Set parameters from arguments
    THRESHOLD = args.threshold
    NUM_BINS = args.num_bins
    HDF5_GENE_MATRIX_FILE = args.hdf5_gene_matrix_file
    MOTIF_MATRIX_FILE = args.motif_matrix_file
    ENSEMBL_GENE_FILE = args.ensembl_gene_file
    OUTPUT_HDF5_FILE = args.output_hdf5_file
    INTERACTION_DATA_FILE = args.interaction_data_file
    NUM_SAMPLES = args.num_samples

    # Check if GPU is available
    USE_GPU = torch.cuda.is_available()

    # Step 1: Load gene data
    ensembl_gene_df = pd.read_csv(ENSEMBL_GENE_FILE)
    ensembl_gene_dict = ensembl_gene_df.groupby('hgnc_symbol')['ensembl_peptide_id'].apply(list).to_dict()
    genes_of_interest = set(ensembl_gene_df['ensembl_peptide_id'])

    # Step 2: Load interaction data and build the graph
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

    # Expand the node mapping to include all unique proteins in interaction data
    all_proteins = set(interaction_data['protein1']).union(set(interaction_data['protein2']))
    node_mapping = {protein: idx for idx, protein in enumerate(all_proteins)}

    # Print the number of nodes in the final node mapping
    print(f"Total number of nodes in the expanded node mapping: {len(node_mapping)}")

    # Rebuild the edge list
    edges = []
    for _, row in binned_interactions.iterrows():
        if row['protein1'] in node_mapping and row['protein2'] in node_mapping:
            edges.append([node_mapping[row['protein1']], node_mapping[row['protein2']]])

    # Create the final edge tensor
    edges = torch.tensor(edges).t().contiguous()
    graph_data = Data(edge_index=edges)

    # Step 3: Load the gene expression matrix (X) and motif matrix (Y)
    with h5py.File(HDF5_GENE_MATRIX_FILE, 'r') as f:
        gene_matrix_np = f['gene_matrix'][:].T  # X: Gene expression data

    with h5py.File(MOTIF_MATRIX_FILE, 'r') as f:
        motif_matrix_np = f['motif_matrix'][:].T # Y: Motif matrix data

    print(f"Gene matrix shape: {gene_matrix_np.shape}")
    print(f"Motif matrix shape: {motif_matrix_np.shape}")

    # Determine the number of samples to process
    num_total_samples = gene_matrix_np.shape[0]
    samples_to_process = num_total_samples if NUM_SAMPLES is None else min(NUM_SAMPLES, num_total_samples)

    # Use GPU if available
    device = torch.device('cuda' if USE_GPU else 'cpu')
    graph_data = graph_data.to(device)

    # Precompute gene index map
    gene_index_map = {hgnc_symbol: idx for idx, hgnc_symbol in enumerate(ensembl_gene_dict.keys())}

    # Step 4: Process each sample and save X and Y into a single HDF5 file
    with h5py.File(OUTPUT_HDF5_FILE, 'w') as hdf_out:
        for sample_idx in range(samples_to_process):
            print(f"Processing sample {sample_idx + 1}...")
            
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

            print(f"Sample {sample_idx + 1} saved.")

if __name__ == "__main__":
    main()
