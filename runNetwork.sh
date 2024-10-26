#!/bin/bash
#SBATCH --partition=M2
#SBATCH --qos=q_a_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --job-name=runMetaCells
#SBATCH --output=./output/output_%j.out 
#SBATCH --error=./error/error_%j.err
#SBATCH --time=0-06:00:00

module load anaconda
eval "$(conda shell.bash hook)"
conda activate base

# make network
python ~/MscProjectDataAnalysis/src/makeNetwork.py --theshold 0.001 --num_bins 50 --hdf5_gene_matrix_file ~/data/gene_matrix.h5 --motif_matrix_file ~/data/motif_matrix.h5 --ensembl_gene_file ~/data/filtered_seurats/MC3/ensembl_protein_ids_cleaned.csv --output_hdf5_file ~/data/filtered_seurats/MC3/graph_data.h5 --interaction_data_file ~/data/GSE149683_File_S3.Motif_enrichments_across_cell_types.txt --num_samples 1000

