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
conda activate my_r_env

# Example usage
# rds_folder <- "~/data/filtered_seurats/originalRDS"
# output_dir_base <- "~/data/filtered_seurats/MC3"
# cell_type_distribution_file <- "~/combined_cell_type_distribution_MC23_gamma40.csv"
# gamma <- 40
Rscript ~/MscProjectDataAnalysis/src/makeMetaCells.r

# Example usage
# mc_folder <- "~/data/filtered_seurats/MC3"
# output_dir <- "~/data/filtered_seurats/data/output_variable_genesM3"
# nfeatures <- 4500
Rscript ~/MscProjectDataAnalysis/src/getVariableGenes.r

# Example usage
# gene_list_path <- "~/data/filtered_seurats/MC3/all_variable_genes.rds"
# output_file_path <- "~data/filtered_seurats/MC3/ensembl_protein_ids_cleaned.csv"
# output_log_path <- "~/data/filtered_seurats/MC3/ensembl_protein_ids_stats.log"
Rscript ~/MscProjectDataAnalysis/src/mapToEnsembl.r


# Example usage
# motif_file_path <- "~/data/GSE149683_File_S3.Motif_enrichments_across_cell_types.txt"
# download from GEO https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149683

# all_variable_genes_path <-  "~/data/filtered_seurats/MC3/all_variable_genes.rds"
# mc_folder <- "~/data/filtered_seurats/MC3/"
# gene_matrix_output <- "~/data/gene_matrix.h5"
# motif_matrix_output <- "~/data/motif_matrix.h5"
# log_file_path <- "~/data/data_processing_stats.log"
Rscript ~/MscProjectDataAnalysis/src/makeH5adFiles.r`

