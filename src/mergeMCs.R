# Load necessary libraries
library(Seurat)
library(dplyr)

# Define the folder containing your RDS files
mc_folder <- "/home/msai/riemerpi001/data/filtered_seurats/MC/"
rds_files <- list.files(mc_folder, pattern = "\\.rds$", full.names = TRUE)

# Initialize a list to store counts matrices
counts_list <- list()

# Loop through each RDS file, extract the counts matrix, and store it in the list
for (i in seq_along(rds_files)) {
  # Load the Seurat object
  seurat_obj <- readRDS(rds_files[i])
  
  # Ensure the Seurat object is updated
  seurat_obj <- UpdateSeuratObject(seurat_obj)
  
  # Extract the counts matrix from the RNA assay
  counts_matrix <- GetAssayData(seurat_obj, slot = "counts", assay = "RNA")
  
  # Add the counts matrix to the list
  counts_list[[i]] <- counts_matrix
}

# Combine the counts matrices by columns
combined_counts <- do.call(cbind, counts_list)

# Rename columns to be unique
colnames(combined_counts) <- make.unique(colnames(combined_counts))

# Create a new Seurat object from the combined counts matrix
MC <- CreateSeuratObject(counts = combined_counts)

# Normalize the data
MC <- NormalizeData(MC)

MC <- FindVariableFeatures(MC, selection.method = "vst", nfeatures = 4500)

all_variable_genes <- VariableFeatures(MC, assay = "RNA")

# the only thing I actually care about lol
saveRDS(all_variable_genes, file = "/home/msai/riemerpi001/data/all_variable_genes.rds")

