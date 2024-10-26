library(Seurat)
library(dplyr)
library(tools)

# Function to dynamically create output directories if they don't exist
create_output_directory <- function(output_dir) {
    if (!dir.exists(output_dir)) {
        cat("Creating output directory:", output_dir, "\n")
        dir.create(output_dir, recursive = TRUE)
    } else {
        cat("Output directory exists:", output_dir, "\n")
    }
}

# Main function to process RDS files and extract variable features
process_and_extract_variable_genes <- function(mc_folder, output_dir, nfeatures = 4500) {
    # Create the output directory if needed
    create_output_directory(output_dir)
    
    # List all RDS files in the specified folder
    rds_files <- list.files(mc_folder, pattern = "\\.rds$", full.names = TRUE)
    
    # Initialize a list to store counts matrices
    counts_list <- list()
    
    # Loop through each RDS file and process it
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
    
    # Find variable features
    MC <- FindVariableFeatures(MC, selection.method = "vst", nfeatures = nfeatures)
    
    # Extract the variable features
    all_variable_genes <- VariableFeatures(MC, assay = "RNA")
    
    # Define the output file path
    output_file <- file.path(output_dir, "all_variable_genes.rds")
    
    # Save the variable genes to an RDS file
    cat("Saving variable genes to:", output_file, "\n")
    saveRDS(all_variable_genes, file = output_file)
}

# Example usage
mc_folder <- "~/data/filtered_seurats/MC3"
output_dir <- "~/data/filtered_seurats/data/output_variable_genesM3"
nfeatures <- 4500

# Call the main function
process_and_extract_variable_genes(mc_folder, output_dir, nfeatures)
