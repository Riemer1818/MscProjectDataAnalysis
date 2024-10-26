library(Seurat)
library(dplyr)
library(Matrix)
library(rhdf5)

# Function to dynamically create output directory if needed
create_output_directory <- function(output_dir) {
    if (!dir.exists(output_dir)) {
        cat("Creating output directory:", output_dir, "\n")
        dir.create(output_dir, recursive = TRUE)
    } else {
        cat("Output directory exists:", output_dir, "\n")
    }
}

# Function to load a list of variable genes
load_variable_genes <- function(file_path) {
    cat("Loading variable genes from:", file_path, "\n")
    readRDS(file_path)
}

# Function to process Seurat object and extract gene and motif matrices
process_seurat_object <- function(rds_file, all_variable_genes, motif_file_path) {
    cat("Processing file:", rds_file, "\n")
    
    # Load and update Seurat object
    seurat_obj <- readRDS(rds_file)
    seurat_obj <- UpdateSeuratObject(seurat_obj)
    
    # Extract and filter gene expression matrix by variable genes
    gene_matrix <- GetAssayData(seurat_obj, assay = "RNA")
    gene_matrix <- gene_matrix[rownames(gene_matrix) %in% all_variable_genes, ]
    gene_matrix <- t(gene_matrix)
    
    # Prepare motif matrix based on motif enrichment data
    motif_matrix <- prepare_motif_matrix(seurat_obj, motif_file_path)
    
    return(list(
        gene_matrix = gene_matrix,
        motif_matrix = motif_matrix
    ))
}

# Function to prepare motif matrix without thresholding
prepare_motif_matrix <- function(seurat_obj, motif_file_path) {
    cat("Preparing motif matrix...\n")
    
    motif_data <- read.delim(motif_file_path, sep = "\t", header = TRUE)
    unique_motifs <- unique(motif_data$motif)
    
    # Initialize sparse matrix for motifs
    motif_matrix <- Matrix(0, nrow = nrow(seurat_obj@meta.data), ncol = length(unique_motifs), sparse = TRUE)
    colnames(motif_matrix) <- unique_motifs
    rownames(motif_matrix) <- rownames(seurat_obj@meta.data)
    
    # Precompute motifs by cell type for efficient access
    celltype_motif_map <- split(motif_data, motif_data$cell_type)
    
    # Populate motif matrix with fold-change values
    for (i in 1:nrow(seurat_obj@meta.data)) {
        celltype <- seurat_obj$cell_type[i]
        
        if (!is.na(celltype) && celltype %in% names(celltype_motif_map)) {
            celltype_motifs <- celltype_motif_map[[celltype]]
            if (nrow(celltype_motifs) > 0) {
                motif_matrix[i, celltype_motifs$motif] <- celltype_motifs$fold_change
            }
        }
    }
    
    return(as(motif_matrix, "sparseMatrix"))
}

# Function to save matrices to HDF5 files
save_matrices_to_hdf5 <- function(gene_matrix, motif_matrix, gene_matrix_output, motif_matrix_output) {
    cat("Saving gene matrix to:", gene_matrix_output, "\n")
    h5createFile(gene_matrix_output)
    h5write(as.matrix(gene_matrix), gene_matrix_output, "gene_matrix")
    
    cat("Saving motif matrix to:", motif_matrix_output, "\n")
    h5createFile(motif_matrix_output)
    h5write(as.matrix(motif_matrix), motif_matrix_output, "motif_matrix")
}

# Function to log statistics to a file
log_statistics <- function(final_gene_matrix, final_motif_matrix, log_file_path) {
    stats_message <- paste0(
        "Final Gene Matrix Dimensions: ", paste(dim(final_gene_matrix), collapse = " x "), "\n",
        "Final Motif Matrix Dimensions: ", paste(dim(final_motif_matrix), collapse = " x "), "\n"
    )
    
    cat(stats_message, file = log_file_path)
    cat("Statistics logged to:", log_file_path, "\n")
}

# Main function to process gene and motif matrices
process_gene_and_motif_matrices <- function(
    all_variable_genes_path, motif_file_path, mc_folder,
    gene_matrix_output, motif_matrix_output, log_file_path
) {
    # Create output directory if needed
    create_output_directory(dirname(gene_matrix_output))
    create_output_directory(dirname(motif_matrix_output))
    create_output_directory(dirname(log_file_path))
    
    # Load variable genes
    all_variable_genes <- load_variable_genes(all_variable_genes_path)
    
    # Initialize lists to store matrices
    gene_matrices_list <- list()
    motif_matrices_list <- list()
    
    # Process each RDS file
    rds_files <- list.files(mc_folder, pattern = "\\.RDS$", full.names = TRUE)
    for (rds_file in rds_files) {
        results <- process_seurat_object(rds_file, all_variable_genes, motif_file_path)
        
        gene_matrices_list[[rds_file]] <- results$gene_matrix
        motif_matrices_list[[rds_file]] <- results$motif_matrix
    }
    
    # Combine matrices across all files
    final_gene_matrix <- do.call(rbind, gene_matrices_list)
    final_motif_matrix <- do.call(rbind, motif_matrices_list)
    
    # Save matrices to HDF5 files
    save_matrices_to_hdf5(final_gene_matrix, final_motif_matrix, gene_matrix_output, motif_matrix_output)
    
    # Log statistics to a file
    log_statistics(final_gene_matrix, final_motif_matrix, log_file_path)
    
    cat("Process completed successfully.\n")
}

# Example usage
motif_file_path <- "~/data/GSE149683_File_S3.Motif_enrichments_across_cell_types.txt"
downlaod from GEO https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149683

all_variable_genes_path <-  "~/data/filtered_seurats/MC3/all_variable_genes.rds"

mc_folder <- "~/data/filtered_seurats/MC3/"
gene_matrix_output <- "~/data/gene_matrix.h5"
motif_matrix_output <- "~/data/motif_matrix.h5"
log_file_path <- "~/data/data_processing_stats.log"
# Call the main function
process_gene_and_motif_matrices(
    all_variable_genes_path, motif_file_path, mc_folder,
    gene_matrix_output, motif_matrix_output, log_file_path
)
