# Load required libraries
library(Seurat)
library(Matrix)
library(data.table)
library(dplyr)
library(rhdf5)

# Function to dynamically create the output directory if needed
create_output_directory <- function(output_dir) {
    if (!dir.exists(output_dir)) {
        cat("Creating output directory:", output_dir, "\n")
        dir.create(output_dir, recursive = TRUE)
    } else {
        cat("Output directory exists:", output_dir, "\n")
    }
}


h5_dataset_exists <- function(file, dataset_name) {
    if (!file.exists(file)) {
        return(FALSE)
    }
    datasets <- h5ls(file)
    return(dataset_name %in% datasets$name)
}

# Main processing function for multiple Seurat files
process_multiple_files <- function(rds_files, all_variable_genes_path, motif_file_path, gene_matrix_output, motif_matrix_output) {
    # Load variable genes and motif data
    all_variable_genes <- readRDS(all_variable_genes_path)
    motif_data <- fread(motif_file_path)

    # Create output directory if needed
    create_output_directory(dirname(gene_matrix_output))
    create_output_directory(dirname(motif_matrix_output))

    for (rds_file in rds_files) {
        cat("Processing file:", rds_file, "\n")

        # Load the Seurat object
        seurat_obj <- readRDS(rds_file)
        seurat_obj <- UpdateSeuratObject(seurat_obj)

        # Filter and transpose gene matrix
        gene_matrix <- GetAssayData(seurat_obj, assay = "RNA", slot = "counts")
        gene_matrix <- gene_matrix[rownames(gene_matrix) %in% all_variable_genes, ]
        gene_matrix <- t(gene_matrix)

        # Normalize the data if not already done
        if (!"data" %in% slotNames(GetAssay(seurat_obj, assay = "RNA"))) {
            cat("Normalizing data...\n")
            seurat_obj <- NormalizeData(seurat_obj, assay = "RNA")
        }

        # Prepare the motif matrix
        cell_types <- unique(seurat_obj$celltype)
        motif_data_filtered <- motif_data[cell_type %in% cell_types, ]

        # Convert motif data and cell type data to data.table for faster processing
        motif_data_filtered_dt <- as.data.table(motif_data_filtered)
        setkey(motif_data_filtered_dt, cell_type)

        # Convert cell types from Seurat object to a data.table
        cell_types_dt <- data.table(cell_id = rownames(seurat_obj@meta.data), cell_type = seurat_obj$celltype)

        # Perform a join to get the motifs for each cell based on the cell type, allowing cartesian join
        merged_data <- cell_types_dt[motif_data_filtered_dt, on = "cell_type", nomatch = 0, allow.cartesian = TRUE]

        # Get unique motifs for setting the column names of the sparse matrix
        unique_motifs <- unique(motif_data_filtered$motif)

        # Create the sparse motif matrix based on the merged data
        motif_matrix <- sparseMatrix(
            i = as.integer(factor(merged_data$cell_id, levels = rownames(seurat_obj@meta.data))),
            j = as.integer(factor(merged_data$motif, levels = unique_motifs)),
            x = merged_data$fold_change,
            dims = c(nrow(seurat_obj@meta.data), length(unique_motifs)),
            dimnames = list(rownames(seurat_obj@meta.data), unique_motifs)
        )

        # Append the gene_matrix to gene_matrices
        if (!exists("gene_matrices")) {
            gene_matrices <- list()
        }
        gene_matrices[[rds_file]] <- gene_matrix


        # Append the gene_matrix to gene_matrices
        if (!exists("motif_matrices")) {
            motif_matrices <- list()
        }
        motif_matrices[[rds_file]] <- motif_matrix

        cat("Finished processing file:", rds_file, "\n")
    }
    # Write the gene and motif matrices to HDF5 files in one go
    cat("Writing gene matrices to HDF5 file...\n")
    gene_matrix <- do.call(rbind, gene_matrices)
    gene_matrix <- as(gene_matrix, "sparseMatrix")
    h5createFile(gene_matrix_output)
    h5write(as.matrix(gene_matrix), gene_matrix_output, "gene_matrix")

    cat("Writing motif matrices to HDF5 file...\n")
    motif_matrix <- do.call(rbind, motif_matrices)
    motif_matrix <- as(motif_matrix, "sparseMatrix")
    h5createFile(motif_matrix_output)
    h5write(as.matrix(motif_matrix), motif_matrix_output, "motif_matrix")

    cat("All files processed successfully. Data saved to HDF5 files.\n")
}

# Process each RDS file, ignoring the specified file
rds_files <- list.files("~/data/filtered_seurats/MC3/", pattern = "\\.rds$", full.names = TRUE)

# Exclude the file "all_variable_genes.rds"
rds_files <- rds_files[!grepl("all_variable_genes\\.rds$", rds_files)]


all_variable_genes_path <- "~/data/filtered_seurats/MC3/all_variable_genes.rds"
motif_file_path <- "~/data/GSE149683_File_S3.Motif_enrichments_across_cell_types.txt"
gene_matrix_output <- "~/data/filtered_seurats/MC3/processed_data/gene_matrix.h5"
motif_matrix_output <- "~/data/filtered_seurats/MC3/processed_data/motif_matrix.h5"

# Run the main processing function
process_multiple_files(rds_files, all_variable_genes_path, motif_file_path, gene_matrix_output, motif_matrix_output)
