# Load necessary libraries
library(Seurat)
library(dplyr)
library(Matrix)
library(rhdf5)


# MAKING X and Y MATRIX

# Load the list of all variable genes
all_variable_genes <- readRDS("/home/msai/riemerpi001/data/all_variable_genes.rds")
file_path <- "/home/msai/riemerpi001/data/GSE149683_File_S3.Motif_enrichments_across_cell_types.txt"

mc_folder <- "/home/msai/riemerpi001/data/filtered_seurats/MC"
rds_files <- list.files(mc_folder, pattern = "\\.rds$", full.names = TRUE)

gene_matrix_path <- "/home/msai/riemerpi001/final_gene_matrix.h5"
motif_matrix_path <- "/home/msai/riemerpi001/final_motif_matrix.h5"

fold_change_threshhold = 0 # Only keep fold-change values >= 0.2

# Initialize empty lists to store matrices
gene_matrices_list <- list()
motif_matrices_list <- list()

# Loop through all RDS files and process each
for (rds_file in rds_files) {
  cat("Processing file:", rds_file, "\n")
  
  # Load the Seurat object
  dat <- readRDS(rds_file)
  
  # Update the Seurat object
  MC <- UpdateSeuratObject(dat)
  
  # Subset the gene expression matrix to only include the genes from all_variable_genes
  gene_matrix <- GetAssayData(MC, assay = "RNA")
  gene_matrix <- gene_matrix[rownames(gene_matrix) %in% all_variable_genes, ]
  
  # Transpose the matrix to have cells as rows and genes as columns
  gene_matrix <- t(gene_matrix)
  
  # Store the matrix in the list
  gene_matrices_list[[rds_file]] <- gene_matrix
  
  # Load the motif data
  df <- read.delim(file_path, sep = "\t", header = TRUE)
  
  # Get the unique motifs (this will determine the number of features/columns)
  unique_motifs <- unique(df$motif)
  
  # Initialize a sparse matrix for storing the fold-change values (rows are cells, columns are motifs)
  motif_matrix <- Matrix(0, nrow = nrow(MC@meta.data), ncol = length(unique_motifs), sparse = FALSE)
  colnames(motif_matrix) <- unique_motifs
  rownames(motif_matrix) <- rownames(MC@meta.data)
  
  # Precompute the motif list for each cell type
  celltype_motif_map <- split(df, df$cell_type)
  
  # Now loop through each cell and fill the matrix
  for (i in 1:nrow(MC@meta.data)) {
    celltype <- MC$celltype[i]
    
    # Check if the celltype exists in the precomputed list
    if (!is.na(celltype) && celltype %in% names(celltype_motif_map)) {
      celltype_motifs <- celltype_motif_map[[celltype]]
      
      # Apply the condition: only keep fold-change values >= 0.2
      valid_motifs <- celltype_motifs[celltype_motifs$fold_change >= fold_change_threshhold, ]
      
      # Vectorized assignment of fold-change values to the corresponding motifs
      if (nrow(valid_motifs) > 0) {
        motif_matrix[i, valid_motifs$motif] <- valid_motifs$fold_change
      }
    }
  }
  
  # Convert the motif matrix into a sparse matrix and store it in the list
  motif_matrix_sparse <- as(motif_matrix, "sparseMatrix")
  motif_matrices_list[[rds_file]] <- motif_matrix_sparse
}

# Concatenate all gene matrices and motif matrices into a single large matrix
final_gene_matrix <- do.call(rbind, gene_matrices_list)
final_motif_matrix <- do.call(rbind, motif_matrices_list)

# Check the dimensions of the final matrices
cat("Final Gene Matrix Dimensions:", dim(final_gene_matrix), "\n")
cat("Final Motif Matrix Dimensions:", dim(final_motif_matrix), "\n")

# Save the final gene matrix as HDF5
h5createFile(gene_matrix_path)
h5write(as.matrix(final_gene_matrix), gene_matrix_path, "gene_matrix")

# Save the final motif matrix as HDF5
h5createFile(motif_matrix_path)
h5write(as.matrix(final_motif_matrix), motif_matrix_path, "motif_matrix")
