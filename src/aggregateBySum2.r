library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
library(biovizBase)
library(dplyr)
library(SuperCell)

# Output directory for saving metacell Seurat object
output_dir <- file.path("~/data/filtered_seurats")

# Retrieve file path from system argument or uncomment to test directly
file_path <- commandArgs(trailingOnly = TRUE)[1]
# file_path <- "data/filtered_seurats/GSM4508928_adrenal_filtered.seurat.RDS"

# Extract the basename of the file without extension
input_file_base <- tools::file_path_sans_ext(basename(file_path))

# Check if the output directory exists, create if not
if (!dir.exists(output_dir)) {
    cat("Creating output directory:", output_dir, "\n")
    dir.create(output_dir, recursive = TRUE)
} else {
    cat("Output directory exists:", output_dir, "\n")
}

# Load the multiomic data
cat("Loading multiomic data from:", file_path, "\n")
multiomic_data <- readRDS(file = file_path)
sc <- UpdateSeuratObject(multiomic_data)
rm(multiomic_data)

# Normalize and find variable features
sc <- NormalizeData(sc, verbose=FALSE)
sc <- FindVariableFeatures(sc, selection.method = "disp", nfeatures = 1000, verbose=FALSE)
hvg <- VariableFeatures(sc)

# Scaling, PCA, UMAP
sc <- ScaleData(sc, verbose=FALSE)
sc <- RunPCA(sc, verbose=FALSE)
sc <- RunUMAP(sc, dims = 1:10)

# Clustering
sc <- FindNeighbors(sc, dims = 1:10)
sc <- FindClusters(sc, resolution = 0.05)

# Reassign identities and sort levels
if ("cell_type" %in% colnames(sc@meta.data)) {
    Idents(sc) <- "cell_type"
} else {
    cat("Warning: 'cell_type' column not found in metadata. Skipping identity reassignment.\n")
}

# Get PCA embedding and perform metacell construction
pcal2 <- Embeddings(sc, reduction = "pca")  # Ensure 'pca' exists

MC <- SCimplify_from_embedding(
  X = pcal2, 
  cell.annotation = sc$cell_type,
  gamma = 10
)

# Compute gene expression of metacells
DefaultAssay(sc) <- "RNA"
MC.ge <- supercell_GE(ge = GetAssayData(sc), mode = "sum", groups = MC$membership)

# Ensure column names are set correctly
colnames(MC.ge) <- paste0("MC_", seq_len(length(unique(MC$membership))))

# Create Seurat object for metacells with only RNA counts
mc_seu <- CreateSeuratObject(counts = MC.ge, assay = "RNA")

# Annotate cell types
celltypes <- MC$SC.cell.annotation
names(celltypes) <- paste0("MC_", names(celltypes))
mc_seu$celltype <- celltypes[colnames(mc_seu)]

# Define output file path
output_file <- paste0(input_file_base, "_MC.rds")
full_output_path <- file.path(output_dir, output_file)

# Print the full output path to confirm the file location
cat("Saving Seurat object to:", full_output_path, "\n")

# Save Seurat object
tryCatch({
    saveRDS(mc_seu, file = full_output_path)
    if (file.exists(full_output_path)) {
        cat("File saved successfully at:", full_output_path, "\n")
    } else {
        cat("File not found after saving attempt at:", full_output_path, "\n")
    }
}, error = function(e) {
    cat("Error during saveRDS:", e$message, "\n")
})

# End the script
# quit(save = "no")
