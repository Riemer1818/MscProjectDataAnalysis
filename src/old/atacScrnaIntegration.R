library(Seurat)
library(ggplot2)
library(patchwork)
library(Signac)
library(EnsDb.Hsapiens.v75)
library(hdf5r)
library(biovizBase)
library(SuperCell)
library(Matrix)


# Read the Seurat object from a file (you can pass the file path as a system argument)
# file_path <- commandArgs(trailingOnly = TRUE)[1]
file_path <- "~/data/GSM4508930_cerebrum_filtered.seurat.RDS"

multiomic_data <- readRDS(file = file_path)

# Update the Seurat object
multiomic_data <- UpdateSeuratObject(multiomic_data)

# Assuming 'multiomic_data' is your Seurat object
# set.seed(123)  # Set seed for reproducibility

# Randomly sample 100 cells
# cell_subset <- sample(colnames(multiomic_data), size = 100)

# Subset the Seurat object to only include these cells
# multiomic_data <- subset(multiomic_data, cells = cell_subset)

# Print the available assays in the Seurat object
Assays(multiomic_data) # RNA and peaks 

# Perform SCTransform on the RNA assay
multiomic_data <- SCTransform(multiomic_data, assay = "RNA", verbose = FALSE, useNames = TRUE) #useNames = NA is deprecated

# Run TF-IDF transformation
multiomic_data <- RunTFIDF(multiomic_data)

# Find top features
multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = "q0")

# Run SVD
multiomic_data <- RunSVD(multiomic_data)

# Extract normalized expression data from Seurat object
expression_matrix <- multiomic_data@assays$RNA@data

# Generate metacells
metacells <- SCimplify(
    expression_matrix,  
    gamma = 20,         
    n.var.genes = 1000, 
    k.knn = 10,         
    n.pc = 20           
)

# Extract the metacell membership and name it according to the Seurat object
metacell_membership <- metacells$membership
names(metacell_membership) <- colnames(multiomic_data)

# Add metacell membership as metadata to Seurat object
multiomic_data <- AddMetaData(multiomic_data, metadata = metacell_membership, col.name = "metacell")

# Get the filename without the path and extension
filename <- basename(file_path)
filename <- tools::file_path_sans_ext(filename)

# Plot UMAP with metacell groupings
plot <- DimPlot(multiomic_data, label = TRUE, repel = FALSE, group.by = "metacell", reduction = "umap") + NoLegend()

# Save the plot with the filename
ggsave(filename = paste0("data/output/plots/", filename, "_metacell_umap_plot.png"), plot = plot, dpi = 300)

# Plot the size of metacells
hist(metacells$supercell_size, breaks = 30, main = "Metacell Size Distribution", xlab = "Number of Cells per Metacell")

# Save the histogram plot
ggsave(filename = paste0("data/output/plots/", filename, "_metacell_size_distribution.png"), plot = last_plot(), dpi = 300)
# Convert the original RNA data to a sparse matrix
rna_sparse_matrix <- Matrix(multiomic_data@assays$RNA@data, sparse = TRUE)

# Convert the original peaks data to a sparse matrix
peaks_sparse_matrix <- Matrix(multiomic_data@assays$peaks@data, sparse = TRUE)


# Generate the RNA metacell matrix using the sparse RNA matrix
rna_metacell_matrix <- supercell_GE(
    rna_sparse_matrix,         # Use the sparse RNA expression matrix
    groups = metacells$membership,   # Metacell membership vector
    mode = "average"                 # Aggregating the expression by averaging
)

# Generate the Peaks metacell matrix using the sparse Peaks matrix
peaks_metacell_matrix <- supercell_GE(
    ge = peaks_sparse_matrix,       # Use the sparse Peaks expression matrix
    groups = metacells$membership,   # Metacell membership vector
    mode = "average"                 # Aggregating the expression by averaging
)


metacell_ids <- unique(metacells$membership)
# Assign column names to the metacell matrix
colnames(rna_metacell_matrix) <- metacell_ids
colnames(peaks_metacell_matrix) <- metacell_ids
# Check the column names
colnames(rna_metacell_matrix)
colnames(peaks_metacell_matrix)


# Create a new Seurat object with the RNA metacell matrix
metacell_seurat <- CreateSeuratObject(counts = rna_metacell_matrix, assay = "RNA")

# Add the Peaks assay to the Seurat object using the sparse matrix
metacell_seurat[["peaks"]] <- CreateAssayObject(counts = peaks_metacell_matrix)

# runs till here easily. The next part is the integration part and I might have to specify the assays to integrate. 



transfer_anchors <- FindTransferAnchors(
    reference = metacell_seurat, 
    query = metacell_seurat,
    reduction = "cca",
    dims = 1:30
)

# Integrate the datasets
new_data <- IntegrateData(anchorset = transfer_anchors)

# build a joint neighbor graph using both assays
new_data <- FindMultiModalNeighbors(
    object = new_data,
    reduction.list = list("pca", "lsi"),
    dims.list = list(1:50, 2:40),
    modality.weight.name = "RNA.weight",
    verbose = TRUE
)

# build a joint UMAP visualization
out <- RunUMAP(
    object = new_data,
    nn.name = "weighted.nn",
    assay = "RNA",
    verbose = TRUE
)

DimPlot(out, label = TRUE, repel = TRUE, reduction = "umap") + NoLegend()

# Save the UMAP plot
ggsave(filename = paste0("data/output/plots/", filename, "_joint_umap_plot.png"), plot = last_plot(), dpi = 300)

# Save the new dataset
saveRDS(new_data, file = paste0("data/output/datasets/", filename, "_new_data.rds"))
