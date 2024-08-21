
# print(R.version.string)

# install.packages("BiocManager", force= TRUE))
# install.packages("patchwork", force= TRUE)
# BiocManager::install("EnsDb.Hsapiens.v75", force= TRUE)
# BiocManager::install("Signac", force= TRUE)
# BiocManager::install("Seurat", force= TRUE)z
# BiocManager::install("hdf5r", force= TRUE)
# BiocManager::install("biovizBase", force= TRUE)
install.packages("Matrix")

library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
# library(hdf5r)
library(biovizBase)

# if (!requireNamespace("remotes")) install.packages("remotes")
# remotes::install_github("GfellerLab/SuperCell")
library(SuperCell)


# multiomic_data <- readRDS(file = "src/data/cerebellum_filtered.seurat.for_website.RDS")

multiomic_data <- readRDS(file = "/home/riemer/Downloads/RDS_downloads/spleen_filtered.seurat.for_website.RDS")
# multiomic_data <- readRDS(file = "/home/riemer/Downloads/RDS_downloads/cerebellum_filtered.seurat.for_website.RDS")



multiomic_data <- UpdateSeuratObject(multiomic_data)


# Assuming 'multiomic_data' is your Seurat object
set.seed(123)  # Set seed for reproducibility

# Randomly sample 500 cells
cell_subset <- sample(colnames(multiomic_data), size = 500)

# Subset the Seurat object to only include these cells
multiomic_data <- subset(multiomic_data, cells = cell_subset)


Assays(multiomic_data) # RNA and peaks 

multiomic_data <- SCTransform(multiomic_data, assay = "RNA", verbose = FALSE, useNames = TRUE) #useNames = NA is deprecated
multiomic_data <- RunTFIDF(multiomic_data)
multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = "q0")
multiomic_data <- RunSVD(multiomic_data)


expression_matrix <- multiomic_data@assays$RNA@data  # Extracting normalized expression data from Seurat object

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

# Plot UMAP with metacell groupings
# DimPlot(multiomic_data, label = TRUE, repel = FALSE, group.by = "metacell", reduction = "umap") + NoLegend()

print(length(multiomic_data$metacell)) # 3313 original cells
print(length(unique(multiomic_data$metacell))) # 166 out of an original 3313

# Plot the size of metacells
hist(metacells$supercell_size, breaks = 30, main = "Metacell Size Distribution", xlab = "Number of Cells per Metacell")

# Load the Matrix package
library(Matrix)

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