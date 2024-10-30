library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
# library(hdf5r)
library(biovizBase)
library(dplyr)
library(SuperCell)

# Retrieve file path from system argument
file_path <- commandArgs(trailingOnly = TRUE)[1]

# Extract the basename of the file without extension
input_file_base <- tools::file_path_sans_ext(basename(file_path))

# Create the output directory based on the basename + "MC"


output_dir <- file.path("~/RIEMERPI001/data/filteredSeurats/MC")
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Load the multiomic data
multiomic_data <- readRDS(file = file_path)
sc <- UpdateSeuratObject(multiomic_data)
rm(multiomic_data)

# Normalize and find variable features
sc <- NormalizeData(sc, verbose=FALSE)
sc <- FindVariableFeatures(sc, selection.method = "disp", nfeatures = 1000, verbose=FALSE)
hvg <- VariableFeatures(sc)

# Plot variable features
plot1 <- VariableFeaturePlot(sc)
LabelPoints(plot = plot1, points = hvg[1:20], repel = TRUE)

# Scaling, PCA, UMAP
sc <- ScaleData(sc, verbose=FALSE)
sc <- RunPCA(sc, verbose=FALSE)
DimPlot(sc, reduction = "pca", group.by = "cell_type")

sc <- RunUMAP(sc, dims = 1:10)
DimPlot(sc, reduction = "umap", group.by = "cell_type")

# Clustering
sc <- FindNeighbors(sc, dims = 1:10)
sc <- FindClusters(sc, resolution = 0.05)
DimPlot(sc, reduction = "umap", group.by = "ident")

# Reassign identities and sort levels
Idents(sc) <- "cell_type" 
levels(sc) <- sort(levels(sc))

# Get PCA embedding and perform metacell construction
pcal2 <- Embeddings(sc, reduction = "pca.l2.harmony")

MC <- SCimplify_from_embedding(
  X = pcal2, 
  cell.annotation = sc$cell_type,
  gamma = 20
)

# Compute gene expression of metacells
DefaultAssay(sc) <- "RNA"
MC.ge <- supercell_GE(ge = GetAssayData(sc), mode = "sum", groups = MC$membership)

DefaultAssay(sc) <- "peaks"
MC.peaks <- supercell_GE(ge = GetAssayData(sc), mode = "sum", groups = MC$membership)

colnames(MC.peaks) <- colnames(MC.ge) <- paste0("MC_", seq_len(length(unique(MC$membership))))

# Create Seurat object for metacells
mc_seu <- CreateSeuratObject(counts = MC.ge, assay = "RNA")

mcp <- CreateAssayObject(MC.peaks)
mc_seu[["peaks"]] <- mcp

# Process metacells
mc_seu <- NormalizeData(mc_seu)
mc_seu <- FindVariableFeatures(mc_seu)
mc_seu <- ScaleData(mc_seu, features = VariableFeatures(mc_seu))
mc_seu <- RunPCA(mc_seu, npcs = 50)
mc_seu <- RunUMAP(mc_seu, dims = 1:30)

# Annotate cell types
celltypes <- MC$SC.cell.annotation.
names(celltypes) <- paste0("MC_", names(celltypes))
mc_seu$celltype <- celltypes[colnames(mc_seu)]

# Plot UMAP of metacells
DimPlot(mc_seu, group.by = "celltype")

# Save UMAP plot
# output_plot <- paste0(input_file_base, "_UMAP.png")
# output_plot <- basename(output_plot)
# ggsave(file.path(output_dir, output_plot), plot = last_plot(), width = 20, height = 16)

# Save Seurat object

output_file <- paste0(input_file_base, "_MC.rds")

output_file <- basename(output_file)

saveRDS(mc_seu, file = file.path(output_dir, output_file))

quit(save = "no")
