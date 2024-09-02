library(Seurat)
library(ggplot2)
library(patchwork)
library(Signac)
library(EnsDb.Hsapiens.v75)
library(hdf5r)
library(biovizBase)
library(Matrix)

output_dir <- "~/data/output/"

# Load the Seurat object
file_path <- "~/data/GSM4508930_cerebrum_filtered.seurat.RDS"
multiomic_data <- readRDS(file = file_path)

# Update the Seurat object if necessary
multiomic_data <- UpdateSeuratObject(multiomic_data)

# Set seed for reproducibility
set.seed(123)

# Randomly sample 5,000 cells from the Seurat object
cell_subset <- sample(colnames(multiomic_data), size = 5000)

# Subset the Seurat object to include only the selected cells
multiomic_data <- subset(multiomic_data, cells = cell_subset)

# Print the available assays in the Seurat object
print(Assays(multiomic_data)) # RNA and peaks 

# Perform SCTransform on the RNA assay
multiomic_data <- SCTransform(multiomic_data, assay = "RNA", verbose = FALSE, useNames = TRUE)

# Run TF-IDF transformation for the peaks assay
multiomic_data <- RunTFIDF(multiomic_data)

# Find top features
multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = "q0")

# Run SVD for the peaks assay
multiomic_data <- RunSVD(multiomic_data)

# Perform PCA for RNA and LSI for peaks
multiomic_data <- RunPCA(multiomic_data, assay = "RNA")
multiomic_data <- RunLSI(multiomic_data, assay = "peaks")

# Find Transfer Anchors for Integration
transfer_anchors <- FindTransferAnchors(
    reference = multiomic_data, 
    query = multiomic_data,
    reduction = "cca",
    dims = 1:30
)

# seems to find transfors anchors for the same object... so that's "positive" 17011 anchors.
celltype.predictions <- TransferData(anchorset = transfer_anchors, refdata = multiomic_data@meta.data$cell_type,  weight.reduction = multiomic_data[["pca"]], dims = 1:30)
multiomic_data <- AddMetaData(multiomic_data, metadata = celltype.predictions)

# Save the histogram plot
hist(multiomic_data$prediction.score.max)
file_name <- paste0(basename(file_path), "_histogram.png")
output_path <- file.path(output_dir, file_name)
ggsave(output_path, plot = last_plot(), width = 16, height = 12, dpi = 300)

# filter on prediction score
multiomic_data.filtered <- subset(multiomic_data, subset = prediction.score.max > 0.5)
multiomic_data.filtered$predicted.id <- factor(multiomic_data.filtered$predicted.id, levels = levels(multiomic_data))  # to make the colors match

p1 <- DimPlot(multiomic_data.filtered, group.by = "predicted.id", label = TRUE, repel = TRUE) + ggtitle("scATAC-seq cells") + 
    NoLegend() + scale_colour_hue(drop = FALSE)

p2 <- DimPlot(multiomic_data, group.by = "cell_type", label = TRUE, repel = TRUE) + ggtitle("scRNA-seq cells") + 
    NoLegend()
    
# Save the combined plot
file_name <- paste0(basename(file_path), "_combined_plot.png")
output_path <- file.path(output_dir, file_name)
ggsave(output_path, plot = CombinePlots(plots = list(p1, p2)), width = 16, height = 12, dpi = 300)

# now I want to do the co-embedding of the two datasets for a visual indication of the (dimension reduced) similarity between the two datasets.

# # all the RNA cells that have a prediction score > 0.5
# multiomic_data.filtered 

# # and their original peaks 
# multiomic_data.filtered@assays$peaks@counts

# # and their original RNA
# multiomic_data.filtered@assays$RNA@counts

# x = multiomic_data.filtered@assays$peaks@counts
# y = multiomic_data.filtered@assays$RNA@counts

# # coembed <- merge(x, y, by = intersect(names(x), names(y)), all = FALSE)
# coembed <- merge(x, y) # takes forever because of densing 

# # Finally, we run PCA and UMAP on this combined object, to visualize the co-embedding of both
# # datasets
# coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
# coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
# coembed <- RunUMAP(coembed, dims = 1:30)
# coembed$celltype <- ifelse(!is.na(coembed$celltype), coembed$celltype, coembed$predicted.id)

# DimPlot(coembed, group.by = "celltype", label = TRUE, repel = TRUE) + NoLegend() + ggtitle("Co-embedding of scRNA-seq and scATAC-seq cells")

