library(Seurat)
library(ggplot2)
library(patchwork)
library(Signac)
library(EnsDb.Hsapiens.v75)
library(hdf5r)
library(biovizBase)
library(Matrix)

output_dir <- "./output/"

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

# # Find top features
# multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = "q0")
# Find top features with a low min.cutoff
multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = 0.01)

# Run SVD for the peaks assay
multiomic_data <- RunSVD(multiomic_data)

# Perform PCA for RNA and LSI for peaks
multiomic_data <- RunPCA(multiomic_data, assay = "RNA")
# multiomic_data <- RunLSI(multiomic_data, assay = "peaks") # not sure if this ever run ? . the function doesn't exist in seurat4+

# Find Transfer Anchors for Integration
transfer_anchors <- FindTransferAnchors(
    reference = multiomic_data, 
    query = multiomic_data,
    reduction = "cca",
    dims = 1:30
)

# seems to find transfors anchors for the same object... so that's "positive" 17011 anchors.
celltype.predictions <- TransferData(anchorset = transfer_anchors, refdata = multiomic_data@meta.data$cell_type,  weight.reduction = multiomic_data[["pca"]], dims = 1:30)
# Define celltype.predictions

# Convert predicted cell types to factors
celltype.predictions$predicted.id <- as.factor(celltype.predictions$predicted.id)

# Extract the predicted cell types as a vector
predicted_cell_types <- celltype.predictions$predicted.id

# Print the distribution of predicted cell types as a histogram using ggplot2
p <- ggplot(data = celltype.predictions, aes(x = predicted.id)) +
  geom_bar() +
  xlab("Predicted Cell Types") +
  ylab("Frequency") +
  ggtitle("Distribution of Predicted Cell Types") +
  theme_minimal()

# Save the histogram plot
file_name <- paste0(basename(file_path), "_celltype_histogram.png")
output_path <- file.path(output_dir, file_name)
ggsave(output_path, plot = p, width = 16, height = 12, dpi = 300)

# Plot the prediction score as a histogram
prediction_score <- celltype.predictions$prediction.score.max

histogram <- ggplot(data = celltype.predictions, aes(x = prediction_score)) +
    geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black") +
    xlab("Prediction Score") +
    ylab("Frequency") +
    ggtitle("Distribution of Prediction Scores") +
    theme_minimal()

# Save the histogram plot
file_name <- paste0(basename(file_path), "_prediction_score_histogram.png")
output_path <- file.path(output_dir, file_name)
ggsave(output_path, plot = histogram, width = 16, height = 12, dpi = 300)

# filter on prediction score


# // from here on it isn't correct anymore. I need to sort out the two different DimPlots. 

# Subset the multiomic_data based on the prediction score

predicted_cell_types.filtered <- subset(celltype.predictions, subset = prediction_score > 0.5)

# multiomic_data.filtered$predicted.id <- factor(multiomic_data.filtered$predicted.id, levels = levels(multiomic_data))  # to make the colors match

p1 <- DimPlot(predicted_cell_types.filtered, group.by = "predicted.id", label = TRUE, repel = TRUE) + ggtitle("scATAC-seq cells") + 
    NoLegend() + scale_colour_hue(drop = FALSE)

p2 <- DimPlot(multiomic_data, group.by = "cell_type", label = TRUE, repel = TRUE) + ggtitle("scRNA-seq cells") + 
    NoLegend()
    
# Save the combined plot
file_name <- paste0(basename(file_path), "_combined_plot.png")
output_path <- file.path(output_dir, file_name)
ggsave(output_path, plot = CombinePlots(plots = list(p1, p2)), width = 16, height = 12, dpi = 300)

# print some statistics about the filtered cells
print(paste("Number of cells in the filtered dataset:", nrow(predicted_cell_types.filtered))) #4799
print(paste("Number of cells in the original dataset:", ncol(multiomic_data))) # should just be 5K
# they got transposed for some reason. 

# not sure what nCounts_RNA is supposed to do tbh.
print(paste("distribution of total features in the filtered dataset:", summary(predicted_cell_types.filtered$nCount_RNA))) # this don't work 
print(paste("distribution of total features in the original dataset:", summary(multiomic_data$nCount_RNA)))


# > summary(predicted_cell_types.filtered)
#                       predicted.id  prediction.score.Astrocytes
#                       Excitatory neurons         :2713   Min.   :0.000000           
#                       Inhibitory neurons         : 658   1st Qu.:0.000000           
#                       Cerebrum_Unknown.3         : 500   Median :0.000000           
#                       Astrocytes                 : 375   Mean   :0.076394           
#                       Astrocytes/Oligodendrocytes: 253   3rd Qu.:0.009367           
#                       SKOR2_NPSR1 positive cells : 125   Max.   :0.988568           
#                       (Other)                    : 175        


# # now I want to do the co-embedding of the two datasets for a visual indication of the (dimension reduced) similarity between the two datasets.

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

