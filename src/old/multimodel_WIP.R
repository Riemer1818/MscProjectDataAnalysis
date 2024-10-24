library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
# library(hdf5r)
library(biovizBase)
library(dplyr)

# if (!requireNamespace("remotes")) install.packages("remotes")
# remotes::install_github("GfellerLab/SuperCell")
library(SuperCell)


file_path <- "./data/GSM4508930_cerebrum_filtered.seurat.RDS"

output_dir <- "./output/GSM4508930_cerebrum_filtered.seurat"

multiomic_data <- readRDS(file = file_path)
multiomic_data <- UpdateSeuratObject(multiomic_data)

# # Assuming 'multiomic_data' is your Seurat object
# set.seed(123)  # Set seed for reproducibility

# # Randomly sample 500 cells
# cell_subset <- sample(colnames(multiomic_data), size = 500)

# # Subset the Seurat object to only include these cells
# multiomic_data <- subset(multiomic_data, cells = cell_subset)

sc <- multiomic_data
rm(multiomic_data)

# https://github.com/GfellerLab/SIB_workshop/blob/main/workbooks/Workbook_1__cancer_cell_lines.md#data-simplification-coarse-graining-construction-of-metacells
sc <- NormalizeData(sc, verbose=FALSE)

sc <- FindVariableFeatures(
  sc, 
  selection.method = "disp", # "vst" is default
  nfeatures = 1000,
  verbose=FALSE
)

hvg <- VariableFeatures(sc, verbose=FALSE)

# Plot variable features 
plot1 <- VariableFeaturePlot(sc)
LabelPoints(plot = plot1, points = hvg[1:20], repel = TRUE)


sc <- ScaleData(sc, verbose=FALSE)
sc <- RunPCA(sc, verbose=FALSE)
DimPlot(sc, reduction = "pca", group.by = "cell_type") # this is not very good tbh... 

sc <- RunUMAP(sc,  dims = 1:10)
DimPlot(sc, reduction = "umap", group.by = "cell_type") # this looks OK.

sc <- FindNeighbors(sc, dims = 1:10)
sc <- FindClusters(sc, resolution = 0.05)
DimPlot(sc, reduction = "umap", group.by = "ident")

Idents(sc) <- "cell_type" 
levels(sc) <- sort(levels(sc))
# Compute upregulated genes in each cell line (versus other cells)
sc.all.markers <-  FindAllMarkers(sc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, test.use = "t")
# saveRDS(sc.all.markers, file = file.path(data.folder, "output", "sc_all_markers.Rds"))

# Top markers (select top markers of each cell line)
sc.top.markers <- sc.all.markers %>%
  group_by(cluster) %>%
  slice_max(n = 2, order_by = avg_log2FC)

sc.top.markers

# plot top markers
VlnPlot(sc, features = sc.top.markers$gene[c(seq(1, 9, 2), seq(2, 10, 2))], ncol = 5, pt.size = 0.0) # this looks bad for 500 genes, too...


## this is the gamma factor that we want to have "quite" small in order to have very similar intermetecell gene expression
gamma <- 10 # Graining level

# Compute metacells using SuperCell package
# the scaling is the issue here. 
# MC <- SCimplify(
#   X = GetAssayData(sc, assay = "RNA", layer = "data"), # single-cell log-normalized gene expression data
#   genes.use = hvg, 
#   gamma = 5,
#   n.pc = 20
# )

# get the embedding
pcal2 <-Embeddings(sc2, reduction = "pca_l2_harmony")

MC <- SCimplify_from_embedding(
  x = pcal2, # embedding
  cell_annotation = sc2$cell_type
  gamma = 20
)

DefaultAssay(sc) <- "RNA"

# Compute gene expression of metacells by simply averaging gene expression within each metacell
MC.ge <- supercell_GE(
  ge = GetAssayData(sc),
  groups = MC$membership
)

# Set the default assay to "peaks"
DefaultAssay(sc) <- "peaks"

MC.peaks <- supercell_GE(
  ge = GetAssayData(sc),
  groups = MC$membership
)

sc$MC_membership = MC$membership

DimPlot(sc, group.by = "MC_membership") + theme(legend.position = "none")

colnames(MC.peaks) = colnames(MC.ge) = paste0("MC_", seq_len(ncol(MC.ge)))

mc_seu = CreateSeuratObject(counts = MC.ge, assay = "RNA", 
                            meta.data = data.frame("celltype" = MC$cell_type,
                                                   "purity" = MC$purity))

mcp = CreateAssayObject(MC.peaks)

mc_seu[["peaks"]] = mcp


mc_seu <- NormalizeData(mc_seu)
mc_seu <- FindVariableFeatures(mc_seu)
mc_seu <- ScaleData(mc_seu, features = VariableFeatures(mc_seu))

mc_seu <- RunPCA(mc_seu, npcs = 20)  # n.pc from your metadata
mc_seu <- RunUMAP(mc_seu, dims = 1:20)

DimPlot(mc_seu, group.by = "celltype")

DefaultAssay(mc_seu) <- "peaks"
print("Set default assay to peaks.")
print("Normalizing peaks data...")
mc_seu <- RunTFIDF(mc_seu) # I think this runs the LSI
print("Done normalizing peaks data.")
print("Finding top features...")
mc_seu <- FindTopFeatures(mc_seu, min.cutoff = 0.01)
print("Done finding top features.")
print("Normalizing peaks data...")
mc_seu <- NormalizeData(mc_seu)
print("Done normalizing peaks data.")





# # Alternatively, counts can be averaged (summed up) followed by a lognormalization step (this approach is used in the MetaCell and SEACell algorithms)
# if(0){
#   MC.counts <- supercell_GE(
#     ge = GetAssayData(sc, slot = "counts"),
#     mode = "sum", # summing counts instead of the default averaging
#     groups = MC$membership
#   )

#   MC.ge <- Seurat::LogNormalize(MC.counts, verbose = FALSE)
# }

# Plot the histogram for the number of cells per metacell
print("Creating histogram of number of cells per metacell...")
hist(table(MC$membership), main = "Number of Cells per Metacell", xlab = "Number of Cells")
dev.copy(png, filename = paste0(output_dir, "/hist_cells_per_metacell.png"))
dev.off()

# Transfer metadata (annotate metacell to a certain cell line)

# Annotate metacells to cells line
MC$cell_type <- supercell_assign(
  cluster = sc@meta.data$cell_type,          # single-cell assignment to cell lines 
  supercell_membership = MC$membership,  # single-cell assignment to metacells
  method = "absolute" # available methods are c("jaccard", "relative", "absolute"), function's help() for explanation
)

# Compute purity of metacells as :
#  * a proportion of the most abundant cell type withing metacells (`method = `"max_proportion)
#  * an entropy of cell type within metacells (`method = "entropy"`)
method_purity <- c("max_proportion", "entropy")[1]
MC$purity <- supercell_purity(
  clusters = sc@meta.data$cell_type,
  supercell_membership = MC$membership, 
  method = "max_proportion"
)

# Metacell purity distribution
summary(MC$purity)

# Plot the histogram
print("Creating histogram of purity of metacells...")
hist(MC$purity, main = paste0("Purity of metacells \nin terms of cell line composition (", method_purity, ")"))
dev.copy(png, filename = paste0(output_dir, "/purity_metacells_histogram.png"))
dev.off()

# this looks like where they are allocationing to dense... which takes forever... 

# looks like there is not many different cell types in the metacells...
# Plot the supercell graph
supercell_plot(
  MC$graph.supercells, 
  group = MC$cell_type, 
  seed = 1, 
  alpha = -pi/2,
  main = "Metacells colored by cell line assignment"
)

dev.copy(png, filename = paste0(output_dir, "/supercells_colored_by_cell_line_assignment.png"))
dev.off()



# here specifically...
# # not sure how to interpret this...
# supercell_plot(
#   MC$graph.singlecell, 
#   group = sc@meta.data$cell_type, 
#   do.frames = FALSE,
#   lay.method = "components",
#   seed = 1, 
#   alpha = -pi/2,
#   main  = "Single cells colored by cell line assignment"
# )

## downstream analysis
# Create a Seurat object from the metacell gene expression matrix

# THIS FUNCTION IS BROKEN?

# MC.seurat <- supercell::supercell_2_Seurat(
#   SC.GE = MC.ge, 
#   SC = MC, 
#   fields = c("cell_type", "purity"),
#   var.genes = MC$genes.use,
#   N.comp = 10
# )

summary(MC)

# # they both don't have colnames 
# colnames(MC.ge)
# colnames(MC.peaks)

print(paste("Number of metacells:", length(unique(MC$membership)))) # 250s (at sample of 5k)

colnames <- 1: length(unique(MC$membership))

colnames(MC.ge) <- colnames
colnames(MC.peaks) <- colnames

# > typeof(MC.ge)
# [1] "S4"
# > typeof(MC.peaks)
# [1] "S4"

# class(MC.ge)
# class(MC.peaks)

# this SHOULD be fast, because I am merely changing the internal representation of the data. It should no longer coerce to dense in the Seurat functions.
# this is NOT fast

print("Converting gene expression matrix to dgCMatrix...")
countsGe <- as.(MC.ge, "dgCMatrix")
print("Converted gene expression matrix to dgCMatrix.")

print("Converting peaks matrix to dgCMatrix...")
countsPeaks <- as.(MC.peaks, "dgCMatrix")
print("Converted peaks matrix to dgCMatrix.")

seurat_object <- CreateSeuratObject(counts = countsGe)
print("Created Seurat object.")

# will take a bit longer to run... 
# Add chromatin assay (assuming MC.peaks is already formatted as a matrix or dgCMatrix)
# THIS ONE is making dense for some reason? 

temp <- CreateAssayObject(counts = countsPeaks)
seurat_object[["peaks"]] <- temp
print("Added chromatin assay.")
)

# Check if the matrix within the Seurat object is still sparse
is_sparse <- inherits(seurat_object[["peaks"]]@counts, "sparseMatrix")
print(paste("Is the 'peaks' assay stored as sparse matrix? ", is_sparse))


# Warning message:
# In asMethod(object) :
#   sparse->dense coercion: allocating vector of size 66.8 GiB
#   sparse->dense coercion: allocating vector of size 66.8 GiB
MCD <- CreateAssayObject(counts = MC.peaks)
seurat_object[["peaks"]] <- MCD
print("Added chromatin assay.")

# Adding metadata, assuming the metadata components like cell_type and supercell_size are vectors
seurat_object$supercell_size <- MC$supercell_size
seurat_object$membership <- MC$membership
seurat_object$cell_type <- MC$cell_type
seurat_object$purity <- MC$purity
print("Added metadata.")

# Set the default assay to RNA and normalize and scale
DefaultAssay(seurat_object) <- "RNA"
print("Set default assay to RNA.")
print("Normalizing RNA data...")
seurat_object <- NormalizeData(seurat_object)
print("Done normalizing RNA data.")
print("Finding variable features...")
seurat_object <- FindVariableFeatures(seurat_object)
print("Done finding variable features.")

# seurat_object <- ScaleData(seurat_object)
# If scaling is memory-intensive, try scaling only the variable features:
seurat_object <- ScaleData(seurat_object, features = VariableFeatures(seurat_object))
print("Done scaling RNA data.")

seurat_object <- RunPCA(seurat_object, npcs = MC$n.pc)  # n.pc from your metadata
print("Performed PCA on RNA data.")

DefaultAssay(seurat_object) <- "peaks"
print("Set default assay to peaks.")
print("Normalizing peaks data...")
seurat_object <- RunTFIDF(seurat_object) # I think this runs the LSI
print("Done normalizing peaks data.")
print("Finding top features...")
seurat_object <- FindTopFeatures(seurat_object, min.cutoff = 0.01)
print("Done finding top features.")
print("Normalizing peaks data...")
seurat_object <- NormalizeData(seurat_object)
print("Done normalizing peaks data.")

# seurat_object <- ScaleData(seurat_object)
# if scaling is memory-intensive, try scaling only the variable features:
seurat_object <- ScaleData(seurat_object, features = VariableFeatures(seurat_object))
print("Done scaling peaks data.")

print("Done: making seurat object")


## Integration of RNA and chromatin data
# Set RNA as the reference assay
reference <- seurat_object
DefaultAssay(reference) <- "RNA"

# Set chromatin as the query assay
query <- seurat_object
DefaultAssay(query) <- "peaks"

print("Finding transfer anchors...")
# Find transfer anchors between RNA and chromatin assays
transfer_anchors <- FindTransferAnchors(
  reference = reference, 
  query = query, 
  reduction = "cca", 
  dims = 1:20,   # Number of dimensions (1:20 for testing)
)

print("Transfer anchors found.")


# ## [1] "Done: NormalizeData"
# ## [1] "Doing: data to normalized data"
# ## [1] "Doing: weighted scaling"
# ## [1] "Done: weighted scaling"

# Idents(MC.seurat) <- "cell_line"

# MC.seurat <- RunUMAP(MC.seurat, dims = 1:10)


# print("Assays in multiomic_data:")
# Assays(multiomic_data) # RNA and peaks 

# multiomic_data <- SCTransform(multiomic_data, assay = "RNA", verbose = FALSE) #useNames = NA is deprecated
# multiomic_data <- RunTFIDF(multiomic_data)
# multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = "q0")
# multiomic_data <- RunSVD(multiomic_data)

# expression_matrix <- multiomic_data@assays$RNA@data  # Extracting normalized expression data from Seurat object
# peaks_matrix <- multiomic_data@assays$peaks@data

# # Check if the column names of the two matrices are equal
# different_columns <- colnames(expression_matrix)[colnames(expression_matrix) != colnames(peaks_matrix)]
# if (length(different_columns) > 0) {
#   print("The following columns have different names:")
#   print(different_columns) # does this not indicate that every cell is accounted for? 
# } else {
#   print("The column names of the two matrices are equal.")
# }

# # Generate metacells
# print("Generating metacells...")
# metacells <- SCimplify(
#   expression_matrix,  
#   gamma = 20,         
#   n.var.genes = 1000, 
#   k.knn = 10,         
#   n.pc = 20           
# )

# # Print the number of metacells
# print(paste("Number of metacells:", length(unique(metacells$membership)))) # 250s (at sample of 5k)

# # Create a new metacell object for peaks with the same membership
# peak_metacells<- metacells$membership[colnames(peaks_matrix)]
# expression_metacells <- metacells$membership[colnames(expression_matrix)]

# # Plot the size of metacells | the number of cells in each metacell is still a bit hgh and should be lowered by increasing the k value
# print("Plotting metacell size distribution...")
# hist(table(metacells$supercell_size), main = "Metacell Size Distribution", xlab = "Number of Cells per Metacell")

# rna_sparse_matrix <- Matrix(multiomic_data@assays$RNA@data, sparse = TRUE)
# peaks_sparse_matrix <- Matrix(multiomic_data@assays$peaks@data, sparse = TRUE)

# # Generate the RNA metacell matrix using the sparse RNA matrix
# print("Generating RNA metacell matrix...")
# rna_metacell_matrix <- supercell_GE(
#   rna_sparse_matrix,         # Use the sparse RNA expression matrix
#   groups = metacells$membership,   # Metacell membership vector
#   mode = "sum"                 # Aggregating the expression by averaging
# )

# length(metacells)
# length(metacells$membership)

# metacells$cell_type <- supercell_assign(
#   cluster = multiomic_data@meta.data$cell_type,          # single-cell assignment to cell lines 
#   supercell_membership = metacells$membership,  # single-cell assignment to metacells
#   method = "absolute" # available methods are c("jaccard", "relative", "absolute"), function's help() for explanation
# )

# #  ## Annotate metacells to cell lines
# #   metacell2$SC_cell_line <- supercell_assign(
# #     sc.meta$cell_line, 
# #     supercell_membership = metacell2$membership
# #   )

# # rm(multiomic_data)
# # gc()

# # I don't get colnames because they are being stored (for some reason?) as 
# # > metacells$membership[1]
# # CCGACGCGCCTTCTCTACGATATCAAGGTCACCTAGGAGA 
# #                                    128 
# # and I don't know how to split them up... 

# numeric_values = as.numeric(metacells$membership) # 5K long

# colnames(rna_metacell_matrix) <- 1:length(unique(metacells$membership))

# # Generate the Peaks metacell matrix using the sparse Peaks matrix
# print("Generating Peaks metacell matrix...")
# peaks_metacell_matrix <- supercell_GE(
#   ge = peaks_sparse_matrix,       # Use the sparse Peaks expression matrix
#   groups = metacells$membership,   # Metacell membership vector
#   mode = "sum"                 # Aggregating the expression by summing
# )

# # I'm pretty sure there is a better way to do this 
# colnames(peaks_metacell_matrix) <-1:length(unique(metacells$membership))


# metacells.seurat <- supercell_2_Seurat(
#   SC.GE = rna_metacell_matrix,     # Gene expression matrix (ensure it's in the correct format)
#   SC = metacells,                    # Supercell object (metadata or annotation)
#   fields = c("cell_type"),           # Metadata fields (only once)
#   var.genes = metacells$genes.use,   # Variable genes (ensure this is correctly structured)
#   N.comp = 10                        # Number of principal components to compute
# )


# # colnames(rna_metacell_matrix)
# # colnames(peaks_metacell_matrix)

# # so actually. What we should do: 
# # 1. compute metacells based on the RNA data
# # 2. make metacells using this membership vector (for both peaks and RNA)
# # 3. confirm that the two metacells are indeed made from the same membership vector

# # Create a new Seurat object with the RNA metacell matrix
# print("Creating Seurat object with RNA metacell matrix...")
# metacell_data[["RNA"]] <- CreateAssayObject(counts = rna_metacell_matrix, assay = "RNA")

# # library(future)
# # cores = parallel::detectCores()/4
# # plan(multicore, workers = cores)  # Adjust workers to match your CPU cores

# # Add the Peaks assay to the Seurat object using the sparse matrix
# print("Adding Peaks assay to the Seurat object...")
# metacell_data[["peaks"]] <- CreateAssayObject(counts = peaks_metacell_matrix)

# print("Assays in metacell_data:")
# print(Assays(metacell_data)) # RNA and peaks 

# # # Perform SCTransform on the RNA assay
# # print("Performing SCTransform on RNA assay...") # this was adding SCT assay... which is not what I have in the multiomics_dat in scRNA_ATAC_merge.R
# # metacell_data <- SCTransform(metacell_data, assay = "RNA", verbose = FALSE)

# # # Run TF-IDF transformation for the peaks assay
# # print("Running TF-IDF transformation for peaks assay...")
# # metacell_data <- RunTFIDF(metacell_data)

# # # Find top features
# # print("Finding top features...")
# # metacell_data <- FindTopFeatures(metacell_data, min.cutoff = 0.01)

# # # Run SVD for the peaks assay
# # print("Running SVD for peaks assay...")
# # metacell_data <- RunSVD(metacell_data)

# ## I thnk I might not hve to do all of this and can just immediately do the anchor transfer because it performs CCA itself...

# # ## why do I have to rnormalize and scale here? I get erorrs otherwise, but do not have to do this explicitly when not using metacells.

# # # Normalize the data
# # print("Normalizing the data...")
# # metacell_data <- NormalizeData(metacell_data, assay = "RNA")
# # metacell_data <- NormalizeData(metacell_data, assay = "peaks")

# # # Scale the data
# # print("Running scaling cells...")
# # metacell_data <- ScaleData(metacell_data, assay = "RNA")
# # metacell_data <- ScaleData(metacell_data, assay = "peaks")
# # ## I thnk I don't have to because I was using metacell_seurat object and not the metacell_data object

# # print('Running FindVariableFeatures...')
# metacell_data <- FindVariableFeatures(metacell_data, selection.method = "vst", nfeatures = 2000)

# metacell_data <- NormalizeData(metacell_data, assay = "RNA")
# metacell_data <- NormalizeData(metacell_data, assay = "peaks")

# # ## not sure why I have to normalize, scale and find variable features again...

# # ##### PCA and LSI 
# # print('Running PCA...')
# # metacell_data <- RunPCA(metacell_data, assay = "RNA")

# # print("Performing LSI for peaks...")
# # metacell_data <- RunLSI(metacell_data, assay = "peaks")

# # seems to find transfors anchors for the same object... so that's "positive" 17011 anchors.
# print("Finding transfer anchors...")
# print(paste("Number of metacells:", dim(metacell_data)[2]))

# transfer_anchors <- FindTransferAnchors(
#   reference = metacell_data, 
#   query = metacell_data,
#   reduction = "cca",
#   dims =  1:20, # for testing
#   k.score = 20 # for testing
# )

# # need to double check that this is actually using the right data.... seems weird that it calls multiomic_data and not the metacell_data
# # celltype.predictions <- TransferData(anchorset = transfer_anchors, refdata = multiomic_data@meta.data$cell_type,  weight.reduction = multiomic_data[["pca"]], dims = 1:30)
# # print("Transfer anchors found.")

# # ok so here I want to transfer the cell_type to the metacells. However how can I ever guarentee that the metacells have the right cell type?
# # what can I do: 
#   # I could use the metacell membership to get the cell type of the cells that make up the metacells
#   # but i'm not sure what I am even trying to do in this function... 
# celltype.predictions <- TransferData(anchorset = transfer_anchors, refdata = metacell_data$cell_type,  weight.reduction = metacell_data[["pca"]], dims = 1:20)
# print("Transfer anchors found.")

# # Define celltype.predictions
# print("Converting predicted cell types to factors...")
# celltype.predictions$predicted.id <- as.factor(celltype.predictions$predicted.id)
# print("Predicted cell types converted to factors.")

# # Extract the predicted cell types as a vector
# predicted_cell_types <- celltype.predictions$predicted.id

# # Print the distribution of predicted cell types as a histogram using ggplot2
# print("Creating histogram of predicted cell types...")
# p <- ggplot(data = celltype.predictions, aes(x = predicted.id)) +
#   geom_bar() +
#   xlab("Predicted Cell Types") +
#   ylab("Frequency") +
#   ggtitle("Distribution of Predicted Cell Types") +
#   theme_minimal()

# # Save the histogram plot
# file_name <- paste0(basename(file_path), "_metacell_celltype_histogram.png")
# output_path <- file.path(output_dir, file_name)
# ggsave(output_path, plot = p, width = 16, height = 12, dpi = 300)
# print("Histogram of predicted cell types saved.")

# # Plot the prediction score as a histogram
# print("Creating histogram of prediction scores...")
# prediction_score <- celltype.predictions$prediction.score.max
# histogram <- ggplot(data = celltype.predictions, aes(x = prediction_score)) +
#     geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black") +
#     xlab("Prediction Score") +
#     ylab("Frequency") +
#     ggtitle("Distribution of Prediction Scores") +
#     theme_minimal()

# # Save the histogram plot
# file_name <- paste0(basename(file_path), "_metacell_prediction_score_histogram.png")
# output_path <- file.path(output_dir, file_name)
# ggsave(output_path, plot = histogram, width = 16, height = 12, dpi = 300)
# print("Histogram of prediction scores saved.")


# # filter on prediction score
# print("Filtering metacell_data based on prediction score...")
# metacell_data.filtered <- subset(metacell_data, subset = prediction.score.max > 0.5)
# metacell_data.filtered$predicted.id <- factor(metacell_data.filtered$predicted.id, levels = levels(metacell_data))  # to make the colors match
# print("metacell_data filtered based on prediction score.")

# p1 <- DimPlot(metacell_data.filtered, group.by = "predicted.id", label = TRUE, repel = TRUE) + ggtitle("scATAC-seq cells (Metacells)") + 
#     NoLegend() + scale_colour_hue(drop = FALSE)

# p2 <- DimPlot(metacell_data, group.by = "cell_type", label = TRUE, repel = TRUE) + ggtitle("scRNA-seq cells") + 
#     NoLegend()

# # Save the combined plot
# print("Creating combined plot...")
# file_name <- paste0(basename(file_path), "_metacell_combined_plot.png")
# output_path <- file.path(output_dir, file_name)
# ggsave(output_path, plot = CombinePlots(plots = list(p1, p2)), width = 16, height = 12, dpi = 300)
# print("Combined plot saved.")

# # print some statistics about the filtered cells
# print(paste("Number of cells in the filtered dataset:", nrow(metacell_data.filtered)))
# print(paste("Number of cells in the original dataset:", nrow(metacell_data)))
# print(paste("Distribution of total features in the filtered dataset:", summary(metacell_data.filtered$nCount_RNA)))
# print(paste("Distribution of total features in the original dataset:", summary(metacell_data$nCount_RNA)))