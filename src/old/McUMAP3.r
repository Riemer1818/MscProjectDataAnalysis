# Load necessary libraries
library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
library(biovizBase)
library(dplyr)
library(SuperCell)

# Define file paths and titles for testing
file_paths <- c(
  "/home/msai/riemerpi001/data/filtered_seurats/originalRDS/GSM4508928_adrenal_filtered.seurat.RDS",
  "/home/msai/riemerpi001/data/filtered_seurats/originalRDS/GSM4508929_cerebellum_filtered.seurat.RDS"
)
plot_title <- "Combined UMAP Overlay: Adrenal & Cerebellum"
output_dir <- "/home/msai/riemerpi001/data/filtered_seurats/UMAP_overlay_plots"

# Function to load and preprocess Seurat objects
load_and_preprocess <- function(file_path) {
  cat("Loading multiomic data from:", file_path, "\n")
  multiomic_data <- readRDS(file = file_path)
  sc <- UpdateSeuratObject(multiomic_data)
  rm(multiomic_data)
  
  # Normalize, find variable features, scale, PCA, and UMAP
  sc <- NormalizeData(sc, verbose = FALSE)
  sc <- FindVariableFeatures(sc, selection.method = "disp", nfeatures = 1000, verbose = FALSE)
  sc <- ScaleData(sc, verbose = FALSE)
  sc <- RunPCA(sc, verbose = FALSE)
  
  return(sc)
}

# Load and merge both datasets
sc_adrenal <- load_and_preprocess(file_paths[1])
sc_cerebellum <- load_and_preprocess(file_paths[2])

# Add a dataset identifier to each object
sc_adrenal$dataset <- "Adrenal"
sc_cerebellum$dataset <- "Cerebellum"

# Merge the two Seurat objects
sc_combined <- merge(sc_adrenal, sc_cerebellum)

# Run UMAP on the combined dataset
sc_combined <- RunUMAP(sc_combined, dims = 1:10)

# Extract combined UMAP embeddings
umap_combined <- as.data.frame(Embeddings(sc_combined, "umap"))
colnames(umap_combined) <- c("UMAP_1", "UMAP_2")
umap_combined$group <- sc_combined$dataset

# Function to create metacell UMAPs for a given gamma for a specific dataset
create_metacell_umap <- function(sc, gamma_value, group_name) {
    # Get PCA embedding and perform metacell construction
    pcal2 <- Embeddings(sc, reduction = "pca")
    MC <- SCimplify_from_embedding(
        X = pcal2,
        cell.annotation = sc$cell_type,
        gamma = gamma_value
    )
  
    # Compute gene expression of metacells
    DefaultAssay(sc) <- "RNA"
    MC.ge <- supercell_GE(ge = GetAssayData(sc), mode = "sum", groups = MC$membership)
    colnames(MC.ge) <- paste0("MC_", seq_len(length(unique(MC$membership))))
  
    # Create Seurat object for metacells with only RNA counts
    mc_seu <- CreateSeuratObject(counts = MC.ge, assay = "RNA")
    celltypes <- MC$SC.cell.annotation
    names(celltypes) <- paste0("MC_", names(celltypes))
    mc_seu$cell_type <- celltypes[colnames(mc_seu)]
  
    # Add metacell membership information to the UMAP data
    umap_combined$metacell_membership <- MC$membership
    umap_mc_centroids <- umap_combined %>%
        filter(group == group_name) %>%
        group_by(metacell_membership) %>%
        summarise(
            UMAP_1 = mean(UMAP_1),
            UMAP_2 = mean(UMAP_2)
        )
    umap_mc_centroids <- as.data.frame(umap_mc_centroids)
    rownames(umap_mc_centroids) <- colnames(mc_seu)
  
    # Assign UMAP coordinates to the metacells in mc_seu
    mc_seu[["umap"]] <- CreateDimReducObject(
        embeddings = as.matrix(umap_mc_centroids[, c("UMAP_1", "UMAP_2")]),
        key = "UMAP_",
        assay = DefaultAssay(mc_seu)
    )
  
    # Extract UMAP embeddings for metacell data
    umap_mc_df <- as.data.frame(Embeddings(mc_seu, "umap"))
    colnames(umap_mc_df) <- c("UMAP_1", "UMAP_2")
    umap_mc_df$group <- paste0("Gamma_", gamma_value, " - ", group_name)
    umap_mc_df$celltype <- mc_seu$cell_type
    
    return(umap_mc_df)
}

# Generate metacell UMAPs for different gamma values for both datasets
umap_gamma_10_adrenal <- create_metacell_umap(sc_adrenal, gamma_value = 10, group_name = "Adrenal")
umap_gamma_10_cerebellum <- create_metacell_umap(sc_cerebellum, gamma_value = 10, group_name = "Cerebellum")
umap_gamma_20_adrenal <- create_metacell_umap(sc_adrenal, gamma_value = 20, group_name = "Adrenal")
umap_gamma_20_cerebellum <- create_metacell_umap(sc_cerebellum, gamma_value = 20, group_name = "Cerebellum")

# Combine all UMAP datasets
umap_combined_metacells <- rbind(umap_combined, umap_gamma_10_adrenal, umap_gamma_10_cerebellum, umap_gamma_20_adrenal, umap_gamma_20_cerebellum)

# Create separate plots for each group
p_combined <- ggplot(umap_combined, aes(x = UMAP_1, y = UMAP_2, color = group)) +
    geom_point(alpha = 0.7, size = 1.5) +
    labs(title = "Original Cells Combined") +
    theme_minimal()

p_gamma_10 <- ggplot(rbind(umap_gamma_10_adrenal, umap_gamma_10_cerebellum), aes(x = UMAP_1, y = UMAP_2, color = group)) +
    geom_point(alpha = 0.7, size = 1.5) +
    labs(title = "Metacells (Gamma = 10)") +
    theme_minimal()

p_gamma_20 <- ggplot(rbind(umap_gamma_20_adrenal, umap_gamma_20_cerebellum), aes(x = UMAP_1, y = UMAP_2, color = group)) +
    geom_point(alpha = 0.7, size = 1.5) +
    labs(title = "Metacells (Gamma = 20)") +
    theme_minimal()

# Combine all plots side by side
combined_plot <- p_combined + p_gamma_10 + p_gamma_20 + plot_layout(ncol = 3)

fileLoc <- file.path(output_dir, paste0("combined_adrenal_cerebellum_UMAP_overlay_gamma.png"))
# Save the combined plot
ggsave(file = fileLoc, plot = combined_plot, width = 24, height = 8)

# Print the combined plot for display
print(combined_plot)
