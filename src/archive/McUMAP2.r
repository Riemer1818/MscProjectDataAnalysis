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

# Define a single file path and title for testing

file_path <- "/home/msai/riemerpi001/data/filtered_seurats/originalRDS/GSM4508930_cerebrum_filtered.seurat.RDS"
plot_title <- "Cerebrum UMAP Overlay"
output_dir <- "/home/msai/riemerpi001/data/filtered_seurats/UMAP_overlay_plots"

# Extract the base name for file naming
input_file_base <- tools::file_path_sans_ext(basename(file_path))

# Load the multiomic data (Seurat object)
cat("Loading multiomic data from:", file_path, "\n")
multiomic_data <- readRDS(file = file_path)
sc <- UpdateSeuratObject(multiomic_data)
rm(multiomic_data)

# Normalize and find variable features for original cells
sc <- NormalizeData(sc, verbose = FALSE)
sc <- FindVariableFeatures(sc, selection.method = "disp", nfeatures = 1000, verbose = FALSE)

# Scaling, PCA, UMAP for original cells
sc <- ScaleData(sc, verbose = FALSE)
sc <- RunPCA(sc, verbose = FALSE)
sc <- RunUMAP(sc, dims = 1:10)

# Extract original UMAP embeddings
umap_orig <- as.data.frame(Embeddings(sc, "umap"))
colnames(umap_orig) <- c("UMAP_1", "UMAP_2")
umap_orig$group <- "Original"

# Function to create metacell UMAPs for a given gamma
create_metacell_umap <- function(sc, gamma_value) {
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
    umap_orig$metacell_membership <- MC$membership
    umap_mc_centroids <- umap_orig %>%
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
    umap_mc_df$group <- paste0("Gamma_", gamma_value)
    umap_mc_df$celltype <- mc_seu$cell_type
    
    return(umap_mc_df)
}

# Generate metacell UMAPs for different gamma values
umap_gamma_10 <- create_metacell_umap(sc, gamma_value = 10)
umap_gamma_20 <- create_metacell_umap(sc, gamma_value = 20)
umap_gamma_40 <- create_metacell_umap(sc, gamma_value = 40)

# Combine all UMAP datasets
umap_combined <- rbind(umap_orig, umap_gamma_10, umap_gamma_20, umap_gamma_40)

# Create separate plots for each group
p_orig <- ggplot(umap_orig, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(alpha = 0.7, size = 1.5, color = "blue") +
    labs(title = "Original Cells") +
    theme_minimal()

p_gamma_10 <- ggplot(umap_gamma_10, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(alpha = 0.7, size = 1.5, color = "red") +
    labs(title = "Metacells (Gamma = 10)") +
    theme_minimal()

p_gamma_20 <- ggplot(umap_gamma_20, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(alpha = 0.7, size = 1.5, color = "green") +
    labs(title = "Metacells (Gamma = 20)") +
    theme_minimal()

p_gamma_40 <- ggplot(umap_gamma_40, aes(x = UMAP_1, y = UMAP_2)) +
    geom_point(alpha = 0.7, size = 1.5, color = "purple") +
    labs(title = "Metacells (Gamma = 40)") +
    theme_minimal()

# Combine all plots side by side
combined_plot <- p_orig + p_gamma_10 + p_gamma_20 + p_gamma_40 + plot_layout(ncol = 4)

fileLoc = file.path(output_dir, paste0(input_file_base, "_UMAP_overlay_gamma.png"))
# Save the combined plot
ggsave(file = fileLoc, plot = combined_plot, width = 20, height = 8)

# Print the combined plot for display
print(combined_plot)

