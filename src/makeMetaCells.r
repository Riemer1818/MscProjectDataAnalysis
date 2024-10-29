library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
library(biovizBase)
library(dplyr)
library(SuperCell)
library(tools)

# Function to dynamically create the output directory if it doesn't exist
create_output_directory <- function(output_dir) {
    if (!dir.exists(output_dir)) {
        cat("Creating output directory:", output_dir, "\n")
        dir.create(output_dir, recursive = TRUE)
    } else {
        cat("Output directory exists:", output_dir, "\n")
    }
}

# Main function for processing RDS files and generating metacells
process_rds_files <- function(rds_folder, output_dir_base, cell_type_distribution_file, gamma) {
    # Ensure gamma is set and positive
    if (gamma <= 0) {
        stop("Gamma should be a positive number.")
    }
    
    # Create the base output directory if needed
    create_output_directory(output_dir_base)
    
    # List all RDS files in the specified folder
    rds_files <- list.files(rds_folder, pattern = "\\.RDS$", full.names = TRUE)
    
    combined_cell_type_distribution <- data.frame()
    
    # Iterate over each RDS file
    for (file_path in rds_files) {
        # Extract the basename of the file without extension
        input_file_base <- file_path_sans_ext(basename(file_path))
        
        # Load the multiomic data
        cat("Loading multiomic data from:", file_path, "\n")
        multiomic_data <- readRDS(file = file_path)
        sc <- UpdateSeuratObject(multiomic_data)
        rm(multiomic_data)
        
        # Normalize and find variable features
        sc <- NormalizeData(sc, verbose = FALSE)
        sc <- FindVariableFeatures(sc, selection.method = "disp", nfeatures = 1000, verbose = FALSE)
        hvg <- VariableFeatures(sc)
        
        # Scaling, PCA, UMAP
        sc <- ScaleData(sc, verbose = FALSE)
        sc <- RunPCA(sc, verbose = FALSE)
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
        pcal2 <- Embeddings(sc, reduction = "pca")
        
        MC <- SCimplify_from_embedding(
            X = pcal2,
            cell.annotation = sc$cell_type,
            gamma = gamma
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
        
        # Compute cell type distribution
        cell_type_distribution <- table(mc_seu$celltype)
        
        # Convert the table to a data frame for easy combination
        cell_type_distribution_df <- as.data.frame(cell_type_distribution)
        colnames(cell_type_distribution_df) <- c("CellType", "Count")
        
        # Add a column with the RDS file name for tracking
        cell_type_distribution_df$File <- basename(file_path)
        
        # Append the current distribution to the combined dataframe
        combined_cell_type_distribution <- rbind(combined_cell_type_distribution, cell_type_distribution_df)
        
        # Define output file path in the top-level output directory
        output_file <- paste0(input_file_base, "_MC.rds")
        full_output_path <- file.path(output_dir_base, output_file)
        
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
        
        # Garbage collection to free up memory
        gc()
    }
    
    # Write combined cell type distribution to file
    write.csv(combined_cell_type_distribution, file = cell_type_distribution_file, row.names = FALSE)
    cat("Saved combined cell type distribution to:", cell_type_distribution_file, "\n")
}

# Example usage of the process_rds_files function
rds_folder <- "~/data/filtered_seurats/originalRDS"
output_dir_base <- "~/data/filtered_seurats/MC3" 
cell_type_distribution_file <- "~/data/filtered_seurats/combinedCellDistribution.csv" 
gamma <- 40

process_rds_files(rds_folder, output_dir_base, cell_type_distribution_file, gamma)