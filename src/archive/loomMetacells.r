library(Matrix)
library(ggplot2)
library(patchwork)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v75)
library(biovizBase)
library(dplyr)
library(SuperCell)
library(loomR) # for loom file reading

# Load the loom file
file <- "~/RIEMERPI001/data/GSE156793_S3_gene_count.loom"
loom <- connect(file)

# Load UMAP and cell annotations
pca1 <- loom[["col_attrs/Main_cluster_umap_1"]][]
pca2 <- loom[["col_attrs/Main_cluster_umap_2"]][]
pca_embedding <- cbind(pca1, pca2)
cell_annotation <- loom[["col_attrs/Main_cluster_name"]][]  # Cell type annotations
group_by <- loom[["col_attrs/Main_cluster_name"]][]  # Replace with other attributes as needed

# Function to construct metacells for a specific group
construct_metacells <- function(pca_embedding, cell_annotation, group_name) {
  group_idx <- which(group_by == group_name)
  group_pca <- pca_embedding[group_idx, ]
  group_annotation <- cell_annotation[group_idx]
  
  # Construct metacells for this group
  MC_group <- SCimplify_from_embedding(
    X = group_pca, 
    cell.annotation = group_annotation,
    gamma = 10  # Adjust gamma based on granularity needs
  )
  
  return(list(MC_group = MC_group, group_idx = group_idx))
}

# Get unique groups (cell types or clusters)
unique_groups <- unique(group_by)


# think will be too big and cannot get to work?
# metacell_result.ge <- supercell_GE(ge = loom$matrix, mode = "sum", groups = metacell_result$MC_group$membership)


# Select only the first group
group <- unique_groups[2]

# Construct metacells for the first group
metacell_result <- construct_metacells(pca_embedding, cell_annotation, group)
# metacell_result <- construct_metacells(pca_embedding, cell_annotation, unique_groups)


# too big? will do it in chunks instead
metacell_result.ge <- supercell_GE(ge = loom$matrix, mode = "sum", groups = metacell_result$membership)


# Extract the constructed metacells and group membership
MC_group <- metacell_result$MC_group
group_idx <- metacell_result$group_idx

# Print the metacell object for inspection
print(MC_group)

# Now compute the gene expression matrix (MC.ge) for this group in memory-efficient chunks
message("Computing gene expression matrix for the metacell group in chunks...")

# Get the number of genes and metacells
num_genes <- loom$shape[1]  # Number of genes (rows)
num_metacells <- length(unique(MC_group$membership))

# Initialize an empty matrix to store metacell expression
metacell_expression <- matrix(0, nrow = num_genes, ncol = num_metacells)

# Define chunk size (number of genes to process at a time)
chunk_size <- 1000  # Adjust based on memory limits

# Process the gene expression matrix in chunks
for (start_idx in seq(1, num_genes, by = chunk_size)) {
  end_idx <- min(start_idx + chunk_size - 1, num_genes)  # End of the chunk
  
  # Extract the chunk of gene expression data
  message(paste("Processing genes", start_idx, "to", end_idx))
  gene_chunk_expression <- loom[["matrix"]][start_idx:end_idx, group_idx]
  
  # For each metacell, sum the expression of the corresponding cells within this gene chunk
  for (mcell in unique(MC_group$membership)) {
    metacell_cells <- which(MC_group$membership == mcell)
    metacell_expression[start_idx:end_idx, mcell] <- rowSums(gene_chunk_expression[, metacell_cells, drop = FALSE])
  }
  
}

colnames(metacell_expression) <- paste0("MC_", seq_len(num_metacells))


# does coerce to dgCmatrix, but it's not a big matrix

# Create Seurat object from the metacell gene expression matrix
mc_seu <- CreateSeuratObject(counts = metacell_expression, assay = "RNA")

# # Normalize the data and perform PCA and UMAP
# mc_seu <- NormalizeData(mc_seu)
# mc_seu <- FindVariableFeatures(mc_seu)
# mc_seu <- ScaleData(mc_seu, features = VariableFeatures(mc_seu))
# mc_seu <- RunPCA(mc_seu, npcs = 50)
# mc_seu <- RunUMAP(mc_seu, dims = 1:30)

# what I think I should do: concatinate the metacells and the seurat object and then save it as an rds file


# Save Seurat object
saveRDS(mc_seu, file ="~/RIEMERPI001/data/metacells_seurat.rds")

# Close loom file connection
loom$close()

# Quit the session
quit(save = "no")
