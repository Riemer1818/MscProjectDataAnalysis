# Load necessary library
library(biomaRt)

# Set up the connection to Ensembl
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

# Load the list of genes (adjust path as necessary)
genes <- readRDS("/home/msai/riemerpi001/data/all_variable_genes.rds")

# Query BioMart to retrieve Ensembl Protein IDs based on gene symbols
results <- getBM(attributes = c('hgnc_symbol', 'ensembl_peptide_id'),
                 filters = 'hgnc_symbol',
                 values = genes,
                 mart = ensembl)

# Remove rows with NA values in the ensembl_peptide_id column
results_cleaned <- na.omit(results)

# Save the cleaned results to a CSV file
write.csv(results_cleaned, "/home/msai/riemerpi001/data/ensembl_protein_ids_cleaned.csv", row.names = FALSE)

