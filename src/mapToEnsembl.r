library(biomaRt)

# Function to connect to Ensembl using biomaRt
connect_to_ensembl <- function(dataset = "hsapiens_gene_ensembl") {
    cat("Connecting to Ensembl BioMart...\n")
    useMart("ensembl", dataset = dataset)
}

# Function to load gene list from a specified RDS file
load_gene_list <- function(gene_list_path) {
    cat("Loading gene list from:", gene_list_path, "\n")
    readRDS(gene_list_path)
}

# Function to query BioMart and retrieve Ensembl Protein IDs based on gene symbols
retrieve_ensembl_protein_ids <- function(genes, ensembl) {
    cat("Querying Ensembl for protein IDs...\n")
    getBM(attributes = c('hgnc_symbol', 'ensembl_peptide_id'),
          filters = 'hgnc_symbol',
          values = genes,
          mart = ensembl)
}

# Function to clean results by removing rows with NA values in 'ensembl_peptide_id'
clean_results <- function(results) {
    cat("Cleaning the results...\n")
    cleaned_results <- na.omit(results)
    return(cleaned_results)
}

# Function to log statistics to a file
log_statistics <- function(original_genes, cleaned_results, output_log_path) {
    total_genes <- length(original_genes)
    cleaned_genes <- nrow(cleaned_results)
    lost_genes <- total_genes - cleaned_genes
    duplicate_entries <- sum(duplicated(cleaned_results$hgnc_symbol))
    
    stats_message <- paste0(
        "Total genes queried: ", total_genes, "\n",
        "Total genes found (with Ensembl Protein ID): ", cleaned_genes, "\n",
        "Total genes lost (NA entries): ", lost_genes, "\n",
        "Number of duplicate entries: ", duplicate_entries, "\n"
    )
    
    cat(stats_message, file = output_log_path)
    cat("Statistics logged to:", output_log_path, "\n")
}

# Function to save cleaned results to a CSV file
save_results_to_csv <- function(results, output_file_path) {
    cat("Saving cleaned results to:", output_file_path, "\n")
    write.csv(results, output_file_path, row.names = FALSE)
}

# Main function to perform the entire process
process_gene_list <- function(gene_list_path, output_file_path, output_log_path, dataset = "hsapiens_gene_ensembl") {
    # Step 1: Connect to Ensembl
    ensembl <- connect_to_ensembl(dataset)
    
    # Step 2: Load the gene list
    genes <- load_gene_list(gene_list_path)
    
    # Step 3: Retrieve Ensembl Protein IDs
    results <- retrieve_ensembl_protein_ids(genes, ensembl)
    
    # Step 4: Clean the results
    results_cleaned <- clean_results(results)
    
    # Step 5: Save the cleaned results to a CSV file
    save_results_to_csv(results_cleaned, output_file_path)
    
    # Step 6: Log statistics to the output log file
    log_statistics(genes, results_cleaned, output_log_path)
    
    cat("Process completed successfully.\n")
}

# Example usage
gene_list_path <- "/home/msai/riemerpi001/data/all_variable_genes.rds"
output_file_path <- "/home/msai/riemerpi001/data/ensembl_protein_ids_cleaned.csv"
output_log_path <- "/home/msai/riemerpi001/data/ensembl_protein_ids_stats.log"

# Call the main function
process_gene_list(gene_list_path, output_file_path, output_log_path)
