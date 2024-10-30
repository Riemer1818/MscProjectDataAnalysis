# Load necessary libraries
library(Seurat)
library(dplyr)
library(Matrix)
library(rhdf5)


# File paths for input and output
input_motif_matrix_path <- "~/final_motif_matrix.h5"  # Replace with your actual file path
output_motif_matrix_path <- paste0("~/final_motif_matrix_thresh_", fold_change_threshold, ".h5")

# Define the threshold value
fold_change_threshold <- 0.2


# Load the motif matrix
motif_matrix <- h5read(input_motif_matrix_path, "motif_matrix")

# Calculate the number of values below the threshold in each column
num_below_threshold <- colSums(motif_matrix < fold_change_threshold)

# Find columns where more than 10% of values are below the threshold
num_values <- nrow(motif_matrix)
significant_columns <- which(num_below_threshold > (0.1 * num_values))

# Display the indices of the columns with significant changes
cat("Columns with significant changes (more than 10% values below threshold):", significant_columns, "\n")

# Extract a few columns that have significant changes for plotting
# Selecting the first 4 such columns, but you can adjust this as needed
significant_changing_columns <- motif_matrix[, significant_columns[1:4]]

# Apply the threshold to the matrix
motif_matrix[motif_matrix < fold_change_threshold] <- 0

# Extract the same columns after applying the threshold
significant_changing_columns_after <- motif_matrix[, significant_columns[1:4]]

# Create a 2x4 plot layout for comparison
par(mfrow = c(2, 4))

# Plot the distributions for the columns before thresholding
for (i in 1:4) {
  hist(significant_changing_columns[, i], main = paste("Before Threshold Col", significant_columns[i]), 
       xlab = "Values", ylab = "Frequency", col = "blue", breaks = 50)
}

# Plot the distributions for the columns after thresholding
for (i in 1:4) {
  hist(significant_changing_columns_after[, i], main = paste("After Threshold Col", significant_columns[i]), 
       xlab = "Values", ylab = "Frequency", col = "green", breaks = 50)
}

# Reset the plot layout
par(mfrow = c(1, 1))

# Save the modified motif matrix to a new HDF5 file
h5createFile(output_motif_matrix_path)
h5write(as.matrix(motif_matrix), output_motif_matrix_path, "motif_matrix")

# Open the file and dataset to write the attribute
file_id <- H5Fopen(output_motif_matrix_path)
dataset_id <- H5Dopen(file_id, "motif_matrix")

# Write the attribute to the dataset
h5writeAttribute(dataset_id, "threshold", fold_change_threshold)

# Close the dataset and file
H5Dclose(dataset_id)
H5Fclose(file_id)

cat("HDF5 files created successfully with threshold", fold_change_threshold, "\n")
