#!/bin/bash
#SBATCH --job-name=filesplitter
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=2
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err
#SBATCH --time=6:00:00

input_file="./data/cicero/cicero_coaccess_scores_by_cell_type.csv"
output_dir="./data/cicero/output"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Create a unique list of peaks
less -S "$input_file" | cut -f 1 -d, | sort | uniq > "${output_dir}/unique_peaks.csv"

# Read the header to get column names
header=$(head -n 1 "$input_file")

# Extract column names starting from the third column
columns=$(echo "$header" | awk -F, '{for (i=3; i<=NF; i++) print $i}')

# Iterate over each column
for ((i = 3; i <= $(awk -F, '{print NF}' <<<"$header"); i++)); do
    col=$(awk -F, -v col_idx="$i" '{print $col_idx}' <<<"$header")
    # Create a file for each column in the output directory
    output_file="${output_dir}/${col}.csv"
    
    # Create a table of coaccessibility scores for the specific cell type
    less -S "$input_file" | cut -d, -f1,2,$i | awk -F, '$3 != "NA"' > "$output_file"
done

echo "Files have been created for each cell type with non-NA entries in the ./output/ directory."
