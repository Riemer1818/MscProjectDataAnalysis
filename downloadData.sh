#!/bin/bash

download_dataset() {
    local url="$1"
    local output_dir="$2"
    local output_filename="$3"  # Add a third argument for the output filename

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Check if output filename is provided, otherwise use the basename from the URL
    if [ -z "$output_filename" ]; then
        output_filename=$(basename "$url")
    fi

    # Full path to the output file
    local output_path="$output_dir/$output_filename"

    printf "Downloading dataset from %s\n" "$url"

    # Check if the file already exists
    if [ -f "$output_path" ]; then
        echo "The file $output_path already exists. Skipping download."
    else
        # Download the dataset using curl with the specified output filename
        curl -L "$url" -o "$output_path"
    fi
}


gunzip_download() {
    # Example file to be uncompressed
    local FILE="$1"
    # Destination file
    local DEST="${FILE%.gz}"

    # Check if the destination file already exists
    if [ -f "$DEST" ]; then
        echo "File $DEST already exists. Skipping."
    else
        # Uncompress the file, checking for errors
        if gunzip -c "$FILE" > "$DEST"; then
            echo "Uncompressed $FILE to $DEST successfully."
        else
            echo "Error uncompressing $FILE."
        fi
    fi
}

export -f download_dataset
export -f gunzip_download

# Download datasets

# # minimal TF combinations for inducing granulosa-like cells
# download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE213156&format=file&file=GSE213156%5FTF%5Fgene%5Fexpression%2Ecsv%2Egz" "./data"  "GSE213156_TF_gene_expression.csv.gz"
# gunzip ./data/GSE213156_TF_gene_expression.csv.gz


# # TF atlas
# download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%2Eh5ad%2Egz" "./data" "GSE217460_210322_TFAtlas_differentiated.h5ad.gz"
# download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%5Fraw%2Eh5ad%2Egz" "./data" "GSE217460_210322_TFAtlas_differentiated_raw.h5ad.gz"
# gunzip ./data/GSE217460_210322_TFAtlas_differentiated_raw.h5ad.gz
# gunzip ./data/GSE217460_210322_TFAtlas_differentiated.h5ad.gz

# # Human Fetal Atlas
# download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE156793&format=file&file=GSE156793%5FS4%5Fgene%5Fexpression%5Ftissue%2Etxt%2Egz" "./data" "GSE156793_S4_gene_expression_tissue.txt.gz"
# download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE156793&format=file&file=GSE156793%5FS6%5Fgene%5Fexpression%5Fcelltype%2Etxt%2Egz" "./data" "GSE156793_S6_gene_expression_celltype.txt.gz"
# gunzip ./data/GSE156793_S4_gene_expression_tissue.txt.gz
# gunzip ./data/GSE156793_S6_gene_expression_celltype.txt.gz

# # SUESS dataset(s) 
# download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE107185&format=file" "./data" "GSE107185_RAW.tar"
# tar -xvf ./data/GSE107185_RAW.tar -C ./data
