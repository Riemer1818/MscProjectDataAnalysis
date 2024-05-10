source downloadData.sh


mkdir -p ./data
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%2Eh5ad%2Egz" "./data" "GSE217460_210322_TFAtlas_differentiated.h5ad.gz"
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%5Fraw%2Eh5ad%2Egz" "./data" "GSE217460_210322_TFAtlas_differentiated_raw.h5ad.gz"
gunzip ./data/GSE217460_210322_TFAtlas_differentiated_raw.h5ad.gz
gunzip ./data/GSE217460_210322_TFAtlas_differentiated.h5ad.gz


mkdir -p ./data/SUESS
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE107185&format=file" "./data" "GSE107185_RAW.tar"
tar -xvf ./data/GSE107185_RAW.tar -C ./data/SUESS

# Check if the target directory was provided
if [[ -z "./data/SUESS" ]]; then
    echo "Usage: $0 <target-directory>"
    exit 1
fi

cd "./data/SUESS" || exit

# Find and decompress all .gz files in the directory
for file in *.gz; do
    if [[ -f "$file" ]]; then
        echo "Decompressing $file"
        gunzip "$file"
    else
        echo "No .gz files found in $TARGET_DIR"
        break
    fi
done