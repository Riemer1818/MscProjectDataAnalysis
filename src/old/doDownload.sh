source downloadData.sh


mkdir -p ./data
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%2Eh5ad%2Egz" "/data" "GSE217460_210322_TFAtlas_differentiated.h5ad.gz"
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE217460&format=file&file=GSE217460%5F210322%5FTFAtlas%5Fdifferentiated%5Fraw%2Eh5ad%2Egz" "/data" "GSE217460_210322_TFAtlas_differentiated_raw.h5ad.gz"
gunzip ~/data/GSE217460_210322_TFAtlas_differentiated_raw.h5ad.gz
gunzip ~/data/GSE217460_210322_TFAtlas_differentiated.h5ad.gz


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

# A human cell atlas of fetal chromatin accessibility Domcke
# ATAC-seq data and RNA-seq data
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE149683&format=file" "./MscProjectDataAnalysis/data" "GSE149683_RAW.tar"
# Chromatin accessibility data
download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE149683&format=file&file=GSE149683%5FFile%5FS5%2ECicero%5Fcoaccessibility%5Fscores%5Fby%5Fcell%5Ftype%2Ecsv%2Egz"  "./RIEMERPI001/data" "GSE149683_File_S5_Cicero_coaccessibility_scores_by_cell_type.csv.gz"

# a human cell atlas CAO

download_dataset "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE156793&format=file&file=GSE156793%5FS3%5Fgene%5Fcount%2Eloom%2Egz" "./RIEMERPI001/data" "GSE156793_S3_gene_count.loom.gz"