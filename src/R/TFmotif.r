library(Seurat)
library(ggplot2)
library(patchwork)
library(Signac)
library(EnsDb.Hsapiens.v75)
library(hdf5r)
library(biovizBase)
library(Matrix)

"""what I am doing here is the atac-seq data analysis, which is the chromatin accessibility data. In order to get the predicted TF motifs
that I can use to compare with the scRNA-seq data. """

output_dir <- "~/data/output/"

# Load the Seurat object
file_path <- "~/data/GSM4508930_cerebrum_filtered.seurat.RDS"
multiomic_data <- readRDS(file = file_path)

# Update the Seurat object if necessary
multiomic_data <- UpdateSeuratObject(multiomic_data)

class(multiomic_data[['peaks']])

multiomic_data[['peaks']] <- CreateChromatinAssay(
    counts = GetAssayData(multiomic_data[['peaks']], slot = "counts"),
    genome = 'hg38',
    sep = c(":", "-"),
    min.cells = 10,
    min.features = 200
)

# find the fragments data "metadata <- GetAssayData(multiomic_data[['peaks']])" to ensure present.

class(multiomic_data[['peaks']])

peaks <- granges(multiomic_data[['peaks']])
head(peaks)


# > head(peaks)
# GRanges object with 6 ranges and 0 metadata columns:
#       seqnames        ranges strand
#          <Rle>     <IRanges>  <Rle>
#   [1]     chr1    9992-10688      *
#   [2]     chr1   14831-15063      *
#   [3]     chr1   17351-17617      *
#   [4]     chr1   29200-29505      *
#   [5]     chr1 237621-237897      *
#   [6]     chr1 521419-521715      *
#   -------
#   seqinfo: 24 sequences from an unspecified genome; no seqlengths


# Ensure the annotations have been correctly loaded and formatted
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v75)
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- "hg38"

# Access the ChromatinAssay object directly
chrom_assay <- multiomic_data[['peaks']]

# Assign the annotations
chrom_assay@annotation <- annotations # have to do manually because addAnnotations doesn't work...

# Reassign the modified ChromatinAssay back to the Seurat object
multiomic_data[['peaks']] <- chrom_assay

# verify
head(Annotation(multiomic_data[["peaks"]]))

# normalization etc
multiomic_data <- RunTFIDF(multiomic_data)
multiomic_data <- FindTopFeatures(multiomic_data, min.cutoff = 'q0')
multiomic_data <- RunSVD(multiomic_data)

multiomic_data <- RunUMAP(object = multiomic_data, reduction = 'lsi', dims = 2:30)
multiomic_data <- FindNeighbors(object = multiomic_data, reduction = 'lsi', dims = 2:30)
multiomic_data <- FindClusters(object = multiomic_data, verbose = FALSE, algorithm = 3)

# Save the plot
plot <- DimPlot(object = multiomic_data, label = TRUE) + NoLegend()
output_file <- paste0(output_dir, basename(file_path), "_dim_plot.png")
ggsave(filename = output_file, plot = plot)

# making gene activity matrix
gene.activities <- GeneActivity(multiomic_data)

# add the gene activity matrix to the Seurat object as a new assay and normalize it
multiomic_data[['RNA_pred']] <- CreateAssayObject(counts = gene.activities)

multiomic_data <- NormalizeData(
  object = multiomic_data,
  assay = 'RNA',
  normalization.method = 'LogNormalize',
  scale.factor = median(multiomic_data$nCount_RNA)
)

# DefaultAssay(multiomic_data) <- 'RNA'

# # this looks like complete garbage 
# FeaturePlot(
#   object = multiomic_data,
#   features = c('MS4A1', 'CD3D', 'LEF1', 'NKG7', 'TREM1', 'LYZ'),
#   pt.size = 0.1,
#   max.cutoff = 'q95',
#   ncol = 3
# )

## probably the next step would be the cross-modality integration (with scRNA set from HFAcao) 
## according to https://stuartlab.org/signac/articles/pbmc_vignette


## so we now have the gene activities fromthe HFAdomcke. We can now use this to link the predicted TF activities (not yet here) to the HFAdomcke to the HFAcao dataset

## so probably cicero or celloracle