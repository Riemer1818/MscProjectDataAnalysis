library(Seurat)
library(ggplot2)
library(patchwork)
library(Signac)
library(EnsDb.Hsapiens.v75)
library(hdf5r)
library(biovizBase)
library(SuperCell)
library(Matrix)


multiomic_data <- readRDS(file = "cerebellum_filtered.seurat.for_website.RDS") # should be system argument maybe

multiomic_data <- UpdateSeuratObject(multiomic_data)