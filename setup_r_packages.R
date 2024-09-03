# Ensure BiocManager is installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Install Bioconductor packages
BiocManager::install("EnsDb.Hsapiens.v75")
BiocManager::install("Signac")
BiocManager::install("Seurat")
BiocManager::install("hdf5r")
BiocManager::install("biovizBase")
BiocManager::install('glmGamPoi')

# Ensure devtools is installed
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

# Install loomR from GitHub
devtools::install_github(repo = "mojaveazure/loomR", ref = "develop")

# Install Seurat if not already installed
if (!requireNamespace("Seurat", quietly = TRUE)) {
  install.packages("Seurat")
}

if (!requireNamespace("remotes")) install.packages("remotes")
remotes::install_github("GfellerLab/SuperCell")

install.packages("units")
install.packages("Cairo")
install.packages("ggrastr")
install.packages("sf")
install.packages("spdep")


# devtools::install_github("cole-trapnell-lab/monocle3")

# Install Cicero
if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes")
remotes::install_github("cole-trapnell-lab/cicero-release", ref = "monocle3")

