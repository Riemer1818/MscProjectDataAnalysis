import anndata
# import pandas as pd
# import csv

# import cellrank as  cr
import scanpy as sc

# from scipy import sparse

# import umap
# import copy as cp
    
import scipy.sparse as sp

# import numpy as np
# import matplotlib.pyplot as plt
# import loompy
# import celloracle as co


# # Load the LOOM file
# loom_path = "/home/msai/riemerpi001/MscProjectDataAnalysis/data/GSE156793_S3_gene_count.loom"

# # Connect to the loom file
# # with loompy.connect(loom_path) as ds:
#     # Convert the loom dataset to an AnnData object
# adata = sc.read_loom(loom_path)
    
adata = sc.read_h5ad("/home/msai/riemerpi001/MscProjectDataAnalysis/data/GSE156793_S3_gene_count.h5ad")

if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)
else:
    adata.X = adata.X

adata.write("/home/msai/riemerpi001/MscProjectDataAnalysis/data/GSE156793_S3_gene_count_sparse.h5ad")