# Import necessary libraries
import anndata
import scanpy as sc
import umap
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

# Define paths and constants at the top
RAW_DATA_PATH = '/home/msai/riemerpi001/GSE217460_210322_TFAtlas_differentiated_raw.h5ad'
PROCESSED_DATA_PATH = '/home/msai/riemerpi001/GSE217460_210322_TFAtlas_differentiated.h5ad'
TF = "PAX2" # for sanity check
COUNT_VAR = 5
ENTROPY_VAR = 2.5

# Load the data
adata = sc.read_h5ad(RAW_DATA_PATH)
tdata = sc.read_h5ad(PROCESSED_DATA_PATH)

# Copy PCA embeddings from tdata to adata
adata.obsm["X_pca"] = tdata.obsm['X_pca']
adata.obsm["X_pca_harmony"] = tdata.obsm["X_pca_harmony"]

# Clean up memory by deleting tdata
del tdata

# Preprocessing for TF
adata.obs["TF"] = adata.obs["TF"].astype(str)  # Ensure 'TF' column is a string
adata.obs["New_TF"] = adata.obs["TF"].str.split("-").str[-1]  # Extract the part after '-'

# Sanity check for TF
if TF in adata.obs["New_TF"].unique():
    print("TF is in the list")
else:
    print("TF is not in the list")

# Replace the 'TF' column with new categories and clean up
adata.obs["TF"] = adata.obs["New_TF"]
adata.obs = adata.obs.drop(columns=["New_TF"])

# Replace batch numbers with labels
adata.obs["batch"] = adata.obs["batch"].replace({0: "TFatlas_1", 1: "TFatlas_2"})

# Make the data matrix sparse if it isn't already
if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)

# Function to calculate entropy for each TF
def calculate_entropy_per_tf(adata):
    # Count the frequency of each TF in each cluster
    frequency_table = adata.obs.groupby(['TF', 'difflouvain']).size().unstack(fill_value=0)
    
    # Convert counts to probabilities
    probabilities = frequency_table.div(frequency_table.sum(axis=1), axis=0)
    
    # Calculate entropy for each TF
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9), axis=1)  # Adding a small value to avoid log(0)
    
    return entropy

# Function to plot UMAP based on PCA data
def plot_umap(adata, pca_key='X_pca', umap_key='umap', title_prefix=''):
    reducer = umap.UMAP()  # Initialize UMAP reducer
    embedding = reducer.fit_transform(adata.obsm[pca_key])  # Fit UMAP using PCA data
    adata.obsm[umap_key] = embedding  # Store UMAP embedding back into adata
    sc.pl.umap(adata, color='difflouvain', title=f"{title_prefix} UMAP with {pca_key} as basis", legend_loc="on data")

# Function to select samples based on counts and entropy
def select_samples(adata, count_var=COUNT_VAR, entropy_var=ENTROPY_VAR):
    # Filter TFs based on counts greater than count_var
    adata = adata[adata.obs['TF'].isin(adata.obs['TF'].value_counts()[adata.obs['TF'].value_counts() > count_var].index), :]
    
    # Filter TFs based on entropy less than entropy_var
    entropy_per_tf = calculate_entropy_per_tf(adata)
    adata = adata[adata.obs['TF'].isin(entropy_per_tf[entropy_per_tf < entropy_var].index), :]
    
    return adata

# Function for preprocessing the data
def preprocess(adata):
    sc.pp.filter_cells(adata, min_genes=200)  # Filter cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=3)  # Filter genes present in fewer than 3 cells
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize the data
    sc.pp.log1p(adata)  # Log-transform the data
    return adata

# Normalize and preprocess the data
adata = preprocess(adata)

# Print initial number of samples and TFs
initial_sample_count = adata.shape[0]
initial_tf_count = len(adata.obs['TF'].unique())
print(f"Initial number of samples: {initial_sample_count}")
print(f"Initial number of unique TFs: {initial_tf_count}")

# Plot UMAP for regular PCA
plot_umap(adata, pca_key='X_pca', umap_key='umap', title_prefix='Regular')

# Plot UMAP for Harmony integrated PCA
plot_umap(adata, pca_key='X_pca_harmony', umap_key='X_umap_PCA_Harmony', title_prefix='Harmony')

# Calculate entropy per TF and plot histogram
entropy_per_tf = calculate_entropy_per_tf(adata).sort_values(ascending=False)
entropy_per_tf.plot(kind='bar', figsize=(10, 5), color='skyblue', title='Entropy per TF over clusters')

# Count the number of samples per TF and plot the bar chart
sample_counts = adata.obs['TF'].value_counts()

plt.figure(figsize=(20, 10))  # Increase figure width for better readability
entropy_per_tf.plot(kind='bar', color='skyblue')
plt.title('Entropy per TF over clusters')
plt.xlabel('TF')
plt.ylabel('Entropy')
plt.xticks([])  # Remove the x-axis tick labels to avoid overlap
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
plt.savefig('entropy_per_tf_plot.png', bbox_inches='tight', dpi=600)

plt.show()

# Select samples based on count and entropy thresholds
adata = select_samples(adata)

# Print final number of samples and TFs after filtering
final_sample_count = adata.shape[0]
final_tf_count = len(adata.obs['TF'].unique())
print(f"Final number of samples: {final_sample_count}")
print(f"Final number of unique TFs: {final_tf_count}")

# Plot the number of selected TFs after filtering
adata.obs['TF'].value_counts().plot(kind='bar', figsize=(10, 5), color='skyblue', title='TFs as a function of value_counts()')
