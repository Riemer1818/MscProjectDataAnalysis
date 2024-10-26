library(zellkonverter) #BiocManager::install("zellkonverter")
library(cellula)
# install cellula using devtools::install_github("gdagstn/cellula")
library(SingleCellExperiment)  #BiocManager::install("SingleCellExperiment")
library(ComplexHeatmap) #BiocManager::install("ComplexHeatmap")
library(BiocParallel) #BiocManager::install("BiocParallel")

# Loading files and converting them to SingleCellExperiment format
sce <- readH5AD(file = "data/GSE217460_210322_TFAtlas_differentiated_raw.h5ad")
sce2 <- readH5AD(file = "data/GSE217460_210322_TFAtlas_differentiated_raw.h5ad")

# Wrangling
assay(sce2, "counts") <- as(assay(sce2, "X"), "SparseMatrix")
assay(sce2, "X") <- NULL

reducedDim(sce2, "UMAP_Orig") <- reducedDim(sce, "X_umap")
reducedDim(sce2, "PCA") <- reducedDim(sce, "X_pca")
reducedDim(sce2, "PCA_Harmony") <- reducedDim(sce, "X_pca_harmony")
reducedDim(sce2, "DiffusionMap") <- reducedDim(sce, "X_diffmap")

rm(sce)

gc()

saveRDS(sce2, file = "/Users/gdagostino/Desktop/research/Plasticell/datasets/TF_Reprogramming/Joung_TF_Atlas_2023/SCE_TFatlas_Differentiating.RDS")
sce2 <- readRDS("/Users/gdagostino/Desktop/research/Plasticell/datasets/TF_Reprogramming/Joung_TF_Atlas_2023/SCE_TFatlas_Differentiating.RDS")



plot_UMAP(sce2, "UMAP_Orig", color_by = "difflouvain")

#um <- uwot::umap(reducedDim(sce2, "PCA_Harmony")[,1:20], n_neighbors = 30, min_dist = 0.3)
#reducedDim(sce2, "UMAP_Harmony") <- um
#plot_UMAP(sce2, "UMAP_Harmony", color_by = "difflouvain")

#plot_UMAP(sce2, "UMAP_Harmony", color_by = "difflouvain")

#sce2 <- makeGraphsAndClusters(sce2, neighbors = 30, method = "leiden", 
#                              dr = "PCA_Harmony", save_graphs = TRUE, k = c(0.5, 1))

rowData(sce2) = DataFrame("ID" = rownames(sce2), "Symbol" = rownames(sce2))
old_pca = reducedDim(sce2, "PCA")
assay(sce2, "counts") <- as(assay(sce2, "counts"), "matrix")
assay(sce2, "counts") <- as(assay(sce2, "counts"), "dgCMatrix")
sce2 = doNormAndReduce(sce2, batch = "batch", verbose = TRUE)

plot_UMAP(sce2, "UMAP_Orig", "TP53")

tf = "PAX2"
sce2$has_TF = factor(sce2$TF_only == tf, levels = c("FALSE", "TRUE"))
nc = table(sce2$has_TF)[2]
p = plot_UMAP(sce2[,order(sce2$has_TF)], "UMAP_Orig", color_by = "has_TF", 
              point_size = 0.5, outline = FALSE, color_palette = c("gray", "midnightblue")) + 
  labs(title = paste0(tf, " (", nc, " cells)"))
p



sce2$TF_only = unlist(lapply(strsplit(as.character(sce2$TF), "-"), \(x) x[2]))

pheatmap(log(as.matrix(table(sce2$TF_only, sce2$difflouvain)) + 1))

pheatmap(log(as.matrix(table(sce2$TF_only, sce2$difflouvain))[unique(sort(table(sce2$TF_only), decreasing = TRUE)[1:100]),] + 1))

ent = apply(as.matrix(table(sce2$TF_only, sce2$difflouvain)), 1, \(x) cellula:::.getEntropy(x))
ent * rowMeans(as.matrix(table(sce2$TF_only, sce2$difflouvain)))
topTFs = sort((max(ent)-ent) * rowMeans(as.matrix(table(sce2$TF_only, sce2$difflouvain))), decreasing = TRUE)

pheatmap(log(as.matrix(table(sce2$TF_only, sce2$difflouvain))[names(topTFs[1:100]),] + 1))



#udf = reducedDim(sce2, "UMAP_Harmony")[,1:2]
#udf[,1] = cellula:::.rescalen(udf[,1])
#udf[,2] = cellula:::.rescalen(udf[,2])
#over = oveRlay::makeOverlay(udf, res = 100)
#over = over[over$hole == "outer",]


plist = lapply(names(topTFs)[10:18], function(tf) {
sce2$has_TF = factor(sce2$TF_only == tf, levels = c("FALSE", "TRUE"))
nc = table(sce2$has_TF)[2]
p = plot_UMAP(sce2[,order(sce2$has_TF)], "UMAP_Orig", color_by = "has_TF", 
          point_size = 0.5, outline = FALSE, color_palette = c("gray", "midnightblue")) + 
  labs(title = paste0(tf, " (", nc, " cells)"))
p
})
ggarrange(plotlist = plist, ncol = 3, nrow = 3)


tf = "CDX2"
sce2$has_TF = factor(sce2$TF_only == tf, levels = c("FALSE", "TRUE"))
nc = table(sce2$has_TF)[2]
plot_UMAP(sce2[,order(sce2$has_TF)], "UMAP_Orig", color_by = "has_TF", 
              point_size = 0.5, outline = FALSE, color_palette = c("gray", "midnightblue")) + 
  labs(title = paste0(tf, " (", nc, " cells)"))

knn = BiocNeighbors::findKNN(reducedDim(sce2, "PCA_Harmony")[,1:20], k = 200)

tf_knn = apply(knn$index, 1, \(x) sce2$TF_only[x])
tf_knn = t(tf_knn)
rownames(tf_knn) = colnames(sce2)

head(tf_knn)
colnames(tf_knn) = paste0("KNN_", seq_len(200))
tf_knn = as.data.frame(tf_knn)
tf_knn[,"TF"] = sce2$TF_only
tf_knn$TF_max = apply(tf_knn[,1:200], 1, \(x) names(which.max(table(x))))
tf_knn$TF_mscore = apply(tf_knn[,1:200], 1, \(x) max(table(x))/200)

sce2$TF_max = tf_knn$TF_max
sce2$TF_mscore = tf_knn$TF_mscore

plot_UMAP(sce2, "UMAP_Orig", color_by = "TF_mscore", point_size = 2, label_by = "difflouvain")

tf_knnl = split(tf_knn[,], sce2$TF_only)

for(i in seq_along(tf_knnl)) {
  tf_knnl[[i]]$TF_max = apply(tf_knnl[[i]][,1:10], 1, \(x) names(which.max(table(x))))
  tf_knnl[[i]]$TF_mscore = apply(tf_knnl[[i]][,1:10], 1, \(x) max(table(x))/10)
}

tf_knnl2 = lapply(tf_knnl, \(x) apply(x, 1, \(y) names(which.max(table(y[1:10])))))

tf_knnl_tabs <- lapply(tf_knnl, \(x) sort(table(as.matrix(x[,1:10])), decreasing = TRUE))

tf_probs = lapply(tf_knnl, \(x) table(x$TF_max)/sum(table(x$TF_max)))

ll = lapply(tf_probs, \(x) data.frame("to" = names(x), "weight" = as.numeric(x)))
for(i in seq_along(ll)) {
  ll[[i]]$from = names(ll)[i]
  ll[[i]] = ll[[i]][,c("from", "to", "weight")]
}

tfdf = as.data.frame(do.call(rbind,ll))
tfdf = tfdf[tfdf$from %in% names(table(tfdf$from))[table(tfdf$from) > 20],]
tfdf = tfdf[tfdf$weight > 0.1,]

gtf = igraph::graph_from_data_frame(tfdf, directed = FALSE)

ggraph(gtf, layout = "fr") + 
  geom_node_point() + 
  geom_edge_link(aes(width = weight, alpha = weight)) +
  geom_node_label(aes(label = name), size = 2)
ass = assay(sce2, "counts")
vec = sce2$TF_only 

spl = bplapply(unique(sce2$TF_only), \(x) {
  message("Doing ", x, " - ", which(unique(sce2$TF_only) == x), " of ", length(unique(sce2$TF_only)))
  rowMeans(ass[,vec == x,drop=FALSE])
}, BPPARAM = MulticoreParam(workers = 8, progressbar = TRUE))


splmat <- as(do.call(cbind, spl), "dgCMatrix")
colnames(splmat) = unique(sce2$TF_only)

cc = cor(as.matrix(splmat), method = "spearman")
dc = as.dist(1 - cc)
hc = hclust(dc)
clus = dynamicTreeCut::cutreeDynamic(hc, distM = as.matrix(dc), minClusterSize = 20)
max(clus)
split(unique(sce2$TF_only), clus2)
