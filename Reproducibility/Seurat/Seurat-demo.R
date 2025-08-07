library(Seurat)
library(glue)
library(Matrix)

# User input parameters
read_dir <- "Please enter data directory path"
output_dir <- "Please enter output directory path"

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

b1_exprs_filename = "matrix.mtx"
b2_exprs_filename = "matrix.mtx"
b1_cnames_filename = 'cnames.csv'
b2_cnames_filename = 'cnames.csv'
b1_celltype_filename = "cell_type.csv"
b2_celltype_filename = "cell_type.csv"
gnames_filename    = 'gnames.csv'

########################
# Read data

b1_exprs <- readMM(file = paste0(read_dir, 'RNA/', b1_exprs_filename))
b2_exprs <- readMM(file = paste0(read_dir, 'ATAC_GAS/', b2_exprs_filename))
b1_meta <- read.table(file = paste0(read_dir, 'RNA/', b1_celltype_filename),sep=",",header=T,check.names = F)
b2_meta <- read.table(file = paste0(read_dir, 'ATAC_GAS/', b2_celltype_filename),sep=",",header=T,check.names = F)

b1_cnames = read.table(file=paste0(read_dir, 'RNA/', b1_cnames_filename), sep=',', header=T) 
b2_cnames = read.table(file=paste0(read_dir, 'ATAC_GAS/', b2_cnames_filename), sep=',', header=T)
gnames = read.table(file=paste0(read_dir, 'RNA/', gnames_filename), header=T, sep=',')

b1_exprs = as.matrix(t(b1_exprs))
b2_exprs = as.matrix(t(b2_exprs))
rownames(b1_exprs) = gnames$g_names
colnames(b1_exprs) = b1_cnames$c_names
rownames(b2_exprs) = gnames$g_names
colnames(b2_exprs) = b2_cnames$c_names

label1 = b1_meta$cell_type
label2 = b2_meta$cell_type

rna.obj = CreateSeuratObject(counts=b1_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
rna.obj@meta.data[['cell_type']] = label1
rna.obj = NormalizeData(rna.obj, verbose = FALSE)
rna.obj = FindVariableFeatures(rna.obj, selection.method = "vst", nfeatures = 4000,
        		verbose = FALSE)
rna.obj = ScaleData(rna.obj, verbose=FALSE)
rna.obj = RunPCA(rna.obj, npcs = 30, verbose = FALSE)

# atac objects 2
# Normalize and scale for finding anchors
atac.obj = CreateSeuratObject(counts=b2_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
atac.obj = NormalizeData(atac.obj, verbose = FALSE)
atac.obj = ScaleData(atac.obj, verbose=FALSE)
atac.obj = RunPCA(atac.obj, features=VariableFeatures(object = rna.obj), npcs = 30, verbose = FALSE)

anchors <- FindTransferAnchors(reference = rna.obj, query = atac.obj,
    features = VariableFeatures(object = rna.obj),
    reduction='cca')

atac.pred <- TransferData(anchorset = anchors, refdata = rna.obj$cell_type,
    weight.reduction = atac.obj[["pca"]], dims = 1:30)

atac.obj <- AddMetaData(atac.obj, metadata = atac.pred)
atac.obj@meta.data[['cell_type']] = label2

predicted_labels <- atac.obj$predicted.id
true_labels <- atac.obj$cell_type

n_corr = length(which(predicted_labels == true_labels))
n_incorr = length(true_labels) - n_corr
accuracy = n_corr / (n_corr + n_incorr)
print(paste("Accuracy:", accuracy))

# Save results for Further evaluation
write.csv(atac.pred, file=paste0(output_dir, '/seurat_results.csv'), quote=F, row.names=T)

