library(Matrix)
read_dir = "Please enter data directory path"
out_dir = "Please enter output directory path"
source_path = "Please enter data preprocessing utility path"

b1_exprs_filename = "RNA/matrix.mtx"
b2_exprs_filename = "ATAC_GAS/matrix.mtx"

b1_cnames_filename = 'RNA/cnames.csv'
b1_celltype_filename = "RNA/cell_type.csv"
b2_cnames_filename = 'ATAC_GAS/cnames.csv'

b1_gnames_filename = 'RNA/gnames.csv'
b2_gnames_filename = 'ATAC_GAS/gnames.csv'


########################
# read data 

b1_exprs <- readMM(file = paste0(read_dir, b1_exprs_filename))
b2_exprs <- readMM(file = paste0(read_dir, b2_exprs_filename))
b1_meta <- read.table(file = paste0(read_dir, b1_celltype_filename),sep=",",header=T,check.names = F)
# b2_meta <- read.table(file = paste0(read_dir, b2_celltype_filename),sep=",",header=T,row.names=1,check.names = F)

b1_cnames = read.table(file=paste0(read_dir, b1_cnames_filename), sep=',', header=T) 
b2_cnames = read.table(file=paste0(read_dir, b2_cnames_filename), sep=',', header=T)

b1_gnames = read.table(file=paste0(read_dir, b1_gnames_filename), header=T)
b2_gnames = read.table(file=paste0(read_dir, b2_gnames_filename), header=T)

shr_gnames = intersect(b1_gnames$g_names, b2_gnames$g_names)
b1_exprs = as.matrix(t(b1_exprs))
b2_exprs = as.matrix(t(b2_exprs))
rownames(b1_exprs) = b1_gnames$g_names
colnames(b1_exprs) = b1_cnames$c_names
rownames(b2_exprs) = b2_gnames$g_names
colnames(b2_exprs) = b2_cnames$c_names

b1_exprs = b1_exprs[shr_gnames, ]
b2_exprs = b2_exprs[shr_gnames, ]

label1 = data.frame(type=b1_meta$cell_type)
label2 = data.frame(type=b1_meta$cell_type)

count.list = list(b1_exprs, b2_exprs)
label.list = list(label1, label2)

save_processed_data(count.list, label.list, check_unknown=TRUE)