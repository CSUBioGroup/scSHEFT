import os
import numpy as np
import scanpy as sc
import sys
sys.path.append('..')
from utilsForCompeting import evaluator
import scanpy.external as sce
from sklearn.neighbors import KNeighborsClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1234)

sc.settings.verbosity = 3
sc.logging.print_header()

def main():
    # User input parameters
    exp_id = "Please enter experiment ID"
    gene_activity_path = "Please enter Gene Activity Scores file path"
    gene_expression_path = "Please enter Gene Expression Matrix file path"
    output_dir = "Please enter output directory path"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    adata_atac = sc.read_h5ad(gene_activity_path)
    adata_rna = sc.read_h5ad(gene_expression_path)

    adata_rna.obs['domain'] = 'RNA'
    adata_atac.obs['domain'] = 'ATAC'

    adata_all = adata_rna.concatenate(adata_atac, batch_key="domain", batch_categories=["RNA", "ATAC"])
    
    # Preprocessing + Scanorama integration
    sc.pp.normalize_total(adata_all)
    sc.pp.log1p(adata_all)
    sc.pp.highly_variable_genes(adata_all, n_top_genes=2000, subset=True, batch_key="domain")
    sc.pp.pca(adata_all, n_comps=50)
    
    sce.pp.scanorama_integrate(adata_all, key="domain", basis="X_pca", adjusted_basis="X_scanorama")
    
    # Split training and test sets
    rna_mask = adata_all.obs["domain"] == "RNA"
    atac_mask = adata_all.obs["domain"] == "ATAC"
    X_train = adata_all.obsm["X_scanorama"][rna_mask]
    y_train = adata_all.obs["cell_type"][rna_mask].values
    X_test = adata_all.obsm["X_scanorama"][atac_mask]
    y_test = adata_all.obs["cell_type"][atac_mask].values

    # KNN classification
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Evaluation
    meta_atac = adata_atac.obs.copy()
    meta_rna = adata_rna.obs.copy()
    share_mask = meta_atac.cell_type.isin(meta_rna.cell_type.unique()).to_numpy()

    kn_data_pr = y_pred[share_mask]
    kn_data_gt = y_test[share_mask]

    closed_acc, f1_score = evaluator(kn_data_pr, kn_data_gt)

if __name__ == "__main__":
    main() 