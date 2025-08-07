import os
import numpy as np
import pandas as pd
import scanpy as sc
from utilsForCompeting import evaluator
import MultiMAP
from sklearn.neighbors import KNeighborsClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1234)

sc.settings.verbosity = 3
sc.logging.print_header()

def main():
    # User input parameters
    exp_id = "Please enter experiment ID"
    peak_count_path = "Please enter Peak count data file path"
    gene_activity_path = "Please enter Gene Activity Scores file path"
    gene_expression_path = "Please enter Gene Expression Matrix file path"
    cache_dir = "Please enter cache directory path"
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Load data
    adata_raw_atac = sc.read_h5ad(peak_count_path)
    adata_atac = sc.read_h5ad(gene_activity_path)
    adata_rna = sc.read_h5ad(gene_expression_path)

    adata_rna.obs['domain'] = 'RNA'
    adata_atac.obs['domain'] = 'ATAC_genes'
    adata_raw_atac.obs['domain'] = 'ATAC_peaks'

    meta_rna = adata_rna.obs.copy()
    meta_atac = adata_atac.obs.copy()
    meta = pd.concat([meta_rna, meta_atac], axis=0)

    # Cache/load X_lsi
    lsi_cache_dir = cache_dir
    os.makedirs(lsi_cache_dir, exist_ok=True)
    x_lsi_path = os.path.join(lsi_cache_dir, f'{exp_id}-ATAC_X_lsi.npy')
    if os.path.exists(x_lsi_path):
        print(f"Loading cached X_lsi: {x_lsi_path}")
        adata_atac.obsm['X_lsi'] = np.load(x_lsi_path)
    else:
        MultiMAP.TFIDF_LSI(adata_raw_atac)
        np.save(x_lsi_path, adata_raw_atac.obsm['X_lsi'])
        adata_atac.obsm['X_lsi'] = adata_raw_atac.obsm['X_lsi'].copy()

    # Cache/load X_pca
    x_pca_path = os.path.join(lsi_cache_dir, f'{exp_id}-RNA_X_pca.npy')
    if os.path.exists(x_pca_path):
        print(f"Loading cached X_pca: {x_pca_path}")
        adata_rna.obsm['X_pca'] = np.load(x_pca_path)
    else:
        rna_pca = adata_rna.copy()
        sc.pp.normalize_total(rna_pca, target_sum=1e4)
        sc.pp.log1p(rna_pca)
        sc.pp.scale(rna_pca)
        sc.pp.pca(rna_pca)
        np.save(x_pca_path, rna_pca.obsm['X_pca'])
        adata_rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()


    MultiMAP.TFIDF_LSI(adata_raw_atac)
    adata_atac.obsm['X_lsi'] = adata_raw_atac.obsm['X_lsi'].copy()

    # Calculate PCA
    rna_pca = adata_rna.copy()
    sc.pp.scale(rna_pca)
    sc.pp.pca(rna_pca)
    adata_rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()

    # MultiMAP integration
    adata = MultiMAP.Integration([adata_rna, adata_atac], ['X_pca', 'X_lsi'])

    print(adata.obsm.keys())

    # KNN classification and evaluation
    X_train = adata[adata.obs['domain'] == 'RNA'].obsm['X_multimap']
    y_train = adata[adata.obs['domain'] == 'RNA'].obs['cell_type'].to_numpy()
    X_test = adata[adata.obs['domain'] == 'ATAC_genes'].obsm['X_multimap']
    y_test = adata[adata.obs['domain'] == 'ATAC_genes'].obs['cell_type'].to_numpy()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # share_mask evaluation
    rna_cell_types = adata_rna.obs['cell_type'].unique() if hasattr(adata_rna.obs['cell_type'], 'unique') else np.unique(adata_rna.obs['cell_type'])
    share_mask = adata_atac.obs['cell_type'].isin(rna_cell_types).to_numpy() if hasattr(adata_atac.obs['cell_type'], 'isin') else np.isin(adata_atac.obs['cell_type'], rna_cell_types)

    kn_data_pr = y_pred[share_mask]
    kn_data_gt = y_test[share_mask]

    closed_acc, f1_score = evaluator(kn_data_pr, kn_data_gt)

if __name__ == "__main__":
    main() 