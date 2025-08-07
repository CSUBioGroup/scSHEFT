import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import os
import pandas as pd
import numpy as np
from os.path import join
from itertools import chain
import sys
sys.path.append('..')
from utilsForCompeting import evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
scglue.plot.set_publication_params()

def main():
    # User input parameters
    exp_id = "Please enter experiment ID"
    peak_count_path = "Please enter Peak count data file path"
    gene_expression_path = "Please enter Gene Expression Matrix file path"
    cache_path = "Please enter cache path"
    gtf_path = "Please enter GTF file path"
    
    # Create cache directory
    os.makedirs(cache_path, exist_ok=True)
    wpath = os.path.join(cache_path, exp_id)
    os.makedirs(wpath, exist_ok=True)

    # Read data
    atac = sc.read_h5ad(peak_count_path)
    rna = sc.read_h5ad(gene_expression_path)

    # Preprocessing
    rna.layers["counts"] = rna.X.copy()

    new_var = pd.DataFrame(index=rna.var['gene_names'])
    rna.var = new_var 
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")

    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")

    # Load LSI results
    x_lsi = np.load(os.path.join(cache_path, f'{exp_id}_lsi.npy'))
    atac.obsm['X_lsi'] = x_lsi

    scglue.data.get_gene_annotation(
        rna, gtf=gtf_path,
        gtf_by="gene_name"
    )

    # Exclude NaN rows
    rna = rna[:, pd.notna(rna.var["chromStart"])].copy()
    print(rna)

    atac.var[['chrom', 'chromStart', 'chromEnd']] = atac.var['peak_names'].str.extract(r'(.*?):(.*?)-(.*)', expand=True)
    print(atac.var.head())

    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    print(guidance)

    scglue.graph.check_graph(guidance, [rna, atac])

    # Configure data
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="X_pca",
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep="X_lsi"
    )
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()

    # Train GLUE model
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": wpath}
    )

    # Embedding
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    atac.obs['cell_type_bkp'] = atac.obs['cell_type'].values
    atac.obs = atac.obs.drop(columns=['cell_type'])

    # KNN classification and evaluation
    from sklearn.neighbors import KNeighborsClassifier

    atac.obs['cell_type'] = atac.obs['cell_type_bkp']

    # Training and test sets
    X_train = rna.obsm["X_glue"]
    y_train = rna.obs["cell_type"].to_numpy() if hasattr(rna.obs["cell_type"], 'to_numpy') else np.array(rna.obs["cell_type"])
    X_test = atac.obsm["X_glue"]
    y_test = atac.obs["cell_type"].to_numpy() if hasattr(atac.obs["cell_type"], 'to_numpy') else np.array(atac.obs["cell_type"])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # share_mask evaluation
    rna_cell_types = rna.obs['cell_type'].unique() if hasattr(rna.obs['cell_type'], 'unique') else np.unique(rna.obs['cell_type'])
    share_mask = atac.obs['cell_type'].isin(rna_cell_types).to_numpy() if hasattr(atac.obs['cell_type'], 'isin') else np.isin(atac.obs['cell_type'], rna_cell_types)

    kn_data_pr = y_pred[share_mask]
    kn_data_gt = y_test[share_mask]

    closed_acc, f1_score = evaluator(kn_data_pr, kn_data_gt)

if __name__ == "__main__":
    main()