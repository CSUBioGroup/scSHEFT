import os
import numpy as np
import pandas as pd
import scanpy as sc
import sys
sys.path.append('..')
from utilsForCompeting import evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1234)

sc.settings.verbosity = 3
sc.logging.print_header()

def main():
    # User input parameters
    gene_activity_path = "Please enter Gene Activity Scores file path"
    gene_expression_path = "Please enter Gene Expression Matrix file path"
    exp_id = "Please enter experiment ID"
    result_path = "Please enter result save path"
    
    adata_atac = sc.read_h5ad(gene_activity_path)
    adata_rna_facs = sc.read_h5ad(gene_expression_path)

    meta_rna = adata_rna_facs.obs
    meta_atac = adata_atac.obs
    meta = pd.concat([meta_rna, meta_atac], axis=0)

    # Integration using Portal
    from src.model import Model

    # Create a folder for saving results
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Standard portal pipeline
    model = Model(training_steps=2000,
                  lambdacos=10., lambdaAE=10., lambdaLA=10., lambdaGAN=1.0)
    model.preprocess(adata_rna_facs, adata_atac, norm=True)  # Perform preprocess and PCA
    model.train()  # train the model
    model.eval()  # get integrated latent representation of cells

    from knn_classifier import knn_classifier_top_k, faiss_knn, knn_classifier_prob_concerto
    rna_lab = np.array(adata_rna_facs.obs.cell_type.values)
    atac_lab = np.array(adata_atac.obs.cell_type.values)
    feat_A, feat_B = model.latent[:len(rna_lab)], model.latent[len(rna_lab):]

    # KNN classifier
    atac_pred, atac_prob = knn_classifier_prob_concerto(feat_A, feat_B, rna_lab, n_sample=None, knn=30, num_chunks=100)

    shr_mask = np.in1d(atac_lab, np.unique(rna_lab))
    print((np.ravel(atac_pred)[shr_mask] == atac_lab[shr_mask]).mean())

    closed_acc, f1_score = evaluator(atac_pred, atac_lab)

if __name__ == "__main__":
    main()