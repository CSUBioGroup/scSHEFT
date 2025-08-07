import numpy as np
import scanpy as sc
import scnym
import os
from os.path import join
import scipy.sparse as sps
import sys
sys.path.append('..')
from utilsForCompeting import evaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # User input parameters
    exp_id = "Please enter experiment ID"
    gene_activity_path = "Please enter Gene Activity Scores file path"
    gene_expression_path = "Please enter Gene Expression Matrix file path"
    output_path = "Please enter output path"
    
    binz = False
    new_ident = 'no_new_identity'  # no_new_identity

    test_adata = sc.read_h5ad(gene_activity_path)
    train_adata = sc.read_h5ad(gene_expression_path)
    test_adata.obs['cell_type_bkp'] = test_adata.obs.cell_type.values

    if binz:
        train_adata.X = (train_adata.X>0).astype('float32')
        test_adata.X = (test_adata.X>0).astype('float32')
    print('%d cells, %d genes in the training set.' % train_adata.shape)
    print('%d cells, %d genes in the target set.' % test_adata.shape)

    # Preprocess datasets
    sc.pp.normalize_total(train_adata, target_sum=1e6)
    sc.pp.log1p(train_adata)
    sc.pp.normalize_total(test_adata, target_sum=1e6)
    sc.pp.log1p(test_adata)

    # Set test data cells to target data token "Unlabeled"
    test_adata.obs["cell_type"] = "Unlabeled"
    # Concatenate training and test data into a single object
    adata = train_adata.concatenate(test_adata)

    scnym.api.scnym_api(
        adata=adata,
        task="train",
        groupby="cell_type",
        config=new_ident,
        out_path=os.path.join(output_path, f'{exp_id}_binz={binz}_{new_ident}'),
    )
    scnym.api.scnym_api(
        adata=adata,
        task='predict',
        trained_model=os.path.join(output_path, f'{exp_id}_binz={binz}_{new_ident}'),
    )
    
    # Copy scNym predictions to original test data embedding
    test_adata.obs['scNym'] = np.array(adata.obs.loc[[x + '-1' for x in test_adata.obs_names], 'scNym'])
    train_adata.obs['scNym'] = np.array(adata.obs.loc[[x + '-0' for x in train_adata.obs_names], 'scNym'])
    test_adata.obs['max_prob'] = np.array(adata.obs.loc[[x + '-1' for x in test_adata.obs_names], 'scNym_confidence'])
    train_adata.obs['max_prob'] = np.array(adata.obs.loc[[x + '-0' for x in train_adata.obs_names], 'scNym_confidence'])
    test_adata.obsm['X_scnym'] = adata[train_adata.shape[0]:].obsm['X_scnym'].copy()

    prediction = np.array(test_adata.obs['scNym'])
    ground_truth = np.array(test_adata.obs['cell_type_bkp'])
    closed_acc, f1_score = evaluator(prediction, ground_truth)

if __name__ == "__main__":
    main()
