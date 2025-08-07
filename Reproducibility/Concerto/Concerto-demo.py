import os
from concerto_function5_3 import *
import numpy as np
import scanpy as sc
import sys
sys.path.append('..')
from utilsForCompeting import evaluator

# GPU settings
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def main():
    # User input parameters
    exp_id = "Please enter experiment ID"
    gene_activity_path = "Please enter Gene Activity Scores file path"
    gene_expression_path = "Please enter Gene Expression Matrix file path"
    save_path = "Please enter result save path"
    
    # Data loading
    adata_atac = sc.read_h5ad(gene_activity_path)
    adata_rna = sc.read_h5ad(gene_expression_path)
    adata_rna.obs['domain'] = 'RNA'
    adata_atac.obs['domain'] = 'ATAC'

    batch_key = 'domain'
    type_key = 'cell_type'

    adata_all = sc.concat([adata_rna, adata_atac])
    print(adata_all)

    # Preprocessing
    adata = preprocessing_rna(adata_all,
                              min_features=0,
                              n_top_features=None,
                              is_hvg=False,
                              batch_key=batch_key)

    adata_ref = adata[adata.obs[batch_key] == 'RNA']
    adata_query = adata[adata.obs[batch_key] == 'ATAC']

    shr_mask = np.in1d(adata_query.obs[type_key], adata_ref.obs[type_key].unique())
    atac_lab = np.array(adata_query.obs[type_key].values)

    # Create save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ref_tf_path = concerto_make_tfrecord_supervised(adata_ref, tf_path = save_path + f'tfrecord/{exp_id}/ref_tf/',
                                         batch_col_name = batch_key, label_col_name=type_key)
    query_tf_path = concerto_make_tfrecord_supervised(adata_query, tf_path = save_path + f'tfrecord/{exp_id}/query_tf/',
                                         batch_col_name = batch_key, label_col_name=type_key)

    # train (leave spleen out). If you don't want to train the model, you can just load our trained classifier's weight and test it directly.
    weight_path = save_path + f'weight/{exp_id}/'
    ref_tf_path = save_path + f'tfrecord/{exp_id}/ref_tf/'
    query_tf_path = save_path + f'tfrecord/{exp_id}/query_tf/'

    concerto_train_inter_supervised_uda2(ref_tf_path, query_tf_path, weight_path,
                                         super_parameters={'batch_size': 128, 'epoch_pretrain': 1,'epoch_classifier': 10, 'lr': 1e-4,'drop_rate': 0.1})

    # test (only spleen)
    weight_path = save_path + f'weight/{exp_id}/'
    ref_tf_path = save_path + f'tfrecord/{exp_id}/ref_tf/'
    query_tf_path = save_path + f'tfrecord/{exp_id}/query_tf/'

    # Test model
    for epoch in [4]:
        results = concerto_test_inter_supervised2(weight_path, ref_tf_path, query_tf_path,
                                                  super_parameters={'batch_size': 64, 'epoch': epoch, 'lr': 1e-5,
                                                                    'drop_rate': 0.1})

        # KNN classifier
        query_neighbor, query_prob = knn_classifier(results['source_feature'],
                                                    results['target_feature'],
                                                    adata_ref,
                                                    adata_ref.obs_names,
                                                    column_name=type_key,
                                                    k=30)
        kn_data_pr = query_neighbor[shr_mask]
        kn_data_gt = atac_lab[shr_mask]

        closed_acc, f1_score = evaluator(kn_data_pr, kn_data_gt)

    # Neural network classifier
    query_pred, query_prob = results['target_pred'], results['target_prob']
    closed_acc, f1_score = evaluator(query_pred, atac_lab)

if __name__ == "__main__":
    main()