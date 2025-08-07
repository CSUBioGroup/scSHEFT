import numpy as np
import scanpy as sc
import umap
from sklearn.metrics import (
    accuracy_score, f1_score
)
from scipy.spatial.distance import cdist
from annoy import AnnoyIndex

def preprocess_data(rna, atac, hvg_num=None, binz=False, lognorm=[False, True], scale_per_batch=True):
    print("Finding highly variable genes...")
    
    if hvg_num and hvg_num <= rna.shape[1]:
        sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(atac, flavor='seurat_v3', n_top_genes=hvg_num)
        
        hvg_rna = rna.var[rna.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_atac = atac.var[atac.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_total = list(set(hvg_rna) & set(hvg_atac))
        
        if len(hvg_total) < 100:
            raise ValueError(f"The total number of highly variable genes is smaller than 100 ({len(hvg_total)}). Try to set a larger hvg_num.")
    else:
        hvg_total = list(rna.var_names[:hvg_num] if hvg_num else rna.var_names)
    
    # Subset genes
    rna = rna[:, hvg_total].copy()
    atac = atac[:, hvg_total].copy()
    
    # Binarization
    if binz:
        rna.X = (rna.X > 0).astype('float')
        atac.X = (atac.X > 0).astype('float')
    
    # Log normalization
    if lognorm[0]:  # RNA
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
    
    if lognorm[1]:  # ATAC
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
    
    # Scaling
    if scale_per_batch:
        sc.pp.scale(rna, max_value=10)
        sc.pp.scale(atac, max_value=10)
    
    return rna, atac

def find_nearest_neighbors(data, query=None, k=10, metric='manhattan', n_trees=10):
    query = query if query is not None else data
    
    # Build index
    a = AnnoyIndex(data.shape[1], metric=metric)
    for i in range(data.shape[0]):
        a.add_item(i, data[i, :])
    a.build(n_trees)
    
    # Search index
    ind = []
    for i in range(query.shape[0]):
        ind.append(a.get_nns_by_vector(query[i, :], k, search_k=-1))
    return np.array(ind)

def find_anchor_points(f_A, f_B, k=5):
    dis_matrix = cdist(f_A, f_B, metric="cosine")
    top_k_row_indices = np.argsort(dis_matrix, axis=1)[:, :k]
    top_k_column_indices = np.argsort(dis_matrix.T, axis=1)[:, :k]
    
    A = np.array(top_k_row_indices)
    B = np.array(top_k_column_indices)
    a, k = A.shape
    b, _ = B.shape
    
    # Build reverse mapping
    reverse_mapping = {i: set() for i in range(a)}
    for i in range(b):
        for j in range(k):
            reverse_mapping[B[i, j]].add(i)
    
    result = set()
    for w in range(a):
        for v in A[w]:
            if w in reverse_mapping and v in reverse_mapping[w]:
                result.add((w, v))
    
    print(f"Finding {len(result)} anchors")
    
    # Filter results
    filtered_A_B_inds = [[] for _ in range(a)]
    filtered_B_A_inds = [[] for _ in range(b)]
    
    for i, j in result:
        filtered_A_B_inds[i].append(j)
        filtered_B_A_inds[j].append(i)
    
    return filtered_A_B_inds, filtered_B_A_inds, result

def knn_classifier_eval(knn_pr, y_gt, top_k=False, mask=None):
    corr_mask = []
    for x, y in zip(knn_pr, y_gt):
        if top_k:
            corr_mask.append(y in x)
        else:
            corr_mask.append(x == y)
    corr_mask = np.array(corr_mask)
    
    mask = np.ones(len(y_gt)).astype('bool') if mask is None else mask
    acc = corr_mask[mask].mean()
    return acc



def evaluator(kn_data_closed_pr, kn_data_closed_gt):
    kn_data_acc = accuracy_score(kn_data_closed_gt, kn_data_closed_pr)
    kn_data_f1_score = f1_score(kn_data_closed_gt, kn_data_closed_pr, average='macro')
    print(f'Accuracy={kn_data_acc:.4f}, F1_score={kn_data_f1_score:.4f}')
    
    return kn_data_acc, kn_data_f1_score

def umap_for_adata(adata):
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=2,
        metric="correlation",
        n_epochs=None,
        learning_rate=1.0,
        min_dist=0.3,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        repulsion_strength=1,
        negative_sample_rate=5,
        random_state=1234,
        verbose=False
    )
    embedding = reducer.fit_transform(adata.X)
    adata.obsm["X_umap"] = embedding
    return adata 