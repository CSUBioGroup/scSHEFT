import scanpy as sc
import numpy as np
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import scipy.stats as stats
from sklearn.metrics import pairwise_distances
import umap

def hvg_binz_lognorm_scale(rna, atac, hvg_num, binz, lognorm, scale_per_batch):
    n_rna, n_atac = rna.shape[0], atac.shape[0]
    n_feature1, n_feature2 = rna.shape[1], atac.shape[1]

    print("Finding highly variable genes...")
    if hvg_num :
        sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(atac, flavor='seurat_v3', n_top_genes=hvg_num)

        hvg_rna  = rna.var[rna.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_atac = atac.var[atac.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_total = list(set(hvg_rna) & set(hvg_atac))

        if len(hvg_total) < 100:
            raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))
 
        rna  = rna[:, hvg_total].copy()
        atac = atac[:, hvg_total].copy()

    ## pp
    if binz:
        rna.X = (rna.X>0).astype('float')
        atac.X = (atac.X>0).astype('float')

    if lognorm[0]: # RNA need lognorm
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
    
    if lognorm[1]: # ATAC need lognorm
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)

    if scale_per_batch[0]:  # RNA need scale
        sc.pp.scale(rna, max_value=10)

    if scale_per_batch[1]:  # ATAC need scale
        sc.pp.scale(atac, max_value=10) 

    return rna, atac


def NN(data, query=None, k=10, metric='manhattan', n_trees=10):
    """
    @param data Input data
    @param query Data to query against data
    @param k Number of nearest neighbors to compute
    Approximate nearest neighbors using locality sensitive hashing.
    """
    if query is None:
        query = data

    # Build index.
    a = AnnoyIndex(data.shape[1], metric=metric)
    for i in range(data.shape[0]):
        a.add_item(i, data[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(query.shape[0]):
        ind.append(a.get_nns_by_vector(query[i, :], k, search_k=-1))
    ind = np.array(ind)

    return ind


# def get_adj(count, k=15, pca=512, mode="connectivity"):  # pca = 50
#     """
#      metrics='euclidean'
#     """
#     if pca:
#         pcaten = TruncatedSVD(n_components=pca)
#         countp = pcaten.fit_transform(count)
#     else:
#         countp = count
#     A = kneighbors_graph(countp, k, mode=mode, metric="cosine", include_self=False) 
#     adj = A.toarray()
#     return adj 

def anchor_point_distance(f_A, f_B, k=5):
    """
    input: ndarray
    output: list
    """
    dis_matrix = pairwise_distances(f_A,f_B,metric="cosine")
    top_k_row_indices = np.argsort(dis_matrix, axis=1)[:, :k]
    top_k_column_indices = np.argsort(dis_matrix.T, axis=1)[:, :k]
    A = np.array(top_k_row_indices)
    B = np.array(top_k_column_indices)
    a, k = A.shape
    b, _ = B.shape

    reverse_mapping = {i: set() for i in range(0, a)}
    for i in range(b):
        for j in range(k):
            reverse_mapping[B[i, j]].add(i)

    result = set()
    for w in range(a):
        for v in A[w]:
            if w in reverse_mapping and v in reverse_mapping[w]:
                result.add((w, v))

    print(f"Finding {len(result)} anchors")
    filtered_A_B_inds = [[] for _ in range(a)]
    filtered_B_A_inds = [[] for _ in range(b)]
        
    for i, j in result:
        filtered_A_B_inds[i].append(j)
        filtered_B_A_inds[j].append(i)

    return filtered_A_B_inds, filtered_B_A_inds, result # return anchor query: A=>B / B=>A  list type


def knn_classifier_eval(knn_pr, y_gt, top_k=False, mask=None):    
    corr_mask = []
    for x,y in zip(knn_pr, y_gt):
        if top_k:
            corr_mask.append(y in x)
        else:
            corr_mask.append(x==y)
    corr_mask = np.array(corr_mask)
    
    mask = np.ones(len(y_gt)).astype('bool') if mask is None else mask
    acc = corr_mask[mask].mean()
    # print(acc)
    return acc

def annoy_knn(data, query=None, k=10, metric='manhattan', n_trees=10):
    if query is None:
        query = data

    data = normalize(data, axis=1)
    query = normalize(query, axis=1)

    # Build index.
    a = AnnoyIndex(data.shape[1], metric=metric)
    for i in range(data.shape[0]):
        a.add_item(i, data[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(query.shape[0]):
        ind.append(a.get_nns_by_vector(query[i, :], k, search_k=-1))
    ind = np.array(ind)

    return ind

def equal_inter_sampling(X, Y, n_sample):
    n_total = X.shape[0]
    if n_sample < n_total:
        smp_ind = np.linspace(0, n_total-1, num=n_sample).astype('int')
        X = X[smp_ind]
        Y = Y[smp_ind]
    return X, Y

def kNN_approx(X1, X2, Y1, n_sample=20000, knn=10, knn_method='annoy'):
    X1 = X1.copy()
    X2 = X2.copy()
    Y1 = Y1.copy()

    if n_sample is not None:
        X1, Y1 = equal_inter_sampling(X1, Y1, n_sample=n_sample)

    knn_ind = annoy_knn(X1, X2, knn)

    knn_pred_pop = Y1[knn_ind.ravel()].reshape(knn_ind.shape)

    knn_pred = stats.mode(knn_pred_pop, axis=1)[0].ravel()
    return knn_pred

def umap_for_adata(ada):
    reducer = umap.UMAP(n_neighbors=30,
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
                    a=None,
                    b=None,
                    random_state=1234,
                    metric_kwds=None,
                    angular_rp_forest=False,
                    verbose=False)
    embedding = reducer.fit_transform(ada.X)
    ada.obsm["X_umap"] = embedding
    return ada
