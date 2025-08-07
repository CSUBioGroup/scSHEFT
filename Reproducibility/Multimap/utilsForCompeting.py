import os
import numpy as np
import scanpy as sc
import pandas as pd
import umap
from sklearn.metrics import (
    accuracy_score, f1_score
)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
import scipy.stats as stats


def evaluator(kn_data_closed_pr, kn_data_closed_gt):
    """开放集识别评估器"""
    n_kn = len(kn_data_closed_pr)
    
    # 闭集评估
    kn_data_acc = accuracy_score(kn_data_closed_gt, kn_data_closed_pr)
    kn_data_f1_score = f1_score(kn_data_closed_gt, kn_data_closed_pr, average='macro')
    
    print(f'Accuracy={kn_data_acc:.4f}, F1_score={kn_data_f1_score:.4f}')
    
    return kn_data_acc, kn_data_f1_score

def umap_for_adata(adata):
    """为AnnData计算UMAP嵌入"""
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