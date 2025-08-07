import os
import numpy as np
import scanpy as sc
import pandas as pd
import time

from scSHEFT import scSHEFT
from utils import evaluator

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sc.settings.verbosity = 3
sc.logging.print_header()

def main():
    # Please customize these paths according to your data location
    exp_id = "Please enter experiment ID"
    
    # Load data
    print("Loading data...")
    adata_raw_target = sc.read_h5ad("Please enter raw ATAC-seq data file path")
    print(f"Raw target data shape: {adata_raw_target.shape}")
    
    adata_target = sc.read_h5ad("Please enter processed ATAC-seq data file path")
    adata_source = sc.read_h5ad("Please enter RNA-seq data file path")
    
    # Merge metadata
    meta_source = adata_source.obs.copy()
    meta_target = adata_target.obs.copy()
    meta = pd.concat([meta_source, meta_target], axis=0)
    
    # Load embeddings
    target_raw_emb = np.load("Please enter LSI embedding file path")
    print(f"Target raw embedding shape: {target_raw_emb.shape}")
    
    struct_target_emb = np.load("Please enter structural PCA embedding file path")
    print(f"Structural target embedding shape: {struct_target_emb.shape}")
    
    # Preprocessing parameters
    ppd = {
        'binz': False,
        'hvg_num': None,
        'lognorm': [False, True],
        'scale_per_batch': True,
        'type_label': 'cell_type',
        'knn': 10,
    }
    
    # Initialize model
    print("Initializing scSHEFT model...")
    model = scSHEFT(
        encoder_type='linear',
        use_struct=True, 
        stc_w=0.5, 
        stc_cutoff=100,
        n_latent=128, 
        bn=False, 
        dr=0.2,
        cont_w=0.1, 
        cont_tau=0.1,
        align_w=0.2, 
        align_p=0.8, 
        align_cutoff=100,
        center_w=5,
        anchor_w=0.5,
    )
    
    # Preprocess data
    print("Preprocessing data...")
    model.preprocess(
        exp_id,
        [adata_source, adata_target, adata_raw_target],
        target_raw_emb,
        ppd,
        adata_adt_inputs=None,
        stc_emb_inputs=True
    )
    
    # Train model
    print("Training model...")
    model.train(
        batch_size=256, 
        training_steps=1000,
        lr=0.01, 
        weight_decay=0,
        log_step=50, 
        eval_target=True, 
        eval_top_k=1
    )

    # Evaluate model
    print("Evaluating model...")
    model.eval(inplace=True)
    target_pred_type = model.annotate()
    
    # Evaluate performance
    share_mask = meta_target.cell_type.isin(meta_source.cell_type.unique()).to_numpy()
    kn_data_pr = target_pred_type[share_mask]
    kn_data_gt = meta_target.cell_type[share_mask].to_numpy()   
    closed_acc, f1_score = evaluator(kn_data_pr, kn_data_gt)
    

if __name__ == "__main__":
    main() 