import os
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from train import BuildscSHEFT
from os.path import join


from metrics import osr_evaluator

from tool import umap_for_adata

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1234)

sc.settings.verbosity = 3
sc.logging.print_header()

exp_id = 'bm'
data_root = '/mnt/second19T/huangzt/scSHEFT/DataDir/'

# Adjust paths using relative references
adata_atac = sc.read_h5ad(join(data_root, f'{exp_id}-GAS.h5ad'))
adata_rna = sc.read_h5ad(join(data_root, f'{exp_id}-GEM.h5ad'))

meta_rna = adata_rna.obs.copy()
meta_atac = adata_atac.obs.copy()

# low-dimension representations of raw scATAC-seq data
adata_raw_atac = sc.read_h5ad(join(data_root, "bm_multiome_atac.h5ad"))
print(adata_raw_atac)
atac_raw_emb = np.load(join(data_root, f'{exp_id}_lsi.npy'))
print(atac_raw_emb.shape)



# params dict of preprocessing  List for [RNA,ATAC] type
ppd = {'binz': False,
       'hvg_num':adata_atac.shape[1], # None,
       'lognorm':[False,True],
       'scale_per_batch':[True,True],
       'type_label':  'cell_type',
       'knn': 10,
       } 

#model
model = BuildscSHEFT(
    encoder_type='linear',
    use_struct=True, stc_w=0.1, stc_cutoff=100,
    n_latent=64, bn=False, dr=0.2, 
    cont_w=0.1, cont_tau=0.1, 
    align_w=0.1, align_p=0.8, align_cutoff=100, 
    center_w=1,
    anchor_w=0.05,
)


model.preprocess(
    exp_id,
    [adata_rna, adata_atac, adata_raw_atac],  
    atac_raw_emb,   
    adata_adt_inputs=None,
    pp_dict=ppd,
    stc_emb_inputs=True
)

model.train(
    batch_size=256, training_steps=800,
    lr=0.01, weight_decay=0,
    log_step=100
)


model.eval(inplace=True)
atac_pred_type = model.annotate()
# UMAP
ad_atac = sc.AnnData(model.feat_B)
ad_atac.obs = meta_atac.copy()
ad_atac.obs['pred_type'] = atac_pred_type
ad_atac.obs['pred_conf'] = np.max(model.head_B, axis=1)
ad_atac = umap_for_adata(ad_atac)
save_dir = f"/mnt/second19T/huangzt/scSHEFT/Figs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sc.settings.figdir = save_dir
sc.pl.umap(ad_atac, color=['cell_type', 'pred_type', 'pred_conf'], save = "-rd-scSHEFT.png")
adata_atac.obsm['X_umap'] = ad_atac.obsm['X_umap']
adata_atac.write(join(data_root, f'{exp_id}-GAM2_output.h5ad'))

#Evaluation
share_mask = meta_atac.cell_type.isin(meta_rna.cell_type.unique()).to_numpy()
open_score = 1 - np.max(model.head_B, axis=1)
kn_data_pr = atac_pred_type[share_mask]
kn_data_gt = meta_atac.cell_type[share_mask].to_numpy()
kn_data_open_score = open_score[share_mask]
unk_data_open_score = open_score[np.logical_not(share_mask)]

closed_acc, os_auroc, os_aupr, oscr = osr_evaluator(kn_data_pr, kn_data_gt, kn_data_open_score, unk_data_open_score)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(meta_atac.cell_type.to_numpy(), atac_pred_type)
cm = cm/cm.sum(axis=1, keepdims=True)

df_cm = pd.DataFrame(cm, index = meta_atac.cell_type.unique(),
                  columns = meta_atac.cell_type.unique())

plt.figure(figsize = (12,12))
sns.heatmap(df_cm, )
plt.savefig(f"/mnt/second19T/huangzt/scSHEFT/Figs/-rd-scSHEFT-heatmap.png")
