import torch
import os

class Config(object):
    def __init__(self):
        DB = 'bm'
        self.use_cuda = True
        self.threads = 1
        self.exp_id = DB

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        
        if DB == 'bm':
            self.number_of_class = 11  # Number of cell types in CITE-seq data
            self.input_size = 13916  # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.rna_paths = [f"your_rna_gem.npz"]
            self.rna_labels = [f"your_rna_cell_types.txt"]

            self.atac_paths = [f"your_atac_gam.npz"]  # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = [f"your_atac_cell_types.txt"]  # ASAP-seq data cell type labels (coverted to numeric)
            self.rna_protein_paths = []  # Protein expression from CITE-seq data
            self.atac_protein_paths = []  # Protein expression from ASAP-seq data

            # Training config
            self.batch_size = 512
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''
