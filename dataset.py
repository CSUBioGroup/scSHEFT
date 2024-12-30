import torch
import random
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sps
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(
            self, 
            feats, labels, binz=True, train=False, return_id=False
        ):
        self.X = feats
        self.y = labels
        self.train = train
        self.binz = binz
        self.return_id = return_id
        self.sample_num = self.X.shape[0]
        self.input_size = self.X.shape[1]
        self.issparse = sps.issparse(feats)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.train:        # the same as scjoint, but never used
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = self.X[rand_idx].A if self.issparse else self.X[rand_idx]
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype('float32') if self.binz else sample.astype('float32')  # binarize data
            in_label = self.y[rand_idx]
            i = rand_idx
        else:
            sample = self.X[i].A if self.issparse else self.X[i]
            in_data = (sample>0).astype('float32') if self.binz else sample.astype('float32')
            in_label = self.y[i]

        # x = self.data[i].A         # if binarize_data, use this
        if self.return_id:
            return in_data.squeeze(), in_label, i
        else:
            return in_data.squeeze(), in_label
