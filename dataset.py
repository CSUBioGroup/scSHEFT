import torch
import random
import numpy as np
import scipy.sparse as sps
from torch.utils.data import Dataset, DataLoader

random.seed(1)

class ClsDataset(Dataset):
    def __init__(self, feats, labels, binz=True, train=False, return_id=False):
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
        rand_idx = random.randint(0, self.sample_num - 1) if self.train else i
        sample = self.X[rand_idx].A if self.issparse else self.X[rand_idx]
        sample = sample.reshape((1, self.input_size)) if self.train else sample
        in_data = (sample > 0).astype('float32') if self.binz else sample.astype('float32')
        in_label = self.y[rand_idx]
        i = rand_idx if self.train else i

        if self.return_id:
            return in_data.squeeze(), in_label, i
        else:
            return in_data.squeeze(), in_label

def get_pos_ind(ind, knn_ind):
    choice_per_nn_ind = np.random.randint(
        low=0, high=knn_ind.shape[1], size=ind.shape[0])
    pos_ind = knn_ind[ind, choice_per_nn_ind]
    return pos_ind 