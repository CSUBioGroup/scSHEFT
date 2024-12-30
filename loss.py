import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

epsilon = 1e-5

class InfoNCE(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.temperature = temperature
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, 
                                                           dtype=bool)).float())
            
    def forward(self, feat1, feat2):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """

        features = torch.cat([feat1, feat2], dim=0)   
        representations = F.normalize(features, dim=1, p=2)   

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  
        sim_ji = torch.diag(similarity_matrix, -self.batch_size) 
        positives = torch.cat([sim_ij, sim_ji], dim=0)       
        
        nominator = torch.exp(positives / self.temperature) 
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))   
        loss = torch.sum(loss_partial) / (2 * self.batch_size)  
        return loss

class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss

