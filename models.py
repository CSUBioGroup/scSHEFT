import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearEncoder(nn.Module):
    def __init__(self, input_size, n_latent):
        super(LinearEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, n_latent)
    
    def forward(self, data):
        data = data.float().view(-1, self.encoder.in_features)
        return self.encoder(data)

class NonlinearEncoder(nn.Module):
    def __init__(self, input_size, n_latent, bn=False, dr=True, dropout_rate=0.2):
        super(NonlinearEncoder, self).__init__()
        
        if bn:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, n_latent),
                nn.BatchNorm1d(n_latent, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(n_latent, n_latent),
                nn.BatchNorm1d(n_latent, affine=True),
                nn.ReLU(inplace=True),
            )
        elif dr:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_latent, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, n_latent),
                nn.ReLU(inplace=True),
                nn.Linear(n_latent, n_latent),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, data):
        data = data.float().view(-1, self.encoder[0].in_features)
        return self.encoder(data)

class Classifier(nn.Module):
    def __init__(self, n_latent, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(n_latent, num_classes)
    
    def forward(self, embedding):
        return self.classifier(embedding) 