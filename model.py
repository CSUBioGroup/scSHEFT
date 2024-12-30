import torch.nn as nn
dropout_rate = .2

class Net_encoder(nn.Module):
    def __init__(self, input_size, n_latent):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        # self.k = 64
        # self.f = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, n_latent)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding

class Nonlinear_encoder(nn.Module):
    def __init__(self, input_size, n_latent, bn=False, dr=True):
        super(Nonlinear_encoder, self).__init__()
        self.input_size = input_size
        # self.k = 64
        # self.f = 64

        if bn:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, n_latent),
                nn.BatchNorm1d(n_latent, affine=True),
                nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate),
                nn.Linear(n_latent, n_latent),
                nn.BatchNorm1d(n_latent, affine=True),
                nn.ReLU(inplace=True),
                # nn.Dropout(p=dropout_rate),
            )
        elif dr:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_latent, n_latent),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, n_latent),
                nn.ReLU(inplace=True),
                nn.Linear(n_latent, n_latent),
                nn.ReLU(inplace=True),
            )


    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        # embedding = F.normalize(embedding)

        return embedding


class Net_cell(nn.Module):
    def __init__(self, n_latent, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(n_latent, num_of_class)
        )

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)

        return cell_prediction

