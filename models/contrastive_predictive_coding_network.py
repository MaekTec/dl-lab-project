import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastivePredictiveCodingNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.max_k = 5
        self.gru = torch.nn.GRU(input_size=encoder_dim, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(self.max_k)])

    def forward(self, x, k):
        # x has shape (N, L, 1, H, W)
        c = [self.encoder(x[:, i, ...]) for i in range(x.shape[1])]
        c_fc = self.Wk[k](c)
        return c_fc
