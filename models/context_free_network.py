import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextFreeNetwork(nn.Module):

    def __init__(self, encoder, input_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc7 = nn.Linear(input_dim, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        # we only used 2 instead of 3 linear layers, because the original last layer was very small and lead to bad
        # performance in the downstream task when only fine tuning the last layer.

    def forward(self, x):
        # x has shape (N, 9, 1, H, W)
        x = [F.relu(self.encoder(x[:, i, ...])) for i in range(x.shape[1])]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x
