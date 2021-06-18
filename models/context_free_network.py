import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextFreeNetwork(nn.Module):

    def __init__(self, encoder, input_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc7 = nn.Linear(input_dim, 4096)
        self.fc8 = nn.Linear(4096, 100)
        self.fc9 = nn.Linear(100, num_classes)

    def forward(self, x):
        # x has shape (N, 9, 1, H, W)
        x = [F.relu(self.encoder(x[:, i, ...])) for i in range(x.shape[1])]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
