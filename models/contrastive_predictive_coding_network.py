import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelCNN(nn.Module):

    def __init__(self, in_channels, num_layers):
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv1 = nn.Conv2d(in_channels, out_channels=256, kernel_size=(1, 1))
            conv2 = nn.Conv2d(256, out_channels=256, kernel_size=(1, 3), padding='same')
            conv3 = nn.Conv2d(256, out_channels=256, kernel_size=(2, 1), padding='valid')
            conv4 = nn.Conv2d(in_channels, out_channels=256, kernel_size=(1, 1))
            self.layers.append(nn.ModuleList[conv1, conv2, conv3, conv4])

    def forward(self, x):
        # x has shape (N, H, W, D)
        cres = x
        for layer in self.layers:
            c = F.relu(layer[0](cres))
            c = layer[1](c)
            c = F.pad(c, (1, 0, 0, 0, 0, 0)) # pad 1 on top of "image" (this corresponds to masking)
            c = F.relu(layer[2](c))
            c = layer[3](c)
            cres = cres + c
        cres = F.relu(cres)
        return cres


class ContrastivePredictiveCodingNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, num_patches_in_row, max_k=5):
        super().__init__()
        self.encoder = encoder
        self.num_patches_in_row = num_patches_in_row
        self.max_k = max_k

    def forward(self, x):



"""
class ContrastivePredictiveCodingNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, num_patches_in_row, max_k=5):
        super().__init__()
        self.encoder = encoder
        self.num_patches_in_row = num_patches_in_row
        self.max_k = max_k
        self.gru = torch.nn.GRU(input_size=encoder_dim, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)
        self.Ws = nn.ModuleList([nn.Linear(256, encoder_dim) for i in range(self.max_k)])

    def forward(self, x):
        # x has shape (N, L, 1, H, W), L=7*7 in default setting and H=W=64
        seq_length = x.size()[1]
        t = torch.randint(0, seq_length - self.num_patches_in_row)
        t_end_of_row = t + (self.num_patches_in_row - t % self.num_patches_in_row) - 1

        z = torch.stack([self.encoder(x[:, i, ...]) for i in range(seq_length)], dim=1)  # (N, L, encoder_dim)
        output, h_n = self.gru(z[:, t_end_of_row, :])  # (N, L, encoder_dim)
        c = output[:, -1, :]  # (N, encoder_dim)
        c_fc = [Wk(c) for Wk in self.Ws]


        c_size = c.size()
        f = torch.zeros((c_size[0], c_size[1], self.max_k, c_size[2]))
        for t in range(seq_length):
            for k in range(self.max_k):
                i = t+k*self.num_patches_in_row
                if i < seq_length:
                    # (N, encoder_dim) , (N, encoder_dim)
                    f[:, t, k, :] = torch.sum(z[:, i, :] * self.Wk[k](c[:, t, :]), dim=1)  # scalar product over batch
        return f  # (N, L, max_k, H_out) # f is without exp
"""
