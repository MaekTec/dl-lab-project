import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelCNN(nn.Module):

    def __init__(self, in_channels, num_layers, autoregressive_dim):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv1 = nn.Conv2d(in_channels, out_channels=autoregressive_dim, kernel_size=(1, 1))
            conv2 = nn.Conv2d(autoregressive_dim, out_channels=autoregressive_dim, kernel_size=(1, 3), padding=(0, 1))  # same
            conv3 = nn.Conv2d(autoregressive_dim, out_channels=autoregressive_dim, kernel_size=(2, 1))  # valid
            conv4 = nn.Conv2d(autoregressive_dim, out_channels=in_channels, kernel_size=(1, 1))
            self.layers.append(nn.ModuleList([conv1, conv2, conv3, conv4]))

    def forward(self, x):
        # x has shape (N, D, H, W)
        cres = x
        for layer in self.layers:
            c = F.relu(layer[0](cres))
            c = layer[1](c)
            c = F.pad(c, (0, 0, 1, 0, 0, 0))  # pad 1 on top of "image" (this corresponds to mask conv)
            c = F.relu(layer[2](c))
            c = layer[3](c)
            cres = cres + c
        cres = F.relu(cres)
        return cres


class ContrastivePredictiveCodingNetwork(nn.Module):

    def __init__(self, encoder, encoder_dim, num_patches_per_dim, autoregressive_dim=256, target_dim=64, emb_scale=0.1, steps_to_ignore=2, steps_to_predict=3):
        super().__init__()
        self.encoder = encoder
        self.pixel_cnn = PixelCNN(encoder_dim, num_patches_per_dim-1, autoregressive_dim)  # TODO: correct?
        self.num_patches_per_dim = num_patches_per_dim
        self.target_dim = target_dim
        self.emb_scale = emb_scale
        self.steps_to_ignore = steps_to_ignore
        self.steps_to_predict = steps_to_predict

        self.conv_targets = nn.Conv2d(encoder_dim, out_channels=target_dim, kernel_size=(1, 1))
        self.conv_preds = nn.Conv2d(encoder_dim, out_channels=target_dim, kernel_size=(1, 1))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        # x has shape (N, L, 1, HI, WI), L=7*7 in default setting and HI=WI=64 (original CPC)
        seq_length = x.size()[1]
        latents = torch.stack([self.encoder(x[:, i, ...]) for i in range(seq_length)], dim=1)  # (N, L, DE)
        latents = torch.reshape(latents, (x.size()[0], self.num_patches_per_dim, self.num_patches_per_dim, latents.size()[2]))  # (N, HL, WL, DE)
        latents = latents.permute(0, 3, 1, 2)  # (N, DE, HL, WL)

        loss = 0.0
        context = self.pixel_cnn(latents)  # (N, DE, HL, WL)
        targets = self.conv_targets(latents)  # (N, DT , HL, WL)
        batch_dim, _, col_dim, rows = targets.size()
        targets = torch.reshape(targets.permute(0, 2, 3, 1), (-1, self.target_dim))  # (N*HL*WL, DT)
        for i in range(self.steps_to_ignore, self.steps_to_predict):
            col_dim_i = col_dim - i - 1
            total_elements = batch_dim * col_dim_i * rows
            preds_i = self.conv_preds(context)  # (N, DT, HL, WL)
            preds_i = preds_i[:, :, :-(i+1), :] * self.emb_scale   # (N, DP, HLC, WLC)
            preds_i = torch.reshape(preds_i.permute(0, 2, 3, 1), (-1, self.target_dim))  # (N*HLC*WLC, DP)
            logits = torch.matmul(preds_i, targets.T)  # (N*HLC*WLC, N*HL*WL)

            b = torch.arange(total_elements) // (col_dim_i * rows)
            col = torch.arange(total_elements) % (col_dim_i * rows)
            labels = b * col_dim * rows + (i + 1) * rows + col
            loss += self.cross_entropy(torch.unsqueeze(logits.T, dim=0), torch.unsqueeze(labels, dim=0).cuda())
        return loss


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
