import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

"""
This implementation is very similar to the pseudo code in the appendix of the paper.
"""


class PixelCNN(nn.Module):

    def __init__(self, in_channels, num_layers, autoregressive_dim):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv1 = nn.Conv2d(in_channels, out_channels=autoregressive_dim, kernel_size=(1, 1))
            # same convolution
            conv2 = nn.Conv2d(autoregressive_dim, out_channels=autoregressive_dim, kernel_size=(1, 3), padding=(0, 1))
            # valid convolution
            conv3 = nn.Conv2d(autoregressive_dim, out_channels=autoregressive_dim, kernel_size=(2, 1))
            conv4 = nn.Conv2d(autoregressive_dim, out_channels=in_channels, kernel_size=(1, 1))
            self.layers.append(nn.ModuleList([conv1, conv2, conv3, conv4]))

    def forward(self, x):
        # x has shape (N, D, H, W)
        cres = x
        for layer in self.layers:
            c = F.relu(layer[0](cres))
            c = layer[1](c)
            c = F.pad(c, (0, 0, 1, 0, 0, 0))  # pad 1 on top of the "image" (this corresponds to mask conv)
            c = F.relu(layer[2](c))
            c = layer[3](c)
            cres = cres + c
        cres = F.relu(cres)
        return cres


class ContrastivePredictiveCodingNetwork(nn.Module):

    """
    steps_to_ignore=1, because overlap between patches is 50% as the default
    """

    def __init__(self, encoder, encoder_dim, num_patches_per_dim, autoregressive_dim=256, target_dim=64, emb_scale=0.1,
                 steps_to_ignore=1, steps_to_predict=3):
        super().__init__()
        self.encoder = encoder
        self.pixel_cnn = PixelCNN(encoder_dim, num_patches_per_dim-1, autoregressive_dim)
        self.num_patches_per_dim = num_patches_per_dim
        self.target_dim = target_dim
        self.emb_scale = emb_scale
        self.steps_to_ignore = steps_to_ignore
        self.steps_to_predict = steps_to_predict

        self.conv_targets = nn.Conv2d(encoder_dim, out_channels=target_dim, kernel_size=(1, 1))
        self.conv_preds = nn.ModuleList()
        for _ in range(steps_to_ignore, steps_to_predict):
            self.conv_preds.append(nn.Conv2d(encoder_dim, out_channels=target_dim, kernel_size=(1, 1)))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        # x has shape (N, L, 1, HI, WI), L=7*7 (number of patches) in default setting and HI=WI=64 (original CPC)
        seq_length = x.size()[1]
        # encode each patch
        latents = torch.stack([self.encoder(x[:, i, ...]) for i in range(seq_length)], dim=1)  # (N, L, DE)
        # reshape to have a grid of encoded patches, HL=WL=7, DE is encoder output dim
        latents = torch.reshape(latents, (x.size()[0], self.num_patches_per_dim, self.num_patches_per_dim,
                                          latents.size()[2]))  # (N, HL, WL, DE)
        latents = latents.permute(0, 3, 1, 2)  # (N, DE, HL, WL)

        total_loss = 0.0
        total_accuracy = 0
        total = 0
        context = self.pixel_cnn(latents)  # (N, DE, HL, WL)
        # 1x1 conv corresponds to a linear layer for the targets
        targets = self.conv_targets(latents)  # (N, DT , HL, WL), DT is target_dim
        batch_dim, _, col_dim, rows = targets.size()
        targets = torch.reshape(targets.permute(0, 2, 3, 1), (-1, self.target_dim))  # (N*HL*WL, DT)
        for i in range(self.steps_to_ignore, self.steps_to_predict):
            col_dim_i = col_dim - i - 1  # column dim of predictions
            total_elements = batch_dim * col_dim_i * rows  # number of predictions
            # 1x1 conv corresponds to a linear layer for the predictions (W_k in paper)
            preds_i = self.conv_preds[i-self.steps_to_ignore](context)  # (N, DT, HL, WL)
            # crop predictions to predict only existing future latents
            preds_i = preds_i[:, :, :-(i+1), :] * self.emb_scale   # (N, DP, HLC, WLC)
            preds_i = torch.reshape(preds_i.permute(0, 2, 3, 1), (-1, self.target_dim))  # (N*HLC*WLC, DP)
            # calculate dot product between predictions and targets
            logits = torch.matmul(preds_i, targets.T)  # (N*HLC*WLC, N*HL*WL)

            # batch index for each prediction
            b = torch.div(torch.arange(total_elements), (col_dim_i * rows), rounding_mode='trunc')
            # column index for each prediction
            col = torch.arange(total_elements) % (col_dim_i * rows)
            # batch start index + index of row to predict + column index, i+1 because we want to predict the future
            labels = b * col_dim * rows + (i + 1) * rows + col
            labels = labels.cuda()

            # unsqueeze to have a batch of 1
            logits_batch = torch.unsqueeze(logits.T, dim=0)  # (1, (N*HL*WL, N*HLC*WLC))
            labels_batch = torch.unsqueeze(labels, dim=0)  # (1, N*HLC*WLC)
            total_loss += self.cross_entropy(logits_batch, labels_batch)
            total_accuracy += accuracy(logits, labels)[0].item()
            total += 1
        loss = total_loss / total
        acc = total_accuracy / total
        return loss, acc


class ContrastivePredictiveCodingNetworkLinearClassification(nn.Module):

    def __init__(self, encoder, encoder_dim, num_patches_per_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_patches_per_dim = num_patches_per_dim
        self.fc = nn.Linear(encoder_dim*num_patches_per_dim*num_patches_per_dim, num_classes)

    def forward(self, x):
        # x has shape (N, L, 1, HI, WI), L=7*7 in default setting and HI=WI=64 (original CPC)
        seq_length = x.size()[1]
        # encode each patch
        latents = torch.stack([self.encoder(x[:, i, ...]) for i in range(seq_length)], dim=1)  # (N, L, DE)
        # reshape to have a grid of encoded patches, HL=WL=7, DE is encoder output dim
        latents = torch.reshape(latents, (x.size()[0], self.num_patches_per_dim, self.num_patches_per_dim,
                                          latents.size()[2]))  # (N, HL, WL, DE)
        latents = latents.permute(0, 3, 1, 2)  # (N, DE, HL, WL)
        x = torch.flatten(latents, 1)
        x = self.fc(x)
        return x
