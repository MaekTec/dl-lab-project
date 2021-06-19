import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from vit_pytorch import ViT


class ViTBackbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.net = ViT(
            image_size=64,
            patch_size=8,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        return self.net(x)

