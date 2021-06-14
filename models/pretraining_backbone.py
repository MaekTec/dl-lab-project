import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from vit_pytorch import ViT


class ViTBackbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.net = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        return self.net(x)

