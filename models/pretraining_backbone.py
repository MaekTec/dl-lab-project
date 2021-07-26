import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from vit_pytorch import ViT
from torchvision.models.resnet import resnet18


class ViTBackbone(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10):
        super().__init__()

        self.net = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=1024,
            depth=5,
            heads=6,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        return self.net(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = resnet18(pretrained=False)
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, num_classes, bias=True)
        nn.init.xavier_uniform_(self.net.fc.weight)

    def forward(self, x):
        return self.net(x)


class CMC_ViT_Backbone(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10):
        super().__init__()

        self.net_l = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=1024,
            depth=5,
            heads=6,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
            channels=1
        )
        self.net_ab = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=1024,
            depth=5,
            heads=6,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1,
            channels=2
        )

    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)
        #print('x.shape=',x.shape,'l.shape=', l.shape,'ab.shape=', ab.shape)

        feat_l = self.net_l(l)
        feat_ab = self.net_ab(ab)

        return feat_l, feat_ab

