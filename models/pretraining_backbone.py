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
            dim=512,
            depth=4,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        return self.net(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained, num_classes):
        super().__init__()
        self.features = IntermediateLayerGetter(resnet18(pretrained=pretrained), {"avgpool": "out"}).cuda()
        self.fc = nn.Linear(512, num_classes, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.features(x)["out"]
        x = torch.flatten(x, 1)
        return self.fc(x)
