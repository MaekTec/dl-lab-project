import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import *
from data.CIFAR10Custom import CIFAR10Custom
from torch.utils.data._utils.collate import default_collate


def custom_collate(batch):
    img, label = default_collate(batch)
    if isinstance(img, list):
        img = torch.cat(img, dim=0)
        label = torch.cat(label, dim=0)
    return img, label


class ImgRotation:
    """ Produce 4 rotated versions of the input image. """
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        """
        Produce 4 rotated versions of the input image.
        Args:
            img: the input PIL image to be rotated.
        Returns:
            rotated_imgs: a list containing all the rotated versions of img.
            labels: a list containing the corresponding label for each rotated image in rotated_imgs.
        """
        rotated_imgs = [img.rotate(angle) for angle in self.angles]
        labels = list(range(len(self.angles)))
        assert len(rotated_imgs) == len(labels)
        return rotated_imgs, labels


class ApplyAfterRotations:
    """ Apply a transformation to a list of images (e.g. after applying ImgRotation)"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        images, labels = x
        images = [self.transform(i) for i in images]
        return images, labels


class ToTensorAfterRotations:
    """ Transform a list of images to a pytorch tensor (e.g. after applying ImgRotation)"""
    def __call__(self, x):
        images, labels = x
        return [TF.to_tensor(i) for i in images], [torch.tensor(l) for l in labels]


class ToTensorFromPIL:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return target


def get_transforms_pretraining(args):
    """ Returns the transformations for the pretraining task. """
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ImgRotation(),
        ToTensorAfterRotations(),
        ApplyAfterRotations(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std()))
    ])
    return train_transform
