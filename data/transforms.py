import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import *
from data.CIFAR10Custom import CIFAR10Custom
from torch.utils.data._utils.collate import default_collate
import itertools


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


class DivideInTiles:

    def __init__(self, num_tiles_per_dim):
        self.num_tiles_per_dim = num_tiles_per_dim

    def __call__(self, x):
        # x has shape (C x H x W)
        patch_height = x.size()[1] // self.num_tiles_per_dim
        patch_width = x.size()[2] // self.num_tiles_per_dim
        patches = []
        for i in range(self.num_tiles_per_dim):
            for j in range(self.num_tiles_per_dim):
                patches.append(x[:, j*patch_height:j*patch_height+patch_height, i*patch_width:i*patch_width+patch_width])
        #labels = list(range(self.num_tiles_per_dim**2))
        #assert len(patches) == len(labels)
        return patches  #, labels


class ShuffleTiles:

    def __init__(self, num_tiles_per_dim, number_of_permutations):
        self.num_tiles_per_dim = num_tiles_per_dim
        self.num_tiles = self.num_tiles_per_dim**2
        self.N = number_of_permutations
        self.permutation_set = self.generate_permutation_set()

    @staticmethod
    def hamming(p_0, p_1):
        D = np.zeros((len(p_0), len(p_1)))
        for i in range(len(p_0)):
            for j in range(len(p_1)):
                D[i, j] = np.sum(p_0[i] != p_1[j])
        return D

    def generate_permutation_set(self):
        all_permutations = np.array(list(itertools.permutations(list(range(self.num_tiles)))))  # (9!, 9)
        selected_permutations = []
        j = np.random.randint(0, math.factorial(self.num_tiles))
        i = 1
        while i < self.N:
            selected_permutations.append(all_permutations[j])
            all_permutations = np.delete(all_permutations, j, axis=0)
            D = self.hamming(np.array(selected_permutations), all_permutations)
            D_bar = np.sum(D, axis=0)
            j = np.argmax(D_bar)
            i += 1
        return np.array(selected_permutations)  # (N, 9)

    def __call__(self, x):
        images = x
        permutation_index = np.random.randint(0, len(self.permutation_set))
        permutation = self.permutation_set[permutation_index]
        images_reordered = []
        #labels_reordered = []
        for i in permutation:
            images_reordered.append(images[i])
            #labels_reordered.append(labels[i])
        return images_reordered, permutation_index #labels_reordered


class ColorChannelJitter:

    def __init__(self, max_jitter=2):
        self.max_jitter = max_jitter

    def __call__(self, x):
        # x has shape (C x H x W)
        _, h, w = x.size()
        for c in range(3):
            jitter = tuple(np.random.randint(-self.max_jitter + 1, self.max_jitter + 1, 2))
            x[c, ...] = torch.roll(x[c, ...], jitter, (0, 1))

        x = x[:, self.max_jitter:-self.max_jitter, self.max_jitter:-self.max_jitter]
        return x


class ApplyOnList:
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


class CollateList:
    def __call__(self, x):
        images, labels = x
        x = torch.stack(images, dim=0)
        return x, labels


def get_transforms_pretraining_rotation(args):
    """ Returns the transformations for the pretraining task. """
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ImgRotation(),
        ToTensorAfterRotations(),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std()))
    ])
    return train_transform


def get_transforms_pretraining_jigsaw_puzzle():
    """ Returns the transformations for the pretraining task. """
    train_transform = Compose([
        ToTensor(),
        Resize(74),
        DivideInTiles(2),
        ShuffleTiles(2, 24),
        ApplyOnList(RandomCrop(34)),
        ApplyOnList(ColorChannelJitter(1)),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())),
        ApplyOnList(RandomGrayscale(p=0.3)),
        CollateList()
    ])
    return train_transform
