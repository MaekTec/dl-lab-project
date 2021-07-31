import random

import elasticdeform
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import *
from data.CIFAR10Custom import CIFAR10Custom
from torch.utils.data._utils.collate import default_collate
import itertools
import math
from PIL import ImageFilter


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
        if isinstance(x, tuple):
            images, labels = x
        else:
            images = x
        # x has shape (C x H x W)
        patch_height = images.size()[1] // self.num_tiles_per_dim
        patch_width = images.size()[2] // self.num_tiles_per_dim
        patches = []
        for i in range(self.num_tiles_per_dim):
            for j in range(self.num_tiles_per_dim):
                patches.append(images[:, j * patch_height:j * patch_height + patch_height,
                               i * patch_width:i * patch_width + patch_width])
        # labels = list(range(self.num_tiles_per_dim**2))
        # assert len(patches) == len(labels)
        if isinstance(x, tuple):
            return patches, labels
        else:
            return patches


class ShuffleTiles:

    def __init__(self, num_tiles_per_dim, number_of_permutations):
        self.num_tiles_per_dim = num_tiles_per_dim
        self.num_tiles = self.num_tiles_per_dim ** 2
        self.N = number_of_permutations
        self.permutation_set = self.generate_permutation_set()

    @staticmethod
    def hamming(p_0, p_1):
        p_0 = np.expand_dims(p_0, axis=1)
        p_1 = np.expand_dims(p_1, axis=0)
        D = np.sum(p_0 != p_1, axis=2)
        """
        # Same, but slower:
        D = np.zeros((len(p_0), len(p_1)))
        for i in range(len(p_0)):
            for j in range(len(p_1)):
                D[i, j] = np.sum(p_0[i] != p_1[j])
        """
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
        # labels_reordered = []
        for i in permutation:
            images_reordered.append(images[i])
            # labels_reordered.append(labels[i])
        return images_reordered, permutation_index  # labels_reordered


class ColorChannelJitter:

    def __init__(self, max_jitter=2):
        self.max_jitter = max_jitter

    def __call__(self, x):
        # x has shape (C x H x W)
        _, h, w = x.size()
        for c in range(3):
            jitter = tuple(np.random.randint(-self.max_jitter + 1, self.max_jitter + 1, 2))
            x[c, ...] = torch.roll(x[c, ...], jitter, (0, 1))

        x = x[:, self.max_jitter:-self.max_jitter, self.max_jitter:-self.max_jitter]  # crop
        return x


class DivideInGrid:

    def __init__(self, patch_size, overlap):  # both in pixel
        self.patch_size = patch_size
        self.overlap = overlap

    def __call__(self, x):
        if isinstance(x, tuple):
            images, labels = x
        else:
            images = x
        # x has shape (C x H x W)
        assert images.size()[1] == images.size()[2]
        image_size = images.size()[1]
        num_patches_per_dim = int(image_size / (self.patch_size - self.overlap)) - 1
        patches = []
        for i in range(num_patches_per_dim):
            for j in range(num_patches_per_dim):
                patches.append(images[:, i * (self.patch_size - self.overlap):i * (self.patch_size - self.overlap) + self.patch_size,
                               j * (self.patch_size - self.overlap):j * (self.patch_size - self.overlap) + self.patch_size])
        # patches shape is (N, H, W)
        if isinstance(x, tuple):
            return patches, labels
        else:
            return patches


class ElasticDeformation:

    def __init__(self):
        pass

    def __call__(self, x):
        return elasticdeform.deform_random_grid(x, sigma=25, points=3)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """
    Take two random crops of one image as the query and key.
    From original impl:
    https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/moco/loader.py#L6
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class ApplyOnList:
    """ Apply a transformation to a list of images (e.g. after applying ImgRotation)"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        if isinstance(x, tuple):
            images, labels = x
            images = [self.transform(i) for i in images]
            return images, labels
        else:
            x = [self.transform(i) for i in x]
            return x


class ToTensorAfterRotations:
    """ Transform a list of images to a pytorch tensor (e.g. after applying ImgRotation)"""

    def __call__(self, x):
        images, labels = x
        return [TF.to_tensor(i) for i in images], [torch.tensor(l) for l in labels]


class CollateList:
    def __call__(self, x):
        if isinstance(x, tuple):
            images, labels = x
            x = torch.stack(images, dim=0)
            return x, labels
        else:
            x = torch.stack(x, dim=0)
            return x


def get_transforms_pretraining_rotation(args):
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ImgRotation(),
        ToTensorAfterRotations(),
        ApplyOnList(Resize(args.image_size)),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std()))
    ])
    return train_transform


def get_transforms_pretraining_jigsaw_puzzle(args):
    train_transform = Compose([
        ToTensor(),
        Resize(args.image_size*4),
        #ApplyOnList(ColorChannelJitter(2)),  # Has no purpose for the ViT, because image patch is flattened.
        RandomCrop(int(225/256*args.image_size*4)),
        DivideInTiles(args.num_tiles_per_dim),
        ShuffleTiles(args.num_tiles_per_dim, args.number_of_permutations),
        ApplyOnList(RandomCrop(args.image_size)),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())),
        ApplyOnList(RandomGrayscale(p=0.3)),
        CollateList()
    ])
    return train_transform


def get_transforms_pretraining_contrastive_predictive_coding(args):
    train_transform = Compose([
        ToTensor(),
        Resize(int(args.image_size + ((args.num_patches_per_dim-1) * int(args.image_size/2)))),  # 160 for 4x4 grid
        DivideInGrid(args.image_size, int(args.image_size/2)),  # 4x4 grid with 50% overlap
        ApplyOnList(ToPILImage()),
        ApplyOnList(AutoAugment(AutoAugmentPolicy.CIFAR10)),
        ApplyOnList(AutoAugment(AutoAugmentPolicy.CIFAR10)),
        ApplyOnList(ToTensor()),
        # ColorJitter instead of original transformations which are not public available
        ApplyOnList(RandomApply([ColorJitter(brightness=.2, contrast=0.2, saturation=0.2, hue=.2)], p=0.8)),
        ApplyOnList(RandomApply([RandomAffine(0, shear=5)], p=0.2)),
        ApplyOnList(RandomApply([Grayscale(num_output_channels=3)], p=0.25)),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())),
        CollateList()
    ])
    return train_transform


def get_transforms_pretraining_moco(args):
    # Copied from the original impl: https://github.com/facebookresearch/moco/blob/master/main_moco.py
    transform_each_crop = Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std()),
    ])
    train_transform = TwoCropsTransform(transform_each_crop)
    return train_transform


def get_transforms_downstream_jigsaw_puzzle_training(args):
    train_transform = Compose([
        ToTensor(),
        Resize(args.image_size*4),
        #ApplyOnList(ColorChannelJitter(2)),  # Has no purpose for the ViT, because image patch is flattened.
        RandomCrop(int(225/256*args.image_size*4)),
        DivideInTiles(args.num_tiles_per_dim),
        ApplyOnList(RandomCrop(args.image_size)),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())),
        ApplyOnList(RandomGrayscale(p=0.3)),
        CollateList()
    ])
    return train_transform


def get_transforms_downstream_jigsaw_puzzle_validation(args):
    val_transform = Compose([
        ToTensor(),
        Resize(args.image_size * 3),
        DivideInTiles(args.num_tiles_per_dim),
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())),
        CollateList()
    ])
    return val_transform


def get_transforms_downstream_contrastive_predictive_coding_validation(args):
    val_transform = Compose([
        ToTensor(),
        Resize(int(args.image_size + ((args.num_patches_per_dim - 1) * int(args.image_size / 2)))),  # 160 for 4x4 grid
        DivideInGrid(args.image_size, int(args.image_size / 2)),  # 4x4 grid
        ApplyOnList(Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())),
        CollateList()
    ])
    return val_transform


def get_transforms_downstream_training(args):
    train_transform = Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        # random crop and aspect ratio
        transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.9, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(5, (0.1, 2.0)),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        Resize(args.image_size),
        Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())
    ])
    return train_transform


def get_transforms_downstream_validation(args):
    val_transform = Compose([
        ToTensor(),
        Resize(args.image_size),
        Normalize(CIFAR10Custom.mean(), CIFAR10Custom.std())
    ])
    return val_transform

