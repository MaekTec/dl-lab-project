import torch
import torchvision
from PIL import Image
from typing import Any, Callable, Optional, Tuple

# Adapted from: https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10


class CIFAR10Custom(torchvision.datasets.CIFAR10):

    def __init__(
            self,
            root: str,
            train: bool = False,
            val: bool = False,
            test: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            unlabeled: bool = True,
            valset_percentage_of_trainset = 0.1,
            pretrain_task = None
    ) -> None:
        super(CIFAR10Custom, self).__init__(root, not test, transform, target_transform, download)
        self.unlabeled = unlabeled
        self.pretrain_task = pretrain_task

        assert sum([train, val, test]) == 1  # only one can be active
        assert any([train, val, test])  # at least one hast to active

        if not test:
            N = len(self.data)
            split = int(N/10)
            if unlabeled:
                self.data = self.data[split:]
                self.targets = None
            else:
                self.data = self.data[:split]
                self.targets = self.targets[:split]

            print(len(self.data))

            valset_size = int(len(self.data) * valset_percentage_of_trainset)
            trainset_size = len(self.data) - valset_size
            if train:
                self.data = self.data[valset_size:]
                if not unlabeled:
                    self.targets = self.targets[valset_size:]
            if val:
                self.data = self.data[:valset_size]
                if not unlabeled:
                    self.targets = self.targets[:valset_size]

    @staticmethod
    def mean():
        return [x / 255.0 for x in [125.3, 123.0, 113.9]]

    @staticmethod
    def std():
        return [x / 255.0 for x in [63.0, 62.1, 66.7]]

    def __getitem__(self, index: int):  # -> Tuple[Any, Optional[Any]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) or (image,) where target is index of the target class.
        """
        if self.unlabeled:
            img = self.data[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.pretrain_task == 'cmc':
                return img, index
        else:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.pretrain_task == 'cmc':
                return img, target , index

            return img, target
