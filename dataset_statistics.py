from collections import Counter

from data.CIFAR10Custom import CIFAR10Custom

data_root = "dataset"


def statistic(name, dataset):
    print(f'Statistic for {name}')
    print(f'Size = {len(dataset)}')
    print(f'Class distribution = {dict(Counter(dataset.targets))}')


train_data_pretraining = CIFAR10Custom(data_root, train=True, download=True, unlabeled=True)
val_data_pretraining = CIFAR10Custom(data_root, val=True, download=True, unlabeled=True)

train_data_fine_tuning = CIFAR10Custom(data_root, train=True, download=True, unlabeled=False)
val_data_fine_tuning = CIFAR10Custom(data_root, val=True, download=True, unlabeled=False)

statistic("Trainset pretraining", train_data_pretraining)
statistic("Valset pretraining", val_data_pretraining)
statistic("Trainset fine tuning", train_data_fine_tuning)
statistic("Valset fine tuning", val_data_fine_tuning)