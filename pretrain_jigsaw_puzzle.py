import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from pprint import pprint
from data.transforms import get_transforms_pretraining_jigsaw_puzzle
from utils import check_dir, set_random_seed, accuracy, get_logger, accuracy, save_in_log
from models.pretraining_backbone import ViTBackbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torchsummary
from models.context_free_network import ContextFreeNetwork

"""
https://arxiv.org/pdf/1603.09246.pdf
fixed random permutation set
to avoid learning of shortcuts:
- more than one permutation per image
- shuffle tiles as much as possible with Hamming distance
- random gap between tiles
- resize to 256 and random crop 225x225, split in 9 tiles each 75x75 and extract 64x64 from each with random shift
- grayscale images
"""

set_random_seed(0)
global_step = 0
writer = SummaryWriter()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=256, help='batch_size')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_jigsaw_puzzle', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, args.exp_name)

    # build model
    encoder = ViTBackbone(pretrained=False).cuda()
    num_features = encoder.net.mlp_head[1].in_features
    encoder.net.mlp_head[1] = nn.Linear(in_features=num_features, out_features=512).cuda()
    model = ContextFreeNetwork(encoder, 512*4, 24).cuda()  # out_features of ViT * number of tiles

    # load dataset
    data_root = args.data_folder
    train_transform = get_transforms_pretraining_jigsaw_puzzle()
    train_data = CIFAR10Custom(data_root, train=True, transform=train_transform, download=True, unlabeled=True)
    val_data = CIFAR10Custom(data_root, val=True, transform=train_transform, download=True, unlabeled=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=4,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=4,
                                             pin_memory=True, drop_last=False)

    # TODO: loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_acc = validate(val_loader, model, criterion, epoch)

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best.pth".format(epoch)))
            best_val_loss = val_loss


# train one epoch over the whole training dataset.
def train(loader, model, criterion, optimizer, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.train()
    for i, (inputs, labels) in enumerate(loader):
        print(f"Trainstep: {i}")
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += criterion(outputs, labels).item() * batch_size
        total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
        total += batch_size

    mean_train_loss = total_loss / total
    mean_train_accuracy = total_accuracy / total
    scalar_dict = {}
    scalar_dict['Loss/train'] = mean_train_loss
    scalar_dict['Accuracy/train'] = mean_train_accuracy
    save_in_log(writer, epoch, scalar_dict=scalar_dict)
    return mean_train_loss, mean_train_accuracy


# validation function.
def validate(loader, model, criterion, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            batch_size = labels.size(0)
            total_loss += criterion(outputs, labels).item() * batch_size
            total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
            total += batch_size

    mean_val_loss = total_loss / total
    mean_val_accuracy = total_accuracy / total
    scalar_dict = {}
    scalar_dict['Loss/val'] = mean_val_loss
    scalar_dict['Accuracy/val'] = mean_val_accuracy
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_val_loss, mean_val_accuracy


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)