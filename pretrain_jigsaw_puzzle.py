import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from pprint import pprint
from data.transforms import get_transforms_pretraining_jigsaw_puzzle
from training import train, validate
from utils import check_dir, set_random_seed, get_logger, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torchsummary
from models.context_free_network import ContextFreeNetwork
import pickle

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
writer = SummaryWriter()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')
    parser.add_argument('--image-size', type=int, default=64, help='size of image')
    parser.add_argument('--num-tiles-per-dim', type=int, default=3, help='in how many tiles to split the image')
    parser.add_argument('--number-of-permutations', type=int, default=64, help='number of different permutations')
    parser.add_argument("--resnet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use ResNet instead of Vit")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "weight_decay", "bs", "epochs", "image_size", "num_tiles_per_dim", "number_of_permutations", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_jigsaw_puzzle', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    args.splits = args.num_tiles_per_dim**2

    pickle.dump(args, open(os.path.join(args.output_folder, "args.p"), "wb"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, args.exp_name)

    # build model
    encoder_out_dim = 512
    if args.resnet:
        encoder = ResNet18Backbone(num_classes=encoder_out_dim).cuda()
    else:
        encoder = ViTBackbone(image_size=args.image_size, patch_size=16, num_classes=encoder_out_dim).cuda()

    model = ContextFreeNetwork(encoder, encoder_out_dim*args.splits, args.number_of_permutations).cuda()

    logger.info(model)
    torchsummary.summary(model, (args.splits, 3, args.image_size, args.image_size), args.bs)

    # load dataset
    data_root = args.data_folder
    train_transform = get_transforms_pretraining_jigsaw_puzzle(args)
    train_data = CIFAR10Custom(data_root, train=True, transform=train_transform, download=True, unlabeled=True)
    val_data = CIFAR10Custom(data_root, val=True, transform=train_transform, download=True, unlabeled=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=4,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=4,
                                             pin_memory=True, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    for epoch in range(args.epochs):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, scheduler, epoch, writer)
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, writer)

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best.pth".format(epoch)))
            best_val_loss = val_loss


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
