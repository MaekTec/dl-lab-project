import os
import pickle

import numpy as np
import argparse
import torch
import torch.nn as nn
from pprint import pprint
from data.transforms import get_transforms_pretraining_jigsaw_puzzle, \
    get_transforms_pretraining_contrastive_predictive_coding
from models.contrastive_predictive_coding_network import ContrastivePredictiveCodingNetwork
from utils import check_dir, set_random_seed, get_logger, accuracy, save_in_log, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torchsummary
from models.context_free_network import ContextFreeNetwork
from tqdm import tqdm

"""
Papers: https://arxiv.org/pdf/1905.09272.pdf (cpc v2), https://arxiv.org/pdf/1807.03748.pdf (original cpc)
in paper they use the following schema:
- predict from top to down and vise-versa
- resize the image to 300×300 pixels and randomly extract a 260*260 pixel crop,
  then divide this image into a 6*6 grid of 80*80 patches with 36 pixel stride
- data augmentations (randomly drop 2 of 3 color channels, shearing, rotation,
  elastic deformations, color transforms, ...)
- see appendix of paper for more details

This implementation is slightly different from above, due to the smaller image size 
and not all data augmentation are public available.

Helpful implementations:
https://github.com/SeonghoBaek/CPC/blob/master/cpc.py
https://github.com/davidtellez/contrastive-predictive-coding-images
"""

set_random_seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--image-size', type=int, default=64, help='size of image')
    parser.add_argument('--num-patches-per-dim', type=int, default=4, help='in how many patches to split the image')
    parser.add_argument("--resnet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use ResNet instead of Vit")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "weight_decay", "bs", "epochs", "image_size", "num_patches_per_dim", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_contrastive_predictive_coding', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    args.splits = args.num_patches_per_dim**2

    pickle.dump(args, open(os.path.join(args.output_folder, "args.p"), "wb"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, args.exp_name)
    writer = SummaryWriter()

    # build model
    encoder_out_dim = 512
    if args.resnet:
        encoder = ResNet18Backbone(num_classes=encoder_out_dim).cuda()
    else:
        encoder = ViTBackbone(image_size=args.image_size, patch_size=16, num_classes=encoder_out_dim).cuda()

    model = ContrastivePredictiveCodingNetwork(encoder, encoder_out_dim, args.num_patches_per_dim).cuda()

    logger.info(model)
    # doesn't work due to tuple output of model (loss, acc)
    # torchsummary.summary(model, (args.splits, 3, args.image_size, args.image_size), args.bs)

    # load dataset
    data_root = args.data_folder
    train_transform = get_transforms_pretraining_contrastive_predictive_coding(args)
    train_data = CIFAR10Custom(data_root, train=True, transform=train_transform, download=True, unlabeled=True)
    val_data = CIFAR10Custom(data_root, val=True, transform=train_transform, download=True, unlabeled=True)
    # num_workers = 0, because ther is a bug in pytorch: https://github.com/pytorch/pytorch/issues/13246
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    for epoch in range(args.epochs):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, model, optimizer, scheduler, epoch, writer)
        val_loss, val_acc = validate(val_loader, model, epoch, writer)

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best.pth".format(epoch)))
            best_val_loss = val_loss


# train one epoch over the whole training dataset.
def train(loader, model, optimizer, scheduler, epoch, writer):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.train()
    for i, inputs in tqdm(enumerate(loader)):
        #if (i+1) % 200 == 0:
        #    break
        inputs = inputs.cuda()
        optimizer.zero_grad()
        loss, acc = model(inputs)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_accuracy += acc * batch_size
        total += batch_size
    scheduler.step()

    mean_train_loss = total_loss / total
    mean_train_accuracy = total_accuracy / total
    scalar_dict = {'Loss/train': mean_train_loss, 'Accuracy/train': mean_train_accuracy}
    save_in_log(writer, epoch, scalar_dict=scalar_dict)
    return mean_train_loss, mean_train_accuracy


# validation function.
def validate(loader, model, epoch, writer):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(loader)):
            #if (i + 1) % 200 == 0:
            #    break
            inputs = inputs.cuda()
            loss, acc = model(inputs)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_accuracy += acc * batch_size
            total += batch_size

    mean_val_loss = total_loss / total
    mean_val_accuracy = total_accuracy / total
    scalar_dict = {'Loss/val': mean_val_loss, 'Accuracy/val': mean_val_accuracy}
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_val_loss, mean_val_accuracy


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
