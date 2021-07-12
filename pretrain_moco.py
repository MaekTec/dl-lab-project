import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from pprint import pprint
from data.transforms import get_transforms_pretraining_jigsaw_puzzle, \
    get_transforms_pretraining_contrastive_predictive_coding, get_transforms_pretraining_moco
from models.moco_network import MoCoNetwork
from utils import check_dir, set_random_seed, get_logger, accuracy, save_in_log, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torchsummary
from tqdm import tqdm

"""
v1: https://arxiv.org/pdf/1911.05722.pdf
v2: https://arxiv.org/pdf/2003.04297.pdf (this, but we use the MLP head from the ViT)

- pretext task: instance discrimination task:
    - a query matches a key if they are encoded views (e.g., different crops) of the same image
- InfoNCE
- encoder output normalized with L2-norm, batch normalization?
- 224×224-pixel  crop  is  taken  from  a  randomly resized image, and then undergoes random color jittering,
  random horizontal flip, and random grayscale con-version
- momentum = 0.999
"""

set_random_seed(0)
writer = SummaryWriter()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')
    parser.add_argument('--image-size', type=int, default=64, help='size of image')
    parser.add_argument("--resnet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use ResNet instead of Vit")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "weight_decay", "bs", "epochs", "image_size", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_moco', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

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

    model = MoCoNetwork(encoder, encoder_out_dim).cuda()

    logger.info(model)
    #torchsummary.summary(model, (args.splits, 3, args.image_size, args.image_size), args.bs)

    # load dataset
    data_root = args.data_folder
    train_transform = get_transforms_pretraining_moco(args)
    train_data = CIFAR10Custom(data_root, train=True, transform=train_transform, download=True, unlabeled=True)
    val_data = CIFAR10Custom(data_root, val=True, transform=train_transform, download=True, unlabeled=True)
    # num_workers = 0, because ther is a bug in pytorch: https://github.com/pytorch/pytorch/issues/13246
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=0,
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
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, scheduler, epoch)
        val_loss, val_acc = validate(val_loader, model, epoch)

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best.pth".format(epoch)))
            best_val_loss = val_loss


# train one epoch over the whole training dataset.
def train(loader, model, criterion, optimizer, scheduler, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.train()
    for i, inputs in tqdm(enumerate(loader)):
        inputs = [i.cuda() for i in inputs]
        optimizer.zero_grad()
        outputs, labels = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += criterion(outputs, labels).item() * batch_size
        total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
        total += batch_size
    scheduler.step()

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
        for i, inputs in tqdm(enumerate(loader)):
            inputs = inputs.cuda()
            outputs, labels = model(inputs)

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
