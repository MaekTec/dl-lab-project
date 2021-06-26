import os
import numpy as np
import argparse
import torch
from pprint import pprint
from utils import check_dir, set_random_seed, accuracy, get_logger, accuracy, save_in_log, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.context_free_network import ContextFreeNetwork
from data.transforms import get_transforms_downstream_training, \
    get_transforms_downstream_validation
from tqdm import tqdm
from enum import Enum
import torchsummary

set_random_seed(0)
writer = SummaryWriter()


class PretrainTask(Enum):
    none = 'none'
    rotation = 'rotation'
    jigsaw_puzzle = 'jigsaw_puzzle'

    def __str__(self):
        return self.value


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('pretrain_task', type=PretrainTask, choices=list(PretrainTask))
    parser.add_argument('--weight-init', type=str, default="ImageNet")
    parser.add_argument("--fine-tune-last-layer", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Fine tune only the last layer")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--bs', type=int, default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=60, help='epochs')
    parser.add_argument('--image-size', type=int, default=64, help='size of image')
    parser.add_argument("--resnet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use ResNet instead of Vit")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["pretrain_task", "fine_tune_last_layer", "lr", "weight_decay", "bs", "epochs", "image_size", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'downstream', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def disable_gradients(model) -> None:
    """
    Freezes the layers of a model
    Args:
        model: The model with the layers to freeze
    Returns:
        None
    """
    # Iterate over model parameters and disable requires_grad
    for x in model.parameters():
        x.requires_grad = False
    return model


def main(args):
    logger = get_logger(args.output_folder, args.exp_name)

    # model
    if args.resnet:
        model = ResNet18Backbone(num_classes=10).cuda()
    else:
        model = ViTBackbone(image_size=args.image_size, patch_size=16, num_classes=10).cuda()

    if args.pretrain_task is PretrainTask.none:
        pass

    elif args.pretrain_task is PretrainTask.rotation:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.weight_init)
        del pretrained_dict['net.mlp_head.1.weight']
        del pretrained_dict['net.mlp_head.1.bias']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # replace the last two MLP layers as done in paper.
        #num_ftrs = model.net.mlp_head[1].in_features
        #model.net.mlp_head[0] = nn.Identity()
        #model.net.mlp_head[1] = nn.Linear(in_features=num_ftrs, out_features=10).cuda()
        #torch.nn.init.zeros_(model.net.mlp_head[1].weight)

    elif args.pretrain_task is PretrainTask.jigsaw_puzzle:
        encoder = ViTBackbone(image_size=args.image_size, patch_size=16, num_classes=64).cuda()  # TODO

        num_features = encoder.net.mlp_head[1].in_features
        encoder.net.mlp_head[1] = nn.Linear(in_features=num_features, out_features=10).cuda()
        torch.nn.init.zeros_(encoder.net.mlp_head[1].weight)
        #model = ContextFreeNetwork(encoder, 512 * 4, 24).cuda()  # out_features of ViT * number of tiles
        #model.load_state_dict(torch.load(args.weight_init))

        # replacing the last layer
        #num_ftrs = model.fc9.in_features
        #model.fc9 = nn.Linear(in_features=num_ftrs, out_features=10).cuda()
        #torch.nn.init.zeros_(model.fc9.weight)
    else:
        return

    logger.info(model)
    torchsummary.summary(model, (3, args.image_size, args.image_size), args.bs)

    data_root = args.data_folder
    transform = get_transforms_downstream_training(args)
    transform_validation = get_transforms_downstream_validation(args)

    train_data = CIFAR10Custom(data_root,
                               train=True,
                               download=True,
                               transform=transform,
                               unlabeled=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True)

    val_data = CIFAR10Custom(data_root,
                             val=True,
                             download=True,
                             transform=transform_validation,
                             unlabeled=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                             pin_memory=True, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
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

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))

        val_loss, val_acc = validate(val_loader, model, criterion, epoch)

        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args.model_folder, "ckpt_best.pth".format(epoch)))
            best_val_loss = val_loss


def train(loader, model, criterion, optimizer, scheduler, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.train()
    for i, (inputs, labels) in tqdm(enumerate(loader)):
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
        for i, (inputs, labels) in tqdm(enumerate(loader)):
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
