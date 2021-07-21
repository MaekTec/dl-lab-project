import os
import pickle
import torch.nn as nn
import numpy as np
import argparse
import torch
from pprint import pprint
from models.contrastive_predictive_coding_network import ContrastivePredictiveCodingNetworkLinearClassification
from training import train, validate
from utils import check_dir, set_random_seed, get_logger, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
from models.context_free_network import ContextFreeNetwork
from data.transforms import get_transforms_downstream_training, \
    get_transforms_downstream_validation, get_transforms_pretraining_contrastive_predictive_coding, \
    get_transforms_downstream_contrastive_predictive_coding_validation, get_transforms_pretraining_jigsaw_puzzle, \
    get_transforms_downstream_jigsaw_puzzle_validation, get_transforms_downstream_jigsaw_puzzle_training
from enum import Enum
import torchsummary

set_random_seed(0)
writer = SummaryWriter()


class PretrainTask(Enum):
    none = 'none'
    rotation = 'rotation'
    jigsaw_puzzle = 'jigsaw_puzzle'
    cpc = 'cpc'
    moco = 'moco'

    def __str__(self):
        return self.value


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('pretrain_task', type=PretrainTask, choices=list(PretrainTask))
    parser.add_argument('--pretrain-path', type=str,
                        default="Path to pretraining output folder (directory with models/, logs/")
    parser.add_argument('--weight-init', type=str, default="models/ckpt_best.pth", help="weights within pretrain-path")
    # use if pretraining has specific model args
    parser.add_argument('--args-pretrain', type=str, default="args.p",
                        help="args file from pretraining within pretrain_path")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument("--fine-tune-last-layer", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Fine tune only the last layer")
    parser.add_argument('--lr-ftl', type=float, default=0.0002, help='learning rate for fine tune last layer')
    parser.add_argument('--lr-e2e', type=float, default=0.00005, help='learning rate for learning end-to-end')
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

    hparam_keys = ["pretrain_task", "fine_tune_last_layer", "lr_ftl", "lr_e2e", "weight_decay", "bs",
                   "epochs", "image_size", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'downstream', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    if args.args_pretrain is not None:
        args.args_pretrain = pickle.load(open(os.path.join(args.pretrain_path, args.args_pretrain), "rb"))

    pickle.dump(args, open(os.path.join(args.output_folder, "args.p"), "wb"))

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


def get_model(args):
    # model
    if args.resnet:
        model = ResNet18Backbone(num_classes=10).cuda()
        # torch.nn.init.zeros_(model.net.fc.weight)
    else:
        model = ViTBackbone(image_size=args.image_size, patch_size=16, num_classes=10).cuda()
        # torch.nn.init.zeros_(model.net.mlp_head[1].weight)

    input_dims = (3, args.image_size, args.image_size)
    # for fine-tune-last-layer option
    if args.resnet:
        last_layer = model.net.fc
    else:
        last_layer = model.net.mlp_head

    if args.pretrain_task is PretrainTask.none:
        pass

    elif args.pretrain_task is PretrainTask.rotation:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(args.pretrain_path, args.weight_init))

        if args.resnet:
            del pretrained_dict['net.fc.weight']
            del pretrained_dict['net.fc.bias']
        else:
            del pretrained_dict['net.mlp_head.1.weight']
            del pretrained_dict['net.mlp_head.1.bias']

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    elif args.pretrain_task is PretrainTask.jigsaw_puzzle:
        pretrained_dict = torch.load(os.path.join(args.pretrain_path, args.weight_init))

        # change encoder to be same as in pretraining
        encoder_dim = pretrained_dict["encoder.net.mlp_head.1.weight"].size()[0]
        num_features = model.net.mlp_head[1].in_features
        model.net.mlp_head[1] = nn.Linear(in_features=num_features, out_features=encoder_dim).cuda()

        # do not use last layer from pretrain
        input_dim = pretrained_dict['fc7.weight'].size()[1]
        del pretrained_dict['fc8.weight']
        del pretrained_dict['fc8.bias']

        encoder = model
        model = ContextFreeNetwork(encoder, input_dim, 10).cuda()

        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        last_layer = model.fc8  # use last two layers, because last layer is very small

        # args from pretraining
        args.number_of_permutations = args.args_pretrain.number_of_permutations
        args.num_tiles_per_dim = args.args_pretrain.num_tiles_per_dim
        args.splits = args.args_pretrain.splits

        input_dims = (args.splits, 3, args.image_size, args.image_size)
    elif args.pretrain_task is PretrainTask.cpc:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(args.pretrain_path, args.weight_init))

        if args.resnet:
            del pretrained_dict['encoder.net.fc.weight']
            del pretrained_dict['encoder.net.fc.bias']
            #  TODO
        else:
            encoder_dim = pretrained_dict["encoder.net.mlp_head.1.weight"].size()[0]
            num_features = model.net.mlp_head[1].in_features
            model.net.mlp_head[1] = nn.Linear(in_features=num_features, out_features=encoder_dim).cuda()

        for key in list(pretrained_dict.keys()):
            if "encoder" not in key:
                pretrained_dict.pop(key)
            else:
                pretrained_dict[key.replace("encoder.", "")] = pretrained_dict.pop(key)

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        args.num_patches_per_dim = args.args_pretrain.num_tiles_per_dim
        encoder = model
        model = ContrastivePredictiveCodingNetworkLinearClassification(encoder, encoder_dim, args.num_patches_per_dim,
                                                                       10).cuda()
        last_layer = model.fc

        input_dims = (args.args_pretrain.splits, 3, args.image_size, args.image_size)

    elif args.pretrain_task is PretrainTask.moco:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(args.pretrain_path, args.weight_init))

        if args.resnet:
            del pretrained_dict['f_q.net.fc.weight']
            del pretrained_dict['f_q.net.fc.bias']
        else:
            del pretrained_dict['f_q.net.mlp_head.1.weight']
            del pretrained_dict['f_q.net.mlp_head.1.bias']

        for key in list(pretrained_dict.keys()):
            if "f_q" not in key:
                pretrained_dict.pop(key)
            else:
                pretrained_dict[key.replace("f_q.", "")] = pretrained_dict.pop(key)

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise ValueError

    return model, last_layer, input_dims


def get_transforms(args):
    transform = get_transforms_downstream_training(args)
    transform_validation = get_transforms_downstream_validation(args)

    if args.pretrain_task is PretrainTask.none:
        pass
    elif args.pretrain_task is PretrainTask.rotation:
        pass
    elif args.pretrain_task is PretrainTask.jigsaw_puzzle:
        transform = get_transforms_downstream_jigsaw_puzzle_training(args)
        transform_validation = get_transforms_downstream_jigsaw_puzzle_validation(args)
    elif args.pretrain_task is PretrainTask.cpc:
        transform = get_transforms_pretraining_contrastive_predictive_coding(args)
        transform_validation = get_transforms_downstream_contrastive_predictive_coding_validation(args)
    elif args.pretrain_task is PretrainTask.moco:
        pass
    return transform, transform_validation


def main(args):
    logger = get_logger(args.logs_folder, args.exp_name)

    model, last_layer, input_dims = get_model(args)
    transform, transform_validation = get_transforms(args)

    if args.fine_tune_last_layer:
        args.lr = args.lr_ftl
        disable_gradients(model)

        for x in last_layer.parameters():
            x.requires_grad = True
    else:
        args.lr = args.lr_e2e

    logger.info(model)
    torchsummary.summary(model, input_dims, args.bs)

    data_root = args.data_folder
    train_data = CIFAR10Custom(data_root, train=True, download=True, transform=transform, unlabeled=False)
    val_data = CIFAR10Custom(data_root, val=True, download=True, transform=transform_validation, unlabeled=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                             pin_memory=True, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
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

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))

        val_loss, val_acc = validate(val_loader, model, criterion, epoch, writer)

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
