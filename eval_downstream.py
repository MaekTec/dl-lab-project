import os
import pickle

import numpy as np
import argparse
import torch
from pprint import pprint

from downstream import get_model, get_transforms
from models.contrastive_predictive_coding_network import ContrastivePredictiveCodingNetworkLinearClassification
from utils import check_dir, set_random_seed, accuracy, get_logger, accuracy, save_in_log, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.context_free_network import ContextFreeNetwork
from data.transforms import get_transforms_downstream_training, \
    get_transforms_downstream_validation, get_transforms_pretraining_contrastive_predictive_coding, \
    get_transforms_downstream_contrastive_predictive_coding_validation, get_transforms_pretraining_jigsaw_puzzle, \
    get_transforms_downstream_jigsaw_puzzle_validation, get_transforms_downstream_jigsaw_puzzle_training
from tqdm import tqdm
from downstream import PretrainTask
from enum import Enum
import torchsummary

set_random_seed(0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--downstream-path', type=str,
                        default="Path to downstream output folder (directory with models/, logs/")
    parser.add_argument('--weight-init', type=str, default="models/ckpt_best.pth",
                        help="weights within downstream-path")
    # use if pretraining has specific model args
    parser.add_argument('--args-downstream', type=str, default="args.p",
                        help="args file from downstream within downstream_path")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--bs', type=int, default=256, help='batch_size')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["bs"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'eval_downstream', args.exp_name))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    if args.args_downstream is not None:
        args.args_downstream = pickle.load(open(os.path.join(args.downstream_path, args.args_downstream), "rb"))

    return args


def main(args):
    logger = get_logger(args.logs_folder, args.exp_name)
    writer = SummaryWriter()

    model, _, input_dims = get_model(args.args_downstream)
    model.load_state_dict(torch.load(os.path.join(args.downstream_path, args.weight_init)))
    _, transform_validation = get_transforms(args.args_downstream)

    logger.info(model)
    torchsummary.summary(model, input_dims, args.bs)

    data_root = args.data_folder
    test_data = CIFAR10Custom(data_root, test=True, download=True, transform=transform_validation, unlabeled=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, num_workers=1)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('test_data {}'.format(test_data.__len__()))

    test_loss, test_acc = test(test_loader, model, criterion, writer)
    logger.info('Test loss: {}'.format(test_loss))
    logger.info('Test accuracy: {}'.format(test_acc))


# test function.
def test(loader, model, criterion, writer):
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

    mean_test_loss = total_loss / total
    mean_test_accuracy = total_accuracy / total
    scalar_dict = {}
    scalar_dict['Loss/test'] = mean_test_loss
    scalar_dict['Accuracy/test'] = mean_test_accuracy
    save_in_log(writer, 0, scalar_dict=scalar_dict)

    return mean_test_loss, mean_test_accuracy


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)