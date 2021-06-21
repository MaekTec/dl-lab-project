import os
import numpy as np
import argparse
import torch
from pprint import pprint
from utils import check_dir, set_random_seed, accuracy, get_logger, accuracy, save_in_log
from models.pretraining_backbone import ViTBackbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


set_random_seed(0)
global_step = 0
writer = SummaryWriter()



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('weight_init', type=str, default="ImageNet")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_rotation', args.exp_name))
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
    pretrained_model = ViTBackbone(pretrained=False).cuda()
    print(pretrained_model.net.mlp_head)
    num_ftrs = pretrained_model.net.mlp_head[1].in_features
    print(args.weight_init)
    pretrained_model.load_state_dict(torch.load(args.weight_init))

    #disable_gradients(pretrained_model)
    pretrained_model.net.mlp_head[0]= nn.Identity()
    pretrained_model.net.mlp_head[1] = nn.Linear(in_features=num_ftrs, out_features=10).cuda()
    torch.nn.init.zeros_(pretrained_model.net.mlp_head[1].weight)
    print(pretrained_model)
    data_root = args.data_folder
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #create new func

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
                               transform=transform,
                               unlabeled=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=args.lr, momentum=0.9)
    scheduler =CosineAnnealingLR(optimizer, T_max=15, verbose=True )
    #mdlmslds

    #optimizer = torch.optim.Adam(pretrained_model.parameters(),betas=(0.9,0.999),weight_decay=0.1 )


    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    for epoch in range(50):  # 8
        logger.info("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, pretrained_model, criterion, optimizer, epoch , scheduler)

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))

        val_loss, val_acc = validate(val_loader, pretrained_model, criterion, epoch)

        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        torch.save(pretrained_model.state_dict(), os.path.join(args.model_folder, "downstream_best_.pth".format(epoch)))


def train(loader, model, criterion, optimizer, epoch, scheduler):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.train()
    for i, (inputs, labels) in enumerate(loader):
        #print(f"Trainstep: {i}")
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

