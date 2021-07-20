import os
import numpy as np
import argparse
import torch
from pprint import pprint
from data.transforms import get_transforms_pretraining_rotation, custom_collate, get_transforms_pretraining_mpp
from utils import check_dir, set_random_seed, accuracy, get_logger, accuracy, save_in_log, str2bool
from models.pretraining_backbone import ViTBackbone, ResNet18Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
import torchsummary
from PatchPredictionNetwork import PatchPredictionLoss,PatchPredictionNetwork
from torchvision.transforms import transforms

# https://arxiv.org/pdf/1803.07728.pdf

set_random_seed(0)
global_step = 0
writer = SummaryWriter()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--bs', type=int, default=2, help='batch_size')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument("--resnet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use ResNet instead of Vit")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "epochs", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_patch_prediction', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, args.exp_name)

    model = ViTBackbone(image_size=64, patch_size=args.patch_size,num_classes=1000).cuda()
    patch_prediction_network = PatchPredictionNetwork(transformer=model,patch_size=args.patch_size,dim=1024,mask_prob=0.15, replace_prob=0.50).cuda()

    # load dataset
    data_root = args.data_folder
    train_transform = get_transforms_pretraining_mpp()
    train_data = CIFAR10Custom(data_root, train=True, transform=train_transform, download=True, unlabeled=True)
    val_data = CIFAR10Custom(data_root, val=True, transform=train_transform, download=True, unlabeled=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,pin_memory=True, drop_last=True, )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=2,pin_memory=True, drop_last=True,)

    #criterion = mpp_loss()
    #optimizer =
    criterion = PatchPredictionLoss(patch_size=args.patch_size, channels=3, output_channel_bits=3, max_pixel_val=1.0, mean=None, std=None).cuda()
    optimizer = torch.optim.Adam(patch_prediction_network.parameters(), lr=args.lr)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf

    for epoch in range(args.epochs):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, patch_prediction_network, criterion, optimizer, epoch)
        #scheduler.step()
        val_loss, val_acc = validate(val_loader, patch_prediction_network, criterion, epoch)

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
    for i, inputs in enumerate(loader):
        #print(f"Trainstep: {i}")
        inputs = inputs.cuda()
        #labels = labels.cuda()
        optimizer.zero_grad()
        #print(inputs.shape)
        outputs, masks = model(inputs)
        loss = criterion(outputs, inputs, masks)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        #total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
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
        for i , inputs in enumerate(loader):
            inputs = inputs.cuda()
            outputs, masks = model(inputs)

            batch_size = inputs.shape[0]
            total_loss += criterion(outputs, inputs, masks).item() * batch_size
            #total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
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
