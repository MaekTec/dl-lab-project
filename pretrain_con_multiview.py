# https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c

import os
import argparse
from pprint import pprint
from data.transforms import *

from utils import check_dir, set_random_seed, get_logger, save_in_log, str2bool
from models.pretraining_backbone import ResNet18Backbone , CMC_ViT_Backbone
from torch.utils.tensorboard import SummaryWriter
from data.CIFAR10Custom import CIFAR10Custom
from tqdm import tqdm
from utils.cmc_criterions import ContLoss


set_random_seed(0)
writer = SummaryWriter()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bs', type=int, default=4, help='batch_size')
    parser.add_argument('--epochs', type=int, default=60, help='epochs')
    parser.add_argument('--image-size', type=int, default=64, help='size of image')
    parser.add_argument("--resnet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use ResNet instead of Vit")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--nce_k', type=int, default=2000)
    parser.add_argument('--tau',type=float,default=0.07)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()

    hparam_keys = ["lr", "weight_decay", "bs", "epochs", "image_size", "resnet"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain_cmc', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))
    return args



def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.logs_folder, args.exp_name)

    # build model
    encoder_out_dim = 128
    if args.resnet:
        encoder = ResNet18Backbone(num_classes=encoder_out_dim).to(device)
    else:
        encoder = CMC_ViT_Backbone(image_size=args.image_size, patch_size=16, num_classes=encoder_out_dim).to(device)

    model = encoder
    logger.info(model)


    #torchsummary.summary(model, (args.splits, 3, args.image_size, args.image_size), args.bs)

    # load dataset
    data_root = args.data_folder
    train_transform = get_transforms_pretraining_cmc(args)
    train_data = CIFAR10Custom(data_root, train=True, transform=train_transform, download=True, unlabeled=True, pretrain_task='cmc')
    val_data = CIFAR10Custom(data_root, val=True, transform=train_transform, download=True, unlabeled=True, pretrain_task='cmc')
    # num_workers = 0, because ther is a bug in pytorch: https://github.com/pytorch/pytorch/issues/13246
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=False)


    # for i, (input, index) in enumerate(train_loader):
    #     print('input.shape',input.shape)
    #     print('index.shape',index.shape)
    #     input = input.to(device, dtype=torch.float32)
    #     feat_l, feat_ab = model(input)
    #     #loss = loss_function(feat_l, feat_ab)
    #     #print('loss=',loss.item())
    #     print('feat_l.shape=',feat_l.shape, 'feat_ab.shape',feat_ab.shape)
    #     break

    #contrast = NCEAverage(args.feat_dim, train_data.__len__(), args.nce_k, 0.07, 0.5, True).cuda()
    #criterion_l = NCESoftmaxLoss().cuda()
    #criterion_ab = NCESoftmaxLoss().cuda()

    criterion = ContLoss(args.tau)

    #criterion = torch.nn.CrossEntropyLoss() #TODO
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    for epoch in range(args.epochs):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, scheduler, epoch)
        #train_loss, train_acc = train2(train_loader, model, criterion_l,criterion_ab, optimizer, scheduler, epoch,contrast)
        val_loss, val_acc = validate(val_loader, model, criterion, epoch)

        logger.info('Training loss: {}'.format(train_loss))
        logger.info('Training accuracy: {}'.format(train_acc))
        logger.info('Validation loss: {}'.format(val_loss))
        logger.info('Validation accuracy: {}'.format(val_acc))

        # save model
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(args.model_folder, f"ckpt_best_{epoch}.pth".format(epoch)))
            print('model saved..')
            best_val_loss = val_loss

# train one epoch over the whole training dataset.
def train(loader, model, criterion, optimizer, scheduler, epoch):
    total_loss = 0
    total_accuracy = 0
    total = 0
    model.train()
    epoch_losses_train = []
    print('LR=',scheduler.get_last_lr())
    for i, (inputs, index) in tqdm(enumerate(loader)):
        #print("i=",i)
        #if ((i+1) % 30) == 0:
        #    break
        inputs = inputs.to(device)
        optimizer.zero_grad()
        output_l, output_ab = model(inputs)
        loss = criterion(output_l, output_ab)
        epoch_losses_train.append(loss.cpu().data.item())
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        #total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
        total += batch_size
    scheduler.step()

    mean_train_loss = total_loss / total
    mean_train_loss = sum(epoch_losses_train)/len(epoch_losses_train)
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
    epoch_losses_val = []

    with torch.no_grad():
        for i, (inputs, index) in tqdm(enumerate(loader)):
            #print("i=",i)
            #if ((i+1) % 30) == 0:
            #    break
            inputs = inputs.to(device)
            output_l, output_ab = model(inputs)
            loss = criterion(output_l, output_ab)
            epoch_losses_val.append(loss.cpu().data.item())
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            #total_accuracy += accuracy(outputs, labels)[0].item() * batch_size
            total += batch_size

    mean_val_loss = total_loss / total
    mean_val_loss = sum(epoch_losses_val)/len(epoch_losses_val)
    mean_val_accuracy = total_accuracy / total
    scalar_dict = {}
    scalar_dict['Loss/val'] = mean_val_loss
    scalar_dict['Accuracy/val'] = mean_val_accuracy
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_val_loss, mean_val_accuracy


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda")
    print(f"Running on device: {device}")

    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)




