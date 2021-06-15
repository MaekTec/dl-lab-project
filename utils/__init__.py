import os
import torch
import random
import logging
import datetime
import numpy as np
import torch.backends.cudnn as cudnn


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_random_seed(seed):
    # Fix random seed to reproduce results
    # Documentation https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_logger(logdir, name, evaluate=False):
    # Set logger for saving process experimental information
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    logger.ts = ts
    if evaluate:
        file_path = os.path.join(logdir, "evaluate_{}.log".format(ts))
    else:
        file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    # strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr = logging.StreamHandler()
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)

    return logger


def save_in_log(log, save_step, scalar_dict=None, text_dict=None, image_dict=None, num_classes=1):
    if scalar_dict:
        [log.add_scalar(k, v, save_step) for k, v in scalar_dict.items()]
    if text_dict:
        [log.add_text(k, v, save_step) for k, v in text_dict.items()]
    if image_dict:
        for k, v in image_dict.items():
            if k=='sample':
                log.add_images(k, (v - v.min()) / (v.max() - v.min()), save_step)
            elif k=='vec':
                log.add_images(k, v.unsqueeze(1).unsqueeze(1), save_step)
            elif k=='gt':
                log.add_images(k, v.unsqueeze(1).expand(-1, 3, -1, -1).float()/num_classes, save_step)
            elif k=='pred':
                log.add_images(k, v.argmax(dim=1, keepdim=True).float()/num_classes, save_step)
            elif k=='att':
                assert isinstance(v, list)
                for idx, alpha in enumerate(v):
                    alpha -= torch.min(torch.min(alpha, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0]
                    alpha /= torch.max(torch.max(alpha, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0]
                    log.add_images(k+"_"+str(idx), alpha.unsqueeze(1), save_step)
            else:
                log.add_images(k, v, save_step)
    log.flush()
