import torch
from tqdm import tqdm
from utils import accuracy, save_in_log


# train one epoch over the whole training dataset.
def train(loader, model, criterion, optimizer, scheduler, epoch, writer):
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
    scalar_dict = {'Loss/train': mean_train_loss, 'Accuracy/train': mean_train_accuracy}
    save_in_log(writer, epoch, scalar_dict=scalar_dict)
    return mean_train_loss, mean_train_accuracy


# validation function.
def validate(loader, model, criterion, epoch, writer):
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
    scalar_dict = {'Loss/val': mean_val_loss, 'Accuracy/val': mean_val_accuracy}
    save_in_log(writer, epoch, scalar_dict=scalar_dict)

    return mean_val_loss, mean_val_accuracy
