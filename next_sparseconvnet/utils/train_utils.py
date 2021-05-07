import numpy as np
import torch
import sys
import sparseconvnet as scn
from .data_loaders import DataGen, collatefn, LabelType
from next_sparseconvnet.networks.architectures import UNet

def IoU(true, pred, nclass = 3):
    """
        Intersection over union is a metric for semantic segmentation.
        It returns a IoU value for each class of our input tensors/arrays.
    """
    eps = sys.float_info.epsilon
    confusion_matrix = np.zeros((nclass, nclass))

    for i in range(len(true)):
        confusion_matrix[true[i]][pred[i]] += 1

    IoU = []
    for i in range(nclass):
        IoU.append((confusion_matrix[i, i] + eps) / (sum(confusion_matrix[:, i]) + sum(confusion_matrix[i, :]) - confusion_matrix[i, i] + eps))
    return np.array(IoU)


def train_one_epoch_segmentation(epoch_id, net, criterion, optimizer, loader, nclass = 3):
    """
        Trains the net for all the train data one time
    """
    net.train()
    loss_epoch, iou_epoch = 0, np.zeros(nclass)
    for batchid, (coord, ener, label, event) in enumerate(loader):
        batch_size = len(event)
        ener, label = ener.cuda(), label.cuda()

        optimizer.zero_grad()

        output = net.forward((coord, ener, batch_size))

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

        #IoU
        softmax = torch.nn.Softmax(dim = 1)
        prediction = torch.argmax(softmax(output), 1)
        iou_epoch += IoU(label.cpu(), prediction.cpu(), nclass = nclass)

    loss_epoch = loss_epoch / len(loader)
    iou_epoch = iou_epoch / len(loader)
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_ = f"\t Loss: {loss_epoch:.6f}"
    print(epoch_ + loss_)

    return loss_epoch, iou_epoch


def valid_one_epoch_segmentation(net, criterion, loader, nclass = 3):
    """
        Computes loss and IoU for all the validation data
    """
    net.eval()
    loss_epoch, iou_epoch = 0, np.zeros(nclass)
    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.cuda(), label.cuda()

            output = net.forward((coord, ener, batch_size))

            loss = criterion(output, label)

            loss_epoch += loss.item()

            #IoU
            softmax = torch.nn.Softmax(dim = 1)
            prediction = torch.argmax(softmax(output), 1)
            iou_epoch += IoU(label.cpu(), prediction.cpu())

        loss_epoch = loss_epoch / len(loader)
        iou_epoch = iou_epoch / len(loader)
        loss_ = f"\t Validation Loss: {loss_epoch:.6f}"
        print(loss_)

    return loss_epoch, iou_epoch
