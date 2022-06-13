import numpy as np
import torch
import sys
import sparseconvnet as scn
from .data_loaders import DataGen, collatefn, LabelType
from next_sparseconvnet.networks.architectures import UNet
from torch.utils.tensorboard import SummaryWriter

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

def accuracy(true, pred, **kwrgs):
    return sum(true==pred)/len(true)

def train_one_epoch(epoch_id, net, criterion, optimizer, loader, label_type, nclass = 3):
    """
        Trains the net for all the train data one time
    """
    net.train()
    loss_epoch = 0
    if label_type== LabelType.Segmentation:
        metrics = IoU
        met_epoch = np.zeros(nclass)
    elif label_type == LabelType.Classification:
        metrics = accuracy
        met_epoch = 0
    for batchid, (coord, ener, label, event) in enumerate(loader):
        batch_size = len(event)
        ener, label = ener.cuda(), label.cuda()

        optimizer.zero_grad()

        output = net.forward((coord, ener, batch_size))

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

        softmax = torch.nn.Softmax(dim = 1)
        prediction = torch.argmax(softmax(output), 1)
        met_epoch += metrics(label.cpu(), prediction.cpu(), nclass=nclass)

    loss_epoch = loss_epoch / len(loader)
    met_epoch = met_epoch / len(loader)
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_ = f"\t Loss: {loss_epoch:.6f}"
    print(epoch_ + loss_)

    return loss_epoch, met_epoch


def valid_one_epoch(net, criterion, loader, label_type, nclass = 3):
    """
        Computes loss and metrics (IoU for segmentation and accuracy for classification)
        for all the validation data
    """
    net.eval()
    loss_epoch = 0
    if label_type== LabelType.Segmentation:
        metrics = IoU
        met_epoch = np.zeros(nclass)
    elif label_type == LabelType.Classification:
        metrics = accuracy
        met_epoch = 0

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
            met_epoch += metrics(label.cpu(), prediction.cpu(), nclass=nclass)

        loss_epoch = loss_epoch / len(loader)
        met_epoch = met_epoch / len(loader)
        loss_ = f"\t Validation Loss: {loss_epoch:.6f}"
        print(loss_)

    return loss_epoch, met_epoch



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



def train_net(*,
              nepoch,
              train_data_path,
              valid_data_path,
              train_batch_size,
              valid_batch_size,
              net,
              criterion,
              optimizer,
              label_type,
              checkpoint_dir,
              tensorboard_dir,
              num_workers,
              nevents_train = None,
              nevents_valid = None,
              augmentation  = False):
    """
        Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    train_gen = DataGen(train_data_path, label_type, nevents = nevents_train, augmentation = augmentation)
    valid_gen = DataGen(valid_data_path, label_type, nevents = nevents_valid)

    loader_train = torch.utils.data.DataLoader(train_gen,
                                               batch_size = train_batch_size,
                                               shuffle = True,
                                               num_workers = num_workers,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)
    loader_valid = torch.utils.data.DataLoader(valid_gen,
                                               batch_size = valid_batch_size,
                                               shuffle = True,
                                               num_workers = 1,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)

    start_loss = np.inf
    writer = SummaryWriter(tensorboard_dir)
    for i in range(nepoch):
        train_loss, train_met = train_one_epoch(i, net, criterion, optimizer, loader_train, label_type)
        valid_loss, valid_met = valid_one_epoch(net, criterion, loader_valid, label_type)

        if valid_loss < start_loss:
            save_checkpoint({'state_dict': net.state_dict(),
                             'optimizer': optimizer.state_dict()}, f'{checkpoint_dir}/net_checkpoint_{i}.pth.tar')
            start_loss = valid_loss

        writer.add_scalar('loss/train', train_loss, i)
        writer.add_scalar('loss/valid', valid_loss, i)
        if label_type == LabelType.Segmentation:
            for k, iou in enumerate(train_met):
                writer.add_scalar(f'iou/train_{k}class', iou, i)
            for k, iou in enumerate(valid_met):
                writer.add_scalar(f'iou/valid_{k}class', iou, i)
        elif label_type == LabelType.Classification:
            writer.add_scalar('acc/train', train_met, i)
            writer.add_scalar('acc/valid', valid_met, i)
        writer.flush()
    writer.close()



def predict_gen(data_path, net, label_type, batch_size, nevents):
    """
    A generator that yields a dictionary with output of collate plus
    output of  network.
    Parameters:
    ---------
        data_path : str
                    path to dataset
        net       : torch.nn.Model
                    network to use for prediction
        batch_size: int
        nevents   : int
                    Predict on only nevents first events from the dataset
    Yields:
    --------
        dict
            the elements of the dictionary are:
            coords      : np.array (2d) containing XYZ coordinate bin index
            label       : np.array containing original voxel label
            energy      : np.array containing energies per voxel
            dataset_id  : np.array containing dataset_id as in input file
            predictions : np.array (2d) containing predictions for all the classes
    """

    gen    = DataGen(data_path, label_type, nevents = nevents)
    loader = torch.utils.data.DataLoader(gen,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         num_workers = 1,
                                         collate_fn = collatefn,
                                         drop_last = False,
                                         pin_memory = False)

    net.eval()
    softmax = torch.nn.Softmax(dim = 1)
    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.cuda(), label.cuda()
            output = net.forward((coord, ener, batch_size))
            y_pred = softmax(output).cpu().detach().numpy()

            if label_type == LabelType.Classification:
                out_dict = dict(
                    label = label.cpu().detach().numpy(),
                    dataset_id = event,
                    prediction = y_pred)

            elif label_type == LabelType.Segmentation:
                # event is a vector of batch_size
                # to obtain event per voxel we need to look into inside batch id (last index in coords)
                # and find indices where id changes

                aux_id = coord[:, -1].cpu().detach().numpy()
                _, lengths = np.unique(aux_id, return_counts = True)
                dataset_id = np.repeat(event.numpy(), lengths)

                out_dict = dict(
                    coords      = coord[:, :3].cpu().detach().numpy(),
                    label       = label.cpu().detach().numpy(),
                    energy      = ener.cpu().detach().numpy().flatten(),
                    dataset_id  = dataset_id,
                    predictions = y_pred)
            yield out_dict
