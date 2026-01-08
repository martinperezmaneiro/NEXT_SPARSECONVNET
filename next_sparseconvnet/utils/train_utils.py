import numpy as np
import torch
import sys
import inspect
from time import time
import sparseconvnet as scn
from .data_loaders import DataGen, collatefn, LabelType, worker_init_fn
from next_sparseconvnet.networks.architectures import UNet
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from itertools import cycle

def model_supports_mode(model):
    sig = inspect.signature(model.forward)
    return 'mode' in sig.parameters

def log_losses(writer, loss, prefix, step):
    if isinstance(loss, (list, tuple)):
        for i, l in enumerate(loss):
            writer.add_scalar(f"{prefix}_{i}", l, step)
    else:
        writer.add_scalar(prefix, loss, step)

def get_name_of_scheduler(scheduler):
    """
    Get the name of the scheduler.
    """
    for name, obj in lr_scheduler.__dict__.items():
        if inspect.isclass(obj):
            if isinstance(scheduler, obj):
                return name
    return None

def IoU(true, pred, nclass=3):
    """
    Intersection over union (IoU) metric for semantic segmentation.
    It returns a IoU value for each class of our input tensors/arrays. 
    This is a vectorized and much faster implementation.
    """
    eps = np.finfo(float).eps
    
    # Flatten the arrays to ensure they are 1D
    true = true.flatten()
    pred = pred.flatten()

    # Calculate the confusion matrix using NumPy's bincount
    # This is significantly faster than a Python loop
    n_pixels = len(true)
    y_true_f = nclass * true.astype(int) + pred.astype(int)
    confusion_matrix = np.bincount(y_true_f, minlength=nclass**2).reshape(nclass, nclass)

    # Calculate IoU for each class from the confusion matrix
    union_per_class = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    IoU_per_class = (np.diag(confusion_matrix) + eps) / (union_per_class + eps)
    
    return IoU_per_class

#def IoU(true, pred, nclass = 3):
#    """
#        Intersection over union is a metric for semantic segmentation.
#        It returns a IoU value for each class of our input tensors/arrays.
#    """
#    eps = sys.float_info.epsilon
#    confusion_matrix = np.zeros((nclass, nclass))
#
#    for i in range(len(true)):
#        confusion_matrix[true[i]][pred[i]] += 1
#
#    IoU = []
#    for i in range(nclass):
#        IoU.append((confusion_matrix[i, i] + eps) / (sum(confusion_matrix[:, i]) + sum(confusion_matrix[i, :]) - confusion_matrix[i, i] + eps))
#    return np.array(IoU)

def accuracy(true, pred, **kwrgs):
    return np.mean(true == pred)
    #return sum(true==pred)/len(true)

def train_one_epoch(epoch_id, net, criterion, optimizer, loader, label_type, nclass = 3, device = 'cuda'):
    """
        Trains the net for all the train data one time
    """
    t = time()
    net.train()
    loss_epoch = 0
    if label_type== LabelType.Segmentation:
        metrics = IoU
        met_epoch = np.zeros(nclass)
    elif label_type == LabelType.Classification:
        metrics = accuracy
        met_epoch = 0
    t_batch = time()
    for batchid, (coord, ener, label, event) in enumerate(loader):
        print('Batch: ', batchid)
        print((time() - t_batch) / 60, ' mins - load batch')
        sys.stdout.flush()
        batch_size = len(event)
        t_ = time()
        ener, label = ener.to(device), label.to(device)

        print((time() - t_) / 60, ' mins - tensors to cuda')
        sys.stdout.flush()

        optimizer.zero_grad()

        t_ = time()

        output = net((coord, ener, batch_size))

        print((time() - t_) / 60, ' mins - net forward')
        sys.stdout.flush()

        t_ = time()

        loss = criterion(output, label)

        print((time() - t_) / 60, ' mins - loss computation')
        sys.stdout.flush()

        t_ = time()
        loss.backward()
        print((time() - t_) / 60, ' mins - backward step')
        sys.stdout.flush()

        t_ = time()
        optimizer.step()
        print((time() - t_) / 60, ' mins - optimizer update')
        sys.stdout.flush()

        loss_epoch += loss.item()

        t_ = time()
        with torch.autograd.no_grad():
            softmax = torch.nn.Softmax(dim = 1)
            prediction = torch.argmax(softmax(output), 1)
            met_epoch += metrics(label.cpu().numpy(), prediction.cpu().numpy(), nclass=nclass)
            print((time() - t_) / 60, ' mins - metrics computation')
            sys.stdout.flush()
            t_batch = time()

    loss_epoch = loss_epoch / len(loader)
    met_epoch = met_epoch / len(loader)
    epoch_ = f"Epoch: {epoch_id}"
    loss_ = f"\n Training   Loss: {loss_epoch:.6f}"
    time_ = f"\t Time: {(time() - t) / 60:.2f} mins"
    print(epoch_ + loss_ + time_)
    sys.stdout.flush()

    return loss_epoch, met_epoch

def train_one_epoch_dann(
    epoch_id, 
    net, 
    criterion_label,          # CrossEntropy
    criterion_domain,         # BCEWithLogitsLoss
    optimizer,
    loader_mc,                # MC dataloader (labeled)
    loader_data,              # Data dataloader (unlabeled)
    nclass=3,
    device='cuda'
):
    """
    DANN training loop: MC used for labels, MC+Data used for domain.
    """
    t_start = time()
    net.train()

    loss_epoch, loss_label_epoch, loss_domain_epoch = 0, 0, 0
    met_epoch, met_domain_epoch = 0, 0

    # iterate over MC and Data simultaneously
    # I use cycle() for data as I'm going to have less data instances in general, so it will loop until mc ends
    for batch_id, ((coord_mc, ener_mc, label_mc, evt_mc),
                   (coord_dt, ener_dt, _,      evt_dt)) in enumerate(zip(loader_mc, cycle(loader_data))):

        print("\nBatch:", batch_id)
        sys.stdout.flush()
        batch_size_mc = len(evt_mc)
        batch_size_dt = len(evt_dt)

        # ---------------------------
        # Move tensors to device
        # ---------------------------
        ener_mc   = ener_mc.to(device)
        label_mc  = label_mc.to(device)
        ener_dt   = ener_dt.to(device)

        optimizer.zero_grad()

        # ==================================================
        # 1. Classification loss (MC only)
        # ==================================================
        out_label = net((coord_mc, ener_mc, batch_size_mc), mode='label')
        loss_label = criterion_label(out_label, label_mc)

        # ==================================================
        # 2. Domain loss (MC → domain=1, Data → domain=0)
        # ==================================================

        # MC → domain=1
        out_dom_mc = net((coord_mc, ener_mc, batch_size_mc), mode='domain')
        dom_label_mc = torch.ones((batch_size_mc, 1), device=device)
        loss_dom_mc = criterion_domain(out_dom_mc, dom_label_mc)

        # Data → domain=0
        out_dom_dt = net((coord_dt, ener_dt, batch_size_dt), mode='domain')
        dom_label_dt = torch.zeros((batch_size_dt, 1), device=device)
        loss_dom_dt = criterion_domain(out_dom_dt, dom_label_dt)

        # average the domain loss
        loss_domain = 0.5 * (loss_dom_mc + loss_dom_dt)

        # ==================================================
        # 3. Combined loss
        # ==================================================
        loss = loss_label + loss_domain

        # Backprop
        loss.backward()
        optimizer.step()

        # accumulate loss
        loss_epoch += loss.item()
        loss_label_epoch += loss_label.item()
        loss_domain_epoch += loss_domain.item()

        # ==================================================
        # 4. Metrics (MC only)
        # ==================================================
        with torch.no_grad():
            softmax = torch.nn.Softmax(dim = 1)
            pred = torch.argmax(softmax(out_label), 1)
            met_epoch += accuracy(label_mc.cpu().numpy(),
                                  pred.cpu().numpy(), nclass=nclass)
            
            domain_acc_mc = ((out_dom_mc > 0.5).float() == 1).float().mean().item()
            domain_acc_dt = ((out_dom_dt > 0.5).float() == 0).float().mean().item()

            met_domain_epoch += 0.5 * (domain_acc_mc + domain_acc_dt)

    # normalize
    n_batches = len(loader_mc) # min(len(loader_mc), len(loader_data)) # I use the mc loader because data loader will loop until mc loader ends
    loss_label_epoch /= n_batches
    loss_domain_epoch /= n_batches

    met_epoch /= n_batches
    met_domain_epoch /= n_batches

    print(f"\nEpoch {epoch_id} | Loss={loss_epoch:.4f} | Acc={met_epoch:.4f} | Domain Acc= {met_domain_epoch:.4f} | Time={(time()-t_start)/60:.2f} min")
    sys.stdout.flush()

    return [loss_label_epoch, loss_domain_epoch], [met_epoch, met_domain_epoch]


def valid_one_epoch(net, criterion, loader, label_type, nclass = 3, device = 'cuda'):
    """
        Computes loss and metrics (IoU for segmentation and accuracy for classification)
        for all the validation data
    """
    t = time()
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
            ener, label = ener.to(device), label.to(device)

            # In the case of a DANN architecture, we always validate with mode = 'label',
            # but as it is the default mode, we don't add here a condition
            output = net((coord, ener, batch_size))

            loss = criterion(output, label)

            loss_epoch += loss.item()

            #IoU
            softmax = torch.nn.Softmax(dim = 1)
            prediction = torch.argmax(softmax(output), 1)
            met_epoch += metrics(label.cpu().numpy(), prediction.cpu().numpy(), nclass=nclass)

        loss_epoch = loss_epoch / len(loader)
        met_epoch = met_epoch / len(loader)
        loss_ = f" Validation Loss: {loss_epoch:.6f}"
        time_ = f"\t Time: {(time() - t) / 60:.2f} mins"
        print(loss_ + time_)
        sys.stdout.flush()

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
              scheduler,
              label_type,
              checkpoint_dir,
              tensorboard_dir,
              num_workers,
              train_data_domain_path = None,
              criterion_domain = None,
              nevents_train = None,
              nevents_valid = None,
              augmentation  = False,
              seglabel_name = 'segclass',
              feature_name  = ['energy'],
              nclass = 3,
              device = 'cuda'):
    """
        Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    # flag to know if the net is DANN or not
    use_dann = getattr(net, "is_dann", False)
    if use_dann:
        assert train_data_domain_path is not None, "train_data_domain_path required for DANN"
        assert criterion_domain is not None, "criterion_domain required for DANN"
    
    if device == 'cuda': pin_mem = True
    else: pin_mem = False

    met_name = 'iou' if label_type == LabelType.Segmentation else 'acc'

    t = time()
    train_gen = DataGen(train_data_path, label_type, nevents = nevents_train, augmentation = augmentation, seglabel_name = seglabel_name, feature_name = feature_name)
    valid_gen = DataGen(valid_data_path, label_type, nevents = nevents_valid, seglabel_name = seglabel_name, feature_name = feature_name)    

    loader_train = torch.utils.data.DataLoader(train_gen,
                                               batch_size = train_batch_size,
                                               shuffle = True,
                                               num_workers = num_workers,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = pin_mem, 
                                               persistent_workers = True,
                                               worker_init_fn = worker_init_fn)
    loader_valid = torch.utils.data.DataLoader(valid_gen,
                                               batch_size = valid_batch_size,
                                               shuffle = False,
                                               num_workers = 1,
                                               collate_fn = collatefn,
                                               drop_last = False,
                                               pin_memory = pin_mem, 
                                               persistent_workers = True,
                                               worker_init_fn = worker_init_fn)
    # LOAD NON LABELLED DATA IN CASE WE USE DANN
    if use_dann:
        train_domain_gen = DataGen(train_data_domain_path, label_type, nevents = nevents_train, augmentation = augmentation, seglabel_name = seglabel_name, feature_name = feature_name)
        loader_domain_train = torch.utils.data.DataLoader(train_domain_gen,
                                                            batch_size = train_batch_size,
                                                            shuffle = True,
                                                            num_workers = num_workers,
                                                            collate_fn = collatefn,
                                                            drop_last = True,
                                                            pin_memory = pin_mem, 
                                                            persistent_workers = True,
                                                            worker_init_fn = worker_init_fn)


    print('Data loaded ({:.2f} min)'.format((time() - t) / 60))
    
    start_loss = np.inf
    writer = SummaryWriter(tensorboard_dir)
    for i in range(nepoch):
        if use_dann:
            train_loss, train_met = train_one_epoch_dann(i, net, criterion, criterion_domain, optimizer, loader_train, loader_domain_train, nclass = nclass, device = device)
        else:
            train_loss, train_met = train_one_epoch(i, net, criterion, optimizer, loader_train, label_type, nclass = nclass, device = device)
        valid_loss, valid_met = valid_one_epoch(net, criterion, loader_valid, label_type, nclass = nclass, device = device)
            
        if scheduler:
            if get_name_of_scheduler(scheduler) == 'ReduceLROnPlateau':
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        if valid_loss < start_loss:
            save_checkpoint({'state_dict': net.state_dict(),
                             'optimizer': optimizer.state_dict()}, f'{checkpoint_dir}/net_checkpoint_{i}.pth.tar')
            start_loss = valid_loss

        log_losses(writer, train_loss, 'loss/train', i)
        log_losses(writer, valid_loss, 'loss/valid', i)
        log_losses(writer, train_met, met_name + '/train', i)
        log_losses(writer, valid_met, met_name + '/valid', i)

        writer.flush()
    writer.close()



def predict_gen(data_path, net, label_type, batch_size, nevents, seglabel_name = 'segclass', feature_name = ['energy'], device = 'cuda', num_workers = 1):
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

    gen    = DataGen(data_path, label_type, nevents = nevents, seglabel_name = seglabel_name, feature_name = feature_name)
    loader = torch.utils.data.DataLoader(gen,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         num_workers = num_workers,
                                         collate_fn = collatefn,
                                         drop_last = False,
                                         pin_memory = False,
                                         worker_init_fn = worker_init_fn)

    net.eval()
    softmax = torch.nn.Softmax(dim = 1)
    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.to(device), label.to(device)
            output = net((coord, ener, batch_size))
            y_pred = softmax(output).cpu().detach().numpy()

            if label_type == LabelType.Classification:
                out_dict = dict(
                    label       = label.cpu().detach().numpy(),
                    dataset_id  = event,
                    predictions = y_pred)

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
