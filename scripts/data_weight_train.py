#!/usr/bin/env python
"""
This is a script to add weighted trainings after a ResNet is trained.
"""


import os
import sys
import copy
import torch
import numpy  as np
import pandas as pd
import tables as tb
from argparse     import ArgumentParser
from argparse     import Namespace

import torch.multiprocessing as mp
from next_sparseconvnet.utils.data_loaders     import LabelType
from next_sparseconvnet.networks.architectures import NetArchitecture
from next_sparseconvnet.networks.architectures import ResNet

from next_sparseconvnet.utils.train_utils      import save_checkpoint

from time import time
from next_sparseconvnet.utils.data_loaders import LabelType
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from next_sparseconvnet.utils.weight_train_utils import load_data, train_domain, valid_domain, train_label, valid_label
from next_sparseconvnet.utils.train_utils import get_name_of_scheduler, log_losses


def is_valid_action(parser, arg):
    if not arg in ['train', 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg

def is_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def get_params(confname):
    full_file_name = os.path.expandvars(confname)
    parameters = {}

    builtins = __builtins__.__dict__.copy()

    builtins['LabelType']       = LabelType
    builtins['NetArchitecture'] = NetArchitecture

    with open(full_file_name, 'r') as config_file:
        exec(config_file.read(), {'__builtins__':builtins}, parameters)
    return Namespace(**parameters)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description="parameters for models")
    parser.add_argument("-conf", dest = "confname", required=True,
                        help = "input file with parameters", metavar="FILE",
                        type = lambda x: is_file(parser, x))
    parser.add_argument("-a", dest = "action" , required = True,
                        help = "action to do for NN",
                        type = lambda x : is_valid_action(parser, x))
    args     = parser.parse_args()
    confname = args.confname
    action   = args.action
    parameters = get_params(confname)




    #### RESNET #####
    # CONSTRUCT RESNET
    net = ResNet(parameters.spatial_size,
                    parameters.init_conv_nplanes,
                    parameters.init_conv_kernel,
                    parameters.kernel_sizes,
                    parameters.stride_sizes,
                    parameters.basic_num,
                    start_planes = parameters.start_planes,
                    momentum = parameters.momentum,
                    nlinear = parameters.nlinear)
        
    net = net.to(device)
    print('ResNet constructed')
    # LOAD SAVED RESNET
    dct_weights = torch.load(parameters.resnet_saved_weights, map_location=torch.device(device))['state_dict']
    net.load_state_dict(dct_weights, strict=False)
    print('weights loaded from trained ResNet')
    # FREEZE RESNET FEATURE EXTRACTOR
    for p in net.feature_extractor.parameters():
        p.requires_grad = False
    print('ResNet feature extractor weights freezed')

    #### DOMAIN CLF ####
    # CONSTRUCT DOMAIN CLF
    domain_clf = torch.nn.Sequential(
                            torch.nn.Linear(net.feature_extractor.out_dim, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 1))
    domain_clf = domain_clf.to(device)
    print('Simple domain classifier constructed')

    # GET VARIABLES
    nepoch = parameters.nepoch
    label_type = parameters.labeltype

    if device == 'cuda': pin_mem = True
    else: pin_mem = False

    met_name = 'iou' if label_type == LabelType.Segmentation else 'acc'

    # LOAD DATA
    loader_train_mc, loader_valid_mc, loader_train_dt, loader_valid_dt = load_data(parameters.train_file, 
                                                                                   parameters.train_data_domain_path, 
                                                                                   label_type, 
                                                                                   parameters.nevents_train, 
                                                                                   parameters.augmentation, 
                                                                                   parameters.seglabel_name, 
                                                                                   parameters.feature_name, 
                                                                                   parameters.val_split, 
                                                                                   parameters.train_batch, 
                                                                                   parameters.num_workers, 
                                                                                   pin_mem)

    # DIVERSION: TRAIN DOMAIN OR TRAIN LABEL
    if action == 'train_domain':
        # PICK CRITERION AND OPT, WITH ONLY DOMAIN CLF PARAMETERS
        criterion_domain = torch.nn.BCEWithLogitsLoss()

        optimizer_domain = torch.optim.Adam(filter(lambda p: p.requires_grad, domain_clf.parameters()), # WE TRAIN ONLY DOMAIN CLF HERE
                                     lr = parameters.lr,
                                     betas = parameters.betas,
                                     eps = parameters.eps,
                                     weight_decay = parameters.weight_decay)

        scheduler = None
        # Scheduler for the Learning Rate
        if parameters.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_domain, 
                                                                    factor = parameters.reduce_lr_factor, 
                                                                    patience = parameters.patience, 
                                                                    min_lr = parameters.min_lr)

        t_start = time()
        start_loss = np.inf
        writer = SummaryWriter(parameters.tensorboard_domain_dir)
        for i in range(nepoch):
            train_loss, train_met_mc, train_met_dt = train_domain(net, domain_clf, loader_train_mc, loader_train_dt, optimizer_domain, criterion_domain, device)
            valid_loss, valid_met_mc, valid_met_dt = valid_domain(net, domain_clf, loader_valid_mc, loader_valid_dt,                   criterion_domain, device)

            if scheduler:
                if get_name_of_scheduler(scheduler) == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

            print(f"\nEpoch {i} |  TRAINING  | Loss ={train_loss:.4f} | Domain MC Acc= {train_met_mc:.4f} | Domain DT Acc= {train_met_dt:.4f} | Time={(time()-t_start)/60:.2f} min")
            sys.stdout.flush()
            print(f"\nEpoch {i} | VALIDATION | Loss ={valid_loss:.4f} | Domain MC Acc= {valid_met_mc:.4f} | Domain DT Acc= {valid_met_dt:.4f}")

            if valid_loss < start_loss:
                save_checkpoint({'state_dict': domain_clf.state_dict(), # save the domain classifier
                                'optimizer': optimizer_domain.state_dict()}, f'{parameters.checkpoint_domain_dir}/net_domain_checkpoint_{i}.pth.tar')
                start_loss = valid_loss

            log_losses(writer, train_loss, 'loss/train', i)
            log_losses(writer, valid_loss, 'loss/valid', i)
            log_losses(writer, train_met_mc, met_name + '/train_mc', i)
            log_losses(writer, valid_met_mc, met_name + '/valid_mc', i)
            log_losses(writer, train_met_dt, met_name + '/train_dt', i)
            log_losses(writer, valid_met_dt, met_name + '/valid_dt', i)





    if action == 'train_label':

        dct_dom_weights = torch.load(parameters.domain_clf_saved_weights, map_location=torch.device(device))['state_dict']
        domain_clf.load_state_dict(dct_dom_weights, strict=False)
        
        for p in domain_clf.parameters():
            p.requires_grad = False
        print('weights loaded and freezed from trained domain classifier')

        criterion_label = torch.nn.CrossEntropyLoss(weight = parameters.weight_loss, reduction = 'none')

        optimizer_label = torch.optim.Adam(filter(lambda p: p.requires_grad, net.label_classifier.parameters()), # PASS ONLY LABEL HEAD PARAMETERS
                                     lr = parameters.lr,
                                     betas = parameters.betas,
                                     eps = parameters.eps,
                                     weight_decay = parameters.weight_decay)
        
        scheduler = None
        # Scheduler for the Learning Rate
        if parameters.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, 
                                                                    factor = parameters.reduce_lr_factor, 
                                                                    patience = parameters.patience, 
                                                                    min_lr = parameters.min_lr)
        t_start = time()
        start_loss = np.inf
        writer = SummaryWriter(parameters.tensorboard_weighted_dir)
        for i in range(nepoch):
            train_wloss, train_loss, train_met, train_weights = train_label(net, domain_clf, loader_train_mc, optimizer_label, criterion_label, device, w_min=0.1, w_max=10.0)
            valid_wloss, valid_loss, valid_met, valid_weights = valid_label(net, domain_clf, loader_valid_mc, criterion_label, device)
            # Two losses, for the net to learn we want to use the training weighted loss, but
            # for the validation, we want the model to still be stable, so we use regular 
            # validation loss instead

            if scheduler:
                if get_name_of_scheduler(scheduler) == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

            print(f"\nEpoch {i} |  TRAINING  | Weighted Loss ={train_wloss:.4f} | Normal Loss ={train_loss:.4f} | Acc= {train_met:.4f} | Time={(time()-t_start)/60:.2f} min")
            sys.stdout.flush()
            print(f"\nEpoch {i} | VALIDATION | Weighted Loss ={valid_wloss:.4f} | Normal Loss ={valid_loss:.4f} | Acc= {valid_met:.4f}")

            if valid_loss < start_loss:
                save_checkpoint({'state_dict': net.state_dict(), # save resnet with the updated label classifier part
                                'optimizer': optimizer_label.state_dict()}, f'{parameters.checkpoint_weighted_dir}/net_weighted_checkpoint_{i}.pth.tar')
                start_loss = valid_loss

            writer.add_histogram("weights/train", torch.cat(train_weights), i)
            writer.add_histogram("weights/valid", torch.cat(valid_weights), i)
            log_losses(writer, train_wloss, 'wloss/train', i)
            log_losses(writer, valid_wloss, 'wloss/valid', i)
            log_losses(writer, train_loss, 'loss/train', i)
            log_losses(writer, valid_loss, 'loss/valid', i)
            log_losses(writer, train_met, met_name + '/train', i)
            log_losses(writer, valid_met, met_name + '/valid', i)


