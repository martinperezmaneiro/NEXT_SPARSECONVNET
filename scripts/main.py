#!/usr/bin/env python
"""
This is a main script to perform training or prediction of network on provided data.
To be called as main.py -conf conf_filename -a train
"""


import os
import copy
import torch
from argparse     import ArgumentParser
from argparse     import Namespace

import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.io.dst_io import df_writer
# from invisible_cities.cities.components import index_tables

from next_sparseconvnet.utils.data_io          import index_tables

from next_sparseconvnet.utils.data_loaders     import LabelType
from next_sparseconvnet.utils.data_loaders     import weights_loss
from next_sparseconvnet.networks.architectures import NetArchitecture
from next_sparseconvnet.networks.architectures import UNet
from next_sparseconvnet.networks.architectures import ResNet

from next_sparseconvnet.utils.train_utils      import train_net
from next_sparseconvnet.utils.train_utils      import predict_gen
from next_sparseconvnet.utils.focal_loss       import FocalLoss

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

    if parameters.netarch == NetArchitecture.UNet:
        net = UNet(parameters.spatial_size,
                   parameters.init_conv_nplanes,
                   parameters.init_conv_kernel,
                   parameters.kernel_sizes,
                   parameters.stride_sizes,
                   parameters.basic_num,
                   nclasses = parameters.nclasses,
                   momentum = parameters.momentum)
        net = net.cuda()
    elif parameters.netarch == NetArchitecture.ResNet:
        net = ResNet(parameters.spatial_size,
                     parameters.init_conv_nplanes,
                     parameters.init_conv_kernel,
                     parameters.kernel_sizes,
                     parameters.stride_sizes,
                     parameters.basic_num,
                     momentum = parameters.momentum,
                     nlinear = parameters.nlinear)
        net = net.cuda()

    print('net constructed')

    if parameters.saved_weights:
        dct_weights = torch.load(parameters.saved_weights)['state_dict']
        net.load_state_dict(dct_weights, strict=False)
        print('weights loaded')

        if parameters.freeze_weights:
            for name, param in net.named_parameters():
                if name in dct_weights:
                    param.requires_grad = False


    if action == 'train':

        if parameters.LossType == 'CrossEntropyLoss':
            isfocal = False
        elif parameters.LossType == 'FocalLoss':
            isfocal = True
        else:
            KeyError('Unknown loss type')

        if parameters.weight_loss is True: #calculate mean using first 10000 events from file
            print('Calculating weights')
            weights = torch.Tensor(weights_loss(parameters.train_file, 10000, parameters.labeltype, effective_number=False, seglabel_name=parameters.seglabel_name)).cuda()
            print('Weights are', weights)
        elif isinstance(parameters.weight_loss, list):
            weights = torch.Tensor(parameters.weight_loss).cuda()
            print('Read weights from config')
        else:
            weights = None

        if isfocal:
            criterion = FocalLoss(alpha=weights, gamma=2.)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight = weights)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr = parameters.lr,
                                     betas = parameters.betas,
                                     eps = parameters.eps,
                                     weight_decay = parameters.weight_decay)

        train_net(nepoch = parameters.nepoch,
                  train_data_path = parameters.train_file,
                  valid_data_path = parameters.valid_file,
                  train_batch_size = parameters.train_batch,
                  valid_batch_size = parameters.valid_batch,
                  net = net,
                  label_type = parameters.labeltype,
                  criterion = criterion,
                  optimizer = optimizer,
                  checkpoint_dir = parameters.checkpoint_dir,
                  tensorboard_dir = parameters.tensorboard_dir,
                  num_workers = parameters.num_workers,
                  nevents_train = parameters.nevents_train,
                  nevents_valid = parameters.nevents_valid,
                  augmentation  = parameters.augmentation, 
                  seglabel_name = parameters.seglabel_name)

    if action == 'predict':
        gen = predict_gen(data_path = parameters.predict_file,
                          label_type = parameters.labeltype,
                          net = net,
                          batch_size = parameters.predict_batch,
                          nevents = parameters.nevents_predict, 
                          seglabel_name = parameters.seglabel_name)
        coorname = ['xbin', 'ybin', 'zbin']
        output_name = parameters.out_file

        if parameters.labeltype == LabelType.Segmentation:
            tname = 'VoxelsPred'
        else:
            tname = 'EventPred'
        with tb.open_file(output_name, 'w') as h5out:
            for dct in gen:
                if 'coords' in dct:
                    coords = dct.pop('coords')
                    #unpack coords and add them to dictionary
                    dct.update({coorname[i]:coords[:, i] for i in range(3)})
                predictions = dct.pop('predictions')
                #unpack predictions and add them back to dictionary
                dct.update({f'class_{i}':predictions[:, i] for i in range(predictions.shape[1])})

                #create pandas dataframe and save to output file
                df = pd.DataFrame(dct)
                df_writer(h5out, df, 'DATASET', tname, columns_to_index=['dataset_id'])

        index_tables(output_name)
