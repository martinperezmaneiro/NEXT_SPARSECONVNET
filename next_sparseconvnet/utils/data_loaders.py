import tables as tb
import numpy  as np
import pandas as pd
import torch
import warnings
import itertools
from enum import auto

from invisible_cities.io   .dst_io  import load_dst
from invisible_cities.types.ic_types import AutoNameEnumBase

from . data_io import get_3d_input


class LabelType(AutoNameEnumBase):
    Classification = auto()
    Segmentation   = auto()


class DataGen_classification(torch.utils.data.Dataset):
    def __init__(self, labels, binsX, binsY, binsZ):
        self.binsX  = binsX
        self.binsY  = binsY
        self.binsZ  = binsZ
        self.labels = labels

    def __getitem__(self, idx):
        filename = self.labels.iloc[idx].filename
        event    = self.labels.iloc[idx].event
        label    = self.labels.iloc[idx].label
        x, y, z, ener = get_3d_input(filename, event, self.binsX, self.binsY, self.binsZ)
        return x, y, z, ener, [label], event #tener eventid puede ser util



class DataGen(torch.utils.data.Dataset):
    def __init__(self, filename, label_type, voxel_name = 'Voxels', nevents=None, augmentation = False):
        """ This class yields events from pregenerated MC file.
        Parameters:
            filename : str; filename to read
            table_name : str; name of the table to read
                         currently available BinClassHits and SegClassHits
        """
        self.filename   = filename
        if not isinstance(label_type, LabelType):
            raise ValueError(f'{label_type} not recognized!')
        self.label_type = label_type
        self.voxel_name = voxel_name
        self.events     = read_events_info(filename, nevents)
        self.bininfo    = load_dst(filename, 'DATASET', 'BinsInfo')
        self.h5in = None
        self.augmentation = augmentation
        self.maxbins = [self.bininfo['nbins_x'][0], self.bininfo['nbins_y'][0], self.bininfo['nbins_z'][0]]

    def __getitem__(self, idx):
        idx_ = self.events.iloc[idx].dataset_id
        if self.h5in is None:#this opens a table once getitem gets called
            self.h5in = tb.open_file(self.filename, 'r')
        hits  = self.h5in.root.DATASET[self.voxel_name].read_where('dataset_id==idx_')
        if self.augmentation:
            transform_input(hits, self.maxbins)
        if self.label_type == LabelType.Classification:
            label = np.unique(hits['binclass'])
        elif self.label_type == LabelType.Segmentation:
            label = hits['segclass']
        return hits['xbin'], hits['ybin'], hits['zbin'], hits['energy'], label, idx_

    def __len__(self):
        return len(self.events)
    def __del__(self):
        if self.h5in is not None:
            self.h5in.close()

def collatefn(batch):
    coords = []
    energs = []
    labels = []
    events = torch.zeros(len(batch)).int()
    for bid, data in enumerate(batch):
        x, y, z, E, lab, event = data
        batchid = np.ones_like(x)*bid
        coords.append(np.concatenate([x[:, None], y[:, None], z[:, None], batchid[:, None]], axis=1))
        energs.append(E)
        labels.append(lab)
        events[bid] = event

    coords = torch.tensor(np.concatenate(coords, axis=0), dtype = torch.long)
    energs = torch.tensor(np.concatenate(energs, axis=0), dtype = torch.float).unsqueeze(-1)
    labels = torch.tensor(np.concatenate(labels, axis=0), dtype = torch.long)

    return  coords, energs, labels, events


def weights_loss_segmentation(fname, nevents, effective_number=False, beta=0.9999):
    with tb.open_file(fname, 'r') as h5in:
        dataset_id = h5in.root.DATASET.Voxels.read_where('dataset_id<nevents', field='dataset_id')
        segclass   = h5in.root.DATASET.Voxels.read_where('dataset_id<nevents', field='segclass')

    df = pd.DataFrame({'dataset_id':dataset_id, 'segclass':segclass})
    nclass = max(df.segclass)+1
    mean_freq = np.bincount(df.segclass, minlength=nclass)
    # this is applying per event mean
    # df = df.groupby('dataset_id').segclass.apply(lambda x:np.bincount(x, minlength=nclass)/len(x))
    # mean_freq = df.mean()
    if not effective_number:
        inverse_freq = 1./mean_freq
        return inverse_freq/sum(inverse_freq)
    else:
        effective_num = 1.0 - np.power(beta, mean_freq)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * nclass
        return weights

def weights_loss_classification(fname, nevents, effective_number=False, beta=0.9999):
    with tb.open_file(fname, 'r') as h5in:
        binclass   = h5in.root.DATASET.EventsInfo.cols.binclass[:]

    nsignal = binclass.sum()
    nbackground = len(binclass)-nsignal
    freq = np.array([nbackground, nsignal])
    if not effective_number:
        return freq/sum(freq)
    else:
        effective_num = 1.0 - np.power(beta, freq)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * 2
        return weights

def weights_loss(fname, nevents, label_type, effective_number=False):
    if label_type==LabelType.Segmentation:
        return weights_loss_segmentation(fname, nevents, effective_number=effective_number)
    elif label_type == LabelType.Classification:
        return weights_loss_classification(fname, nevents, effective_number=effective_number)

def transform_input(hits, bin_max, inplace=True):
    bin_names = ['xbin', 'ybin', 'zbin']

    if not inplace:
        hits = hits.copy()
    #mirroring in x, y and z
    for n, m in zip(bin_names, bin_max):
        if np.random.randint(2) == 1:
            hits[n] = m - hits[n]

    def possible_rotations(element):
        x1, x2 = element
        return ((hits[bin_names[x1]].max()-hits[bin_names[x1]].min()<=bin_max[x2]) and
                (hits[bin_names[x2]].max()-hits[bin_names[x2]].min()<=bin_max[x1]))
    if np.random.randint(2) == 1:
        rotations_list = list(filter(possible_rotations, itertools.permutations([0, 1, 2], 2)))
        x1, x2 = rotations_list[np.random.randint(len(rotations_list))]
        names   = [bin_names[x1], bin_names[x2]]
        maxbin  = [bin_max[x1], bin_max[x2]]
        #rotate hits
        hits[names] = hits[names[::-1]]
        #flip second axis; this can make bins negative cause maxbin[x1]!=maxbin[x2]
        hits[names[1]] = maxbin[1]-hits[names[1]]
        #substract (max_index - maxbin) if it is positive for x1
        hits[names[0]]-= max(hits[names[0]].max()-maxbin[0], 0)
        #substract min of 0 and min(hits[names[1]]) to ensure bins are positive for x2
        hits[names[1]]-= min(hits[names[1]].min(), 0)
    if not inplace:
        return hits

def read_event(fname, datid, table='Voxels', group='DATASET', df=True):
    with tb.open_file(fname) as h5in:
        hits = h5in.root[group][table].read_where('dataset_id==datid')
        if df:
            return pd.DataFrame.from_records(hits)
        return hits

def read_events_info(filename, nevents):
    events = load_dst(filename, 'DATASET', 'EventsInfo')
    if nevents is not None:
        if nevents>=len(events):
            warnings.warn(UserWarning(f'length of dataset smaller than {nevents}, using full dataset'))
        else:
            events = events.iloc[:nevents]

    events[events.binclass==2]=1
    return events

def read_events(fname, nevents, table='Voxels', group='DATASET', df=True):
    with tb.open_file(fname) as h5in:
        if nevents is not None:
            hits = h5in.root[group][table].read_where('dataset_id<nevents')
        else:
            hits = h5in.root[group][table].read()

        return pd.DataFrame.from_records(hits)
