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

import sys

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
    def __init__(self, filename, label_type, nevents=None, augmentation = False, seglabel_name = 'segclass', feature_name = ['energy']):
        """ This class yields events from pregenerated MC file.
        Parameters:
            filename : str; filename to read
            table_name : str; name of the table to read
                         currently available BinClassHits and SegClassHits
        """
        self.filename   = filename
        if not isinstance(label_type, LabelType):
            raise ValueError(f'{label_type} not recognized!')
        self.label_type    = label_type
        self.seglabel_name = seglabel_name
        self.feature_name  = feature_name
        self.events        = read_events_info(filename, nevents)
        self.bininfo       = load_dst(filename, 'DATASET', 'BinsInfo')
        self.h5in = None
        self.augmentation = augmentation
        self.maxbins = [self.bininfo['nbins_x'][0], self.bininfo['nbins_y'][0], self.bininfo['nbins_z'][0]]
        self.eventidx = 'stridx' in self.events.columns and 'endidx' in self.events.columns

    def initialize_file(self): #this opens the table once we call the initialization
        if self.h5in is None:
            self.h5in = tb.open_file(self.filename, 'r')

    def __getitem__(self, idx):
        # self.initialize_file() # Make sure it's open

        event_info = self.events.iloc[idx]
        idx_       = event_info['dataset_id']

        if self.eventidx:
            start_row = event_info['stridx']
            end_row   = event_info['endidx']
            hits      = self.h5in.root.DATASET.Voxels[start_row:end_row]
        else:
            hits = self.h5in.root.DATASET.Voxels.read_where(f'dataset_id=={idx_}')

        if self.augmentation:
            transform_input(hits, self.maxbins)

        if self.label_type == LabelType.Classification:
            label = np.unique(hits['binclass'])

        elif self.label_type == LabelType.Segmentation:
            label = hits[self.seglabel_name]

        features = np.vstack([hits[name] for name in self.feature_name]).T
        return hits['xbin'], hits['ybin'], hits['zbin'], features, label, idx_

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
    energs = torch.tensor(np.concatenate(energs, axis=0), dtype = torch.float)
    labels = torch.tensor(np.concatenate(labels, axis=0), dtype = torch.long)

    return  coords, energs, labels, events

def worker_init_fn(worker_id): # Required by PyTorch
    print(f"Initializing worker {worker_id}")
    sys.stdout.flush()
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # This is your DataGen instance
    dataset.initialize_file()      # Pre-open file in the worker

def weights_loss_segmentation(fname, nevents, effective_number=False, beta=0.9999, seglabel_name = 'segclass'):
    if isinstance(nevents, (list, tuple)):
        nevents = nevents[1] - nevents[0]

    with tb.open_file(fname, 'r') as h5in:
        dataset_id = h5in.root.DATASET.Voxels.read_where(f'dataset_id<{nevents}', field='dataset_id')
        seglabel   = h5in.root.DATASET.Voxels.read_where(f'dataset_id<{nevents}', field=seglabel_name)

    df = pd.DataFrame({'dataset_id':dataset_id, 'seglabel':seglabel})
    nclass = max(df.seglabel)+1
    mean_freq = np.bincount(df.seglabel, minlength=nclass)
    # this is applying per event mean
    # df = df.groupby('dataset_id').seglabel.apply(lambda x:np.bincount(x, minlength=nclass)/len(x))
    # mean_freq = df.mean()
    if not effective_number:
        inverse_freq = 1./mean_freq
        return inverse_freq/sum(inverse_freq)
    else:
        effective_num = 1.0 - np.power(beta, mean_freq)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * nclass
        return weights

def weights_loss_classification(fname, effective_number=False, beta=0.9999):
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

def weights_loss(fname, nevents, label_type, effective_number=False, seglabel_name = 'segclass'):
    if label_type==LabelType.Segmentation:
        return weights_loss_segmentation(fname, nevents, effective_number=effective_number, seglabel_name=seglabel_name)
    elif label_type == LabelType.Classification:
        return weights_loss_classification(fname, effective_number=effective_number)


def transform_input(hits, bin_max, max_shift = 5, z_transform=False):
    """
    hits: pandas DataFrame with columns ['xbin','ybin','zbin','energy']
    bin_max: list/array [max_x, max_y, max_z]
    max_shift: max number of voxels that can be shifted
    z_transform: allow z flips / rotations
    """

    cols = ['xbin', 'ybin', 'zbin']

    # ============================================
    # 1. AXIS FLIPS + SHIFTS
    # ============================================
    do_flip = np.random.randint(2, size=3).astype(bool)
    shift = np.random.randint(-max_shift, max_shift+1, size=3)

    # Disable Z flip if needed
    if not z_transform:
        do_flip[2] = False
        shift[2] = 0

    coords = np.stack([hits[c] for c in cols], axis=1)
    # Compute new min/max of the translated event
    new_min = coords.min(axis = 0) + shift
    new_max = coords.max(axis = 0) + shift

    # Check detector boundaries
    inside = np.all(new_min >= 0) and np.all(new_max <= bin_max)

    # Apply flips (no warnings, DataFrame-safe)
    for i, col in enumerate(cols):
        if do_flip[i]:
            hits.loc[:, col] = bin_max[i] - hits[col]
        if (shift[i] != 0) & inside: #ensures event is still inside boundaries
            hits.loc[:, col] = hits[col] + shift[i]

    
    # ============================================
    # 2. ROTATION (single swap + flip)
    # ============================================
    if np.random.randint(2) == 1:

        # Possible axis permutations
        all_pairs = list(itertools.permutations([0,1,2],2))

        # Remove Z involvement if disallowed
        if not z_transform:
            all_pairs = [(i,j) for (i,j) in all_pairs if i != 2 and j != 2]

        # Evaluate geometric constraints
        rot_candidates = []
        for i, j in all_pairs:
            rng_i = hits[cols[i]].max() - hits[cols[i]].min()
            rng_j = hits[cols[j]].max() - hits[cols[j]].min()
            if rng_i <= bin_max[j] and rng_j <= bin_max[i]:
                rot_candidates.append((i,j))
        # Apply one valid rotation
        if rot_candidates:
            i, j = rot_candidates[np.random.randint(len(rot_candidates))]
            col_i, col_j = cols[i], cols[j]

            # -----------------------------------------
            # Vectorized swap 
            # -----------------------------------------
            tmp = hits[col_i].copy()
            hits.loc[:, col_i] = hits[col_j]
            hits.loc[:, col_j] = tmp

            # -----------------------------------------
            # Mirror j-axis
            # -----------------------------------------
            hits.loc[:, col_j] = bin_max[j] - hits[col_j]

            # -----------------------------------------
            # Fix overflow (this was done mainly for when we flipped also in Z, because bin_max for Z was way bigger, now shouldnt be a problem)
            # -----------------------------------------
            overflow_i = hits[col_i].max() - bin_max[i]
            if overflow_i > 0:
                hits.loc[:, col_i] = hits[col_i] - overflow_i

            # -----------------------------------------
            # Fix underflow
            # -----------------------------------------
            underflow_j = hits[col_j].min()
            if underflow_j < 0:
                hits.loc[:, col_j] = hits[col_j] - underflow_j

    return hits


# def transform_input(hits, bin_max, inplace=True):
#     bin_names = ['xbin', 'ybin', 'zbin']

#     if not inplace:
#         hits = hits.copy()
#     #mirroring in x, y and z
#     for n, m in zip(bin_names, bin_max):
#         if np.random.randint(2) == 1:
#             hits[n] = m - hits[n]

#     def possible_rotations(element):
#         x1, x2 = element
#         return ((hits[bin_names[x1]].max()-hits[bin_names[x1]].min()<=bin_max[x2]) and
#                 (hits[bin_names[x2]].max()-hits[bin_names[x2]].min()<=bin_max[x1]))
#     if np.random.randint(2) == 1:
#         rotations_list = list(filter(possible_rotations, itertools.permutations([0, 1, 2], 2)))
#         x1, x2 = rotations_list[np.random.randint(len(rotations_list))]
#         names   = [bin_names[x1], bin_names[x2]]
#         maxbin  = [bin_max[x1], bin_max[x2]]
#         #rotate hits
#         hits[names] = hits[names[::-1]]
#         #flip second axis; this can make bins negative cause maxbin[x1]!=maxbin[x2]
#         hits[names[1]] = maxbin[1]-hits[names[1]]
#         #substract (max_index - maxbin) if it is positive for x1
#         hits[names[0]]-= max(hits[names[0]].max()-maxbin[0], 0)
#         #substract min of 0 and min(hits[names[1]]) to ensure bins are positive for x2
#         hits[names[1]]-= min(hits[names[1]].min(), 0)
#     if not inplace:
#         return hits

# def read_event(fname, datid, table='Voxels', group='DATASET', df=True):
#     with tb.open_file(fname) as h5in:
#         hits = h5in.root[group][table].read_where('dataset_id==datid')
#         if df:
#             return pd.DataFrame.from_records(hits)
#         return hits

# def read_events_info(filename, nevents):
#     events = load_dst(filename, 'DATASET', 'EventsInfo')
#     if nevents is not None:
#         if nevents>=len(events):
#             warnings.warn(UserWarning(f'length of dataset smaller than {nevents}, using full dataset'))
#         else:
#             events = events.iloc[:nevents]

#     events[events.binclass==2]=1 #WTF
#     return events

def read_events_info(filename, nevents=None):
    events = load_dst(filename, 'DATASET', 'EventsInfo')

    if nevents is not None:
        if isinstance(nevents, int):
            if nevents >= len(events):
                warnings.warn(UserWarning(f'Length of dataset smaller than {nevents}, using full dataset'))
            else:
                events = events.iloc[:nevents]

        elif isinstance(nevents, (tuple, list)) and len(nevents) == 2:
            start, end = nevents
            if start >= len(events):
                warnings.warn(UserWarning(f'Start index {start} beyond dataset length; returning empty DataFrame'))
                events = events.iloc[0:0]
            else:
                events = events.iloc[start:end]

        else:
            raise ValueError("`nevents` must be None, an integer, or a (start, end) tuple/list.")

    events.loc[events.binclass == 2, 'binclass'] = 1 #wtf

    return events

# def read_events(fname, nevents, table='Voxels', group='DATASET', df=True):
#     with tb.open_file(fname) as h5in:
#         if nevents is not None:
#             hits = h5in.root[group][table].read_where('dataset_id<nevents')
#         else:
#             hits = h5in.root[group][table].read()

#         return pd.DataFrame.from_records(hits)

def read_events(fname, nevents=None, table='Voxels', group='DATASET', df=True):
    with tb.open_file(fname) as h5in:
        table_data = h5in.root[group][table]

        if nevents is None:
            hits = table_data.read()

        elif isinstance(nevents, int):
            hits = table_data.read_where(f'dataset_id < {nevents}')

        elif isinstance(nevents, (tuple, list)) and len(nevents) == 2:
            start, end = nevents
            hits = table_data.read_where(f'({start} <= dataset_id) & (dataset_id < {end})')

        else:
            raise ValueError("nevents must be None, an integer, or a (start, end) tuple/list")

        return pd.DataFrame.from_records(hits) if df else hits
