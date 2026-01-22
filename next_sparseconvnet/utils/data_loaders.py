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
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
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

    ntot = len(binclass)
    nsignal = binclass.sum()
    nbackground = ntot-nsignal
    freq = np.array([nbackground, nsignal])
    if not effective_number:
        freq = ntot / freq
        return freq / freq.mean()
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


def transform_input(hits, bin_max, max_shift=5, z_transform=False, inplace=True):
    """
    hits: numpy structured ndarray or pandas DataFrame with columns ['xbin','ybin','zbin','energy']
    bin_max: list/array [max_x, max_y, max_z]
    max_shift: max number of voxels to randomly shift the event (default 0 = no shift)
    z_transform: whether Z flips/rotations are allowed
    inplace: whether to modify hits in-place (default True)
    """
    
    bin_names = ['xbin', 'ybin', 'zbin']
    
    if not inplace:
        hits = hits.copy()
    
    # -----------------------------
    # 1. AXIS FLIPS
    # -----------------------------
    for i, (name, max_val) in enumerate(zip(bin_names, bin_max)):
        if i == 2 and not z_transform:
            continue  # skip Z transformations if not allowed
        if np.random.randint(2) == 1:
            hits[name] = max_val - hits[name]

    # -----------------------------
    # 1b. RANDOM SHIFTS
    # -----------------------------
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift+1, size=3)
        # disable Z shift if not allowed
        if not z_transform:
            shift[2] = 0
        # apply shift safely (ensure inside detector)
        new_coords_min = [hits[n].min() + s for n, s in zip(bin_names, shift)]
        new_coords_max = [hits[n].max() + s for n, s in zip(bin_names, shift)]
        if all(0 <= new_min and new_max <= bmax for new_min, new_max, bmax in zip(new_coords_min, new_coords_max, bin_max)):
            for n, s in zip(bin_names, shift):
                hits[n] += s

    # -----------------------------
    # 2. ROTATION (swap + mirror)
    # -----------------------------
    do_rotate = np.random.randint(2) == 1
    if do_rotate:
        # generate all axis pairs
        pairs = list(itertools.permutations([0,1,2],2))
        if not z_transform:
            pairs = [(i,j) for i,j in pairs if i != 2 and j != 2]
        
        # filter valid rotations
        def possible_rotations(pair):
            x1, x2 = pair
            return ((hits[bin_names[x1]].max()-hits[bin_names[x1]].min() <= bin_max[x2]) and
                    (hits[bin_names[x2]].max()-hits[bin_names[x2]].min() <= bin_max[x1]))
        
        valid_pairs = list(filter(possible_rotations, pairs))
        if valid_pairs:
            x1, x2 = valid_pairs[np.random.randint(len(valid_pairs))]
            n1, n2 = bin_names[x1], bin_names[x2]

            # safe swap (works for pandas or numpy structured array)
            tmp = hits[n1].copy()
            hits[n1] = hits[n2].copy()
            hits[n2] = tmp

            # mirror second axis
            hits[n2] = bin_max[x2] - hits[n2]

            # fix overflow/underflow
            overflow_n1 = hits[n1].max() - bin_max[x1]
            if overflow_n1 > 0:
                hits[n1] -= overflow_n1
            underflow_n2 = hits[n2].min()
            if underflow_n2 < 0:
                hits[n2] -= underflow_n2

    if not inplace:
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


def load_resnet_checkpoint_safely(model, checkpoint_path, device="cpu"):
    """
    Loads a checkpoint into `model`.

    1) Tries direct load (same architecture)
    2) If that fails, attempts automatic remapping
    3) Verifies no missing / unexpected keys remain

    Raises RuntimeError if loading is unsafe.
    """

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and wrapped checkpoints
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded directly (matching architecture)")
        return model
    except RuntimeError as e:
        print("Direct load failed, attempting remapping...")
        print(e)

    remapped_sd = {}
    used_keys = set()

    for k, v in state_dict.items():

        # Feature extractor
        if k.startswith((
            "convBN", "basic", "downsample", "bottom",
            "inp", "max", "sparse"
        )):
            new_k = "feature_extractor." + k
            remapped_sd[new_k] = v
            used_keys.add(k)

        # Label classifier
        elif k.startswith("linear1"):
            new_k = "label_classifier.0" + k[len("linear1"):]
            remapped_sd[new_k] = v
            used_keys.add(k)

        elif k.startswith("linear2"):
            new_k = "label_classifier.2" + k[len("linear2"):]
            remapped_sd[new_k] = v
            used_keys.add(k)

    unused = set(state_dict.keys()) - used_keys
    if unused:
        print("Unused checkpoint keys:")
        for k in sorted(unused):
            print("   ", k)

    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)

    if missing or unexpected:
        print("\n Unsafe checkpoint load")
        print("Missing keys:")
        for k in missing:
            print("  ", k)
        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)
        raise RuntimeError("Checkpoint remapping incomplete")

    print("Checkpoint loaded via remapping")
    return model

