'''
Script to apply a mock version of Paolina (PAO): 
* On labelled events
* With an improvement of radii scan for blob search
* Using a threshold from the deconvolution NN to compare both

'''
import sys
import argparse

import numpy as np
import pandas as pd
import tables as tb

def get_args():
    parser = argparse.ArgumentParser(
        description="Apply PAO to a labelled file that also has a deconvolution threshold")
    parser.add_argument(
        "-n", "-nfile", "--nfile",
        dest="nfile",
        type=int,
        required=True,
        help="Number of the labelled file")
    parser.add_argument(
        "-d", "-datatype", "--datatype",
        dest="dt",
        type=str,
        default="MC",
        required=True,
        help="Type of data"
    )
    parser.add_argument(
        "-e", "-events", "--events",
        dest="nevents_per_file",
        type=int,
        default=0,
        required=True,
        help="Number of events per file (for run data)"
    )
    parser.add_argument(
        "-r", "-run_n", "--run_n",
        dest="run_n",
        type=str,
        default='15607',
        required=True,
        help="Run number"
        )
    return parser.parse_args()

args = get_args()
nfile = args.nfile
dt = args.dt
run_n = args.run_n
nevents_per_file = args.nevents_per_file

# VARIABLES
rad = 21
contiguity = np.sqrt(3)
pos_to_coord = True

# ------------------------PATHS------------------------------------

# MC
# File that contains voxels with decolabel score (class_1)
test_file_pred = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_calib/4bar/trains/soph_deco/train_D/pred_file.h5'

# Thresholded file, contains the best threshold applied for each event
test_file_thr = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_calib/4bar/trains/soph_deco/train_D/pred_file_thr.h5'

# Original label file that contains the original labelled voxels and hits from the test dataset
# (not sure why for the voxel part I pick this instead of directly the ones from the pred file, maybe because of the energy?? idk right now)
labelfile = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_calib/4bar/208Tl/prod/PORT_1a/label/prod/sophronia_label_{n}_208Tl.h5'.format(n=nfile)

savefile = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_calib/4bar/208Tl/prod/PORT_1a/label/pao/pao_{n}_208Tl.h5'.format(n=nfile)

# DATA
datafile = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_calib/4bar/trains/soph_deco/train_D/dataset_{run_n}_DEP_deco.h5'.format(run_n = run_n)
savedatafile = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/HE_calib/4bar/trains/soph_deco/train_D/deconvolved_pao/pao/pao_{n}_{run_n}.h5'.format(n=nfile, run_n = run_n)


# ---------------------- PAO FUNCTIONS----------------------------------

import networkx as nx
from itertools import combinations

def connected_component_subgraphs(G):
    return (G.subgraph(c).copy() for c in nx.connected_components(G))

from scipy.spatial import KDTree

def make_track_graphs(event, contiguity, columns = ['xbin', 'ybin', 'zbin', 'energy', 'nhits'], spacing = (1, 1, 1)):
    '''
    Creates a graph using a KDTree for efficient distance computations.
    '''
    voxels = event[columns].values
    nodes = [(tuple(v[:3]), {'energy':v[-2], 'nhits':v[-1]}) for v in voxels]
    pos_nodes = [n for n, _ in nodes]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    
    # Use KDTree for fast neighbor search
    kdtree = KDTree(pos_nodes)
    pairs = kdtree.query_pairs(r=contiguity + sys.float_info.epsilon) # add epsilon to include the contiguity value itself
    
    # Add edges for nodes within contiguity
    for i, j in pairs:
        p1 = np.array(pos_nodes[i])
        p2 = np.array(pos_nodes[j])

        dist = np.linalg.norm((p1 - p2) * spacing)
        graph.add_edge(pos_nodes[i], pos_nodes[j], distance=dist)
    return tuple(connected_component_subgraphs(graph))

get_track_energy = lambda x: sum([x.nodes[n]['energy'] for n in x])


def shortest_paths(track_graph):
    """Compute shortest path lengths between all nodes in a weighted graph."""
    voxel_pos = lambda x: x[0]

    distances = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight='distance'))

    # sort the output so the result is reproducible
    distances = { v1 : {v2:d for v2, d in sorted(dmap.items(), key=voxel_pos)}
                  for v1, dmap in sorted(distances.items(), key=voxel_pos)}
    return distances


def find_extrema_and_length(distance):
    """Find the extrema and the length of a track, given its dictionary of distances."""
    if not distance:
        print('No voxels')
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.)
    first, last, max_distance = None, None, 0
    for (voxel1, dist_from_voxel_1_to), (voxel2, _) in combinations(distance.items(), 2):
        d = dist_from_voxel_1_to[voxel2]
        if d > max_distance:
            first, last, max_distance = voxel1, voxel2, d
    return first, last, max_distance

def find_extrema(track):
    """Find the pair of voxels separated by the greatest geometric
      distance along the track.
    """
    distances = shortest_paths(track)
    extremum_a, extremum_b, _ = find_extrema_and_length(distances)
    return extremum_a, extremum_b

def voxel_in_blob(track_graph, extreme, rad, spacing): # distances, 
    # Instead of using the djikstra distances I will be using a normal distance
    # In this way, this function can be used then with the barycenter too
    # Both distances are nor the same but similar, so I'll stick with the direct distance
    # dist_from_extreme = distances[extreme]
    diag = np.linalg.norm(spacing)

    blob_voxels = []
    for v in track_graph.nodes():
        # v_dist = dist_from_extreme[v]
        # I will be using simply the euclidean distance, it is better and can be reused afterwards.
        v_dist = np.linalg.norm((np.array(v) - np.array(extreme)) * np.array(spacing))
        # Adding the diagonal because we are using voxels, so the radius alone will not include "partial" voxels, por asi decirlo, y asi hacemos que se incluyan
        if v_dist < rad + diag:
            blob_voxels.append(v)
    return blob_voxels

def voxel_pos(t):
    return [np.array(v) for v in t]

def voxel_ener(t, voxels):
    return [t.nodes[v]['energy'] for v in voxels]

def barycenter(t, voxels):
    positions = voxel_pos(voxels)
    energies = voxel_ener(t, voxels)

    barycenter = np.average(positions, weights = energies, axis = 0)
    return barycenter

def vox_to_coord(vox_index, spacing, initial):
    return initial + vox_index * spacing

def get_track_info(event, contiguity =  np.sqrt(3), rad = 21, spacing = np.array([15.55, 15.55, 10]), initial = np.array([-500, -500, 0]), pos_to_coord = True, columns = ['xbin', 'ybin', 'zbin', 'energy', 'nhits']):
    dat_id = event.dataset_id.values[0]
    binclass = event.binclass.values[0]


    tracks = sorted(make_track_graphs(event, contiguity, spacing = spacing, columns = columns), key = get_track_energy, reverse=True)
    track_info = []
    for tID, t in enumerate(tracks):
        

        distances = shortest_paths(t)
        a, b, _ = find_extrema_and_length(distances)

        # Pick the voxels of each blob given an extreme and a radius
        va = voxel_in_blob(t, a, rad, spacing)
        vb = voxel_in_blob(t, b, rad, spacing)

        Eexta, Eextb = sum(voxel_ener(t, va)), sum(voxel_ener(t, vb))
        ext_ovlp = sum(voxel_ener(t, set(va).intersection(set(vb))))

        # Compute the barycenter of each group of voxels of the blob
        ba = barycenter(t, va)
        bb = barycenter(t, vb)

        # Using the new barycenter, take again all the voxels within a radius, to obtain the final blob
        va_ = voxel_in_blob(t, ba, rad, spacing)
        vb_ = voxel_in_blob(t, bb, rad, spacing)

        # Compute finl barycenter as the center blob position
        ca = barycenter(t, va_)
        cb = barycenter(t, vb_)

        Ea, Eb = sum(voxel_ener(t, va_)), sum(voxel_ener(t, vb_))
        ovlp = sum(voxel_ener(t, set(va_).intersection(set(vb_))))

        if Eb > Ea:
            eblob1, eblob2, eext1, eext2, posext1, posext2, bary1, bary2, voxblob1, voxblob2, posblob1, posblob2 = Eb, Ea, Eextb, Eexta, b, a, bb, ba, vb_, va_, cb, ca
        else:
            eblob1, eblob2, eext1, eext2, posext1, posext2, bary1, bary2, voxblob1, voxblob2, posblob1, posblob2 = Ea, Eb, Eexta, Eextb, a, b, ba, bb, va_, vb_, ca, cb
        if pos_to_coord:
            posext1, posext2, bary1, bary2, posblob1, posblob2 = vox_to_coord([posext1, posext2, bary1, bary2, posblob1, posblob2], spacing, initial)

        # Compute some values
        coords = vox_to_coord(voxel_pos(t), spacing, initial)
        rmax = max([np.sqrt(c[0]**2 + c[1]**2) for c in coords])
        min_ = coords.min(axis = 0)
        max_ = coords.max(axis = 0)

        track_info.append([dat_id, tID, binclass, rmax, 
                           min_[0], min_[1], min_[2],
                           max_[0], max_[1], max_[2],
                            eext1, eext2, ext_ovlp,
                            posext1[0], posext1[1], posext1[2], 
                            posext2[0], posext2[1], posext2[2],
                            # bary1[0], bary1[1], bary1[2],
                            # bary2[0], bary2[1], bary2[2],
                            eblob1, eblob2, ovlp,
                            posblob1[0], posblob1[1], posblob1[2], 
                            posblob2[0], posblob2[1], posblob2[2]])

    return pd.DataFrame(track_info, columns = ['dataset_id', 'trackID', 'binclass', 'rmax', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax',
                                                                                        'eext1', 'eext2', 'ext_ovlp',
                                                                                        'ext1_x', 'ext1_y', 'ext1_z', #positions of the found extremes (directly)
                                                                                        'ext2_x', 'ext2_y', 'ext2_z', 
                                                                                        # 'bar1_x', 'bar1_y', 'bar1_z', #positions of the first barycenter before recalibrating the blob
                                                                                        # 'bar2_x', 'bar2_y', 'bar2_z', 
                                                                                        'eblob1', 'eblob2', 'ovlp',
                                                                                        'blob1_x', 'blob1_y', 'blob1_z', #positions of the found blobs
                                                                                        'blob2_x', 'blob2_y', 'blob2_z'])

# ---------------------- REDISTRIBUTE ENERGY FUNCTIONS----------------------------------

from invisible_cities.types.ic_types   import NN

def redistribute_voxel_energy(group: pd.DataFrame, val: str = 'pass', ene: str = 'energy') -> pd.DataFrame:
    """
    Funtion that redistributes energy PER SLICE given the Q of the SiPMs
    """
    tot_E = group[ene].sum()
    mask = group[val].values < 0
    drp = group[mask]
    srv = group[~mask]

    if drp.empty:
        # no hits to distribute
        return srv

    if srv.empty:
        # hits turned NN
        drp = drp.copy()
        drp[['xbin', 'ybin']] = NN  # not put ene as NN because we need that info
        return drp

    # redistribute
    srv = srv.copy()
    srv[ene] = (srv[ene] / srv[ene].sum()) * tot_E
    return srv

def merge_NN_voxel(hits: pd.DataFrame, ene: str = 'energy') -> pd.DataFrame:
    # quickly split NN vs normal
    sel = hits['xbin'].eq(NN)
    if not sel.any():
        return hits

    normal = hits[~sel].copy()
    nn     = hits[sel]

    if normal.empty:
        # nothing to receive: drop all NN
        return normal

    # For each NN, find candidate receivers and closest distance
    # Build a mapping of receiver index -> energy/energy_correction
    corr = pd.DataFrame(0.0, index=normal.index, columns=[ene])

    z_normal = normal.zbin.values
    idx_normal = normal.index.values

    for _, row in nn.iterrows():  # still a loop over NN hits, but usually few
        dz = np.abs(z_normal - row['zbin'])
        m  = np.isclose(dz, dz.min())
        closest = normal.loc[idx_normal[m]]

        wE  = closest[ene] / closest[ene].sum()
        corr.loc[closest.index, ene]  += row[ene]  * wE

    normal[[ene]] += corr
    return normal

def make_thr_cut_voxel(df: pd.DataFrame, zslice: str = 'zbin'):
    df.loc[df.class_1 <= df.threshold, "pass"] = -1
    df.loc[df.class_1 > df.threshold, "pass"] = 0

    df = (df.groupby([zslice], group_keys=False)
                        .apply(redistribute_voxel_energy, "pass", "energy")
                        .reset_index(drop=True))
    df = merge_NN_voxel(df, "energy")
    return df.drop('pass', axis = 1)


def event_range(N, B, nfile):
    start = (nfile - 1) * B
    end = min(start + B, N)
    return np.arange(start, end, 1)


# ---------------------START SCRIPT---------------------------------

if dt == "MC":
    # Read bins info, the predicted voxels and the best threshold for them
    voxels_pred = pd.read_hdf(test_file_pred, 'DATASET/VoxelsPred')
    bins_info = pd.read_hdf(test_file_thr, 'DATASET/BinsInfo') 
    events_info_thr = pd.read_hdf(test_file_thr, 'DATASET/EventsInfo')

    # Add threshold to the voxel file
    voxels_pred = voxels_pred.merge(events_info_thr[['dataset_id', 'threshold', 'event_id']])

    spacing = bins_info[['size_x', 'size_y', 'size_z']].values
    initial = bins_info[['min_x', 'min_y', 'min_z']].values


    # Read the labelled voxels and hits
    label_reco = pd.read_hdf(labelfile, 'DATASET/RecoVoxels')
    label_hits = pd.read_hdf(labelfile, 'DATASET/MCHits') 

    # Correct id label
    min_id = events_info_thr[events_info_thr.label_basename == labelfile.split('/')[-1]].dataset_id.min()
    label_reco['dataset_id'] = label_reco['dataset_id'] + min_id
    label_hits['dataset_id'] = label_hits['dataset_id'] + min_id

    true_extremes = label_hits[label_hits.extlabel != 0].pivot(index='dataset_id', columns = 'extlabel', values=['x', 'y', 'z'])
    true_extremes.columns = [f"{col}{int(i)}" for col, i in true_extremes.columns]
    true_extremes = true_extremes.reset_index() 

    # Summary of the per event info with threshold
    labelinfopred = events_info_thr[np.isin(events_info_thr.dataset_id, label_reco.dataset_id.unique())][['dataset_id', 'binclass', 'total_energy', 'threshold', 'event_id']] 

    # Adding the reco labelled voxels the class_1 and threshold information
    label_reco = label_reco.merge(voxels_pred[['dataset_id', 'xbin', 'ybin', 'zbin', 'class_1', 'threshold']])

    # Apply the Paolina-like function to the whole events
    track_info = label_reco.groupby('dataset_id', group_keys = False).apply(get_track_info, contiguity, rad, spacing, initial, pos_to_coord) 

    # Apply the threshold and redistribute energy
    label_reco_thr = label_reco.groupby('dataset_id', group_keys=False).apply(make_thr_cut_voxel, "zbin")

    # Apply the Paolina-like function to the thresholded events
    track_info_thr = label_reco_thr.groupby('dataset_id', group_keys = False).apply(get_track_info, contiguity, rad, spacing, initial, pos_to_coord) 


    bins_info.to_hdf(savefile, key = 'DATASET/BinsInfo', mode = 'a', append= True, complib="zlib", complevel=4)
    true_extremes.to_hdf(savefile, key = 'DATASET/ExtPos', mode = 'a', append= True, complib="zlib", complevel=4)
    labelinfopred.to_hdf(savefile, key = 'DATASET/EventInfo', mode = 'a', append= True, complib="zlib", complevel=4)
    track_info.to_hdf(savefile, key = 'DATASET/TrackInfo', mode = 'a', append= True, complib="zlib", complevel=4)
    track_info_thr.to_hdf(savefile, key = 'DATASET/TrackInfoThr', mode = 'a', append= True, complib="zlib", complevel=4)

if dt == 'data':
    data_info = pd.read_hdf(datafile, 'DATASET/EventsInfo')
    data_bins = pd.read_hdf(datafile, 'DATASET/BinsInfo')
    # data_voxels = pd.read_hdf(datafile, 'DATASET/Voxels')

    evt_rng = event_range(len(data_info), nevents_per_file, nfile)

    data_info = data_info[np.isin(data_info.dataset_id, evt_rng)]
    stridx, endidx = data_info.stridx.values[0], data_info.endidx.values[-1]
    data_voxels = pd.DataFrame(tb.open_file(datafile, 'r').root.DATASET.Voxels[stridx:endidx])

    # data_voxels = data_voxels[np.isin(data_voxels.dataset_id, evt_rng)]

    spacing = data_bins[['size_x', 'size_y', 'size_z']].values
    initial = data_bins[['min_x', 'min_y', 'min_z']].values

    data_voxels = data_voxels.merge(data_info[['dataset_id', 'threshold', 'total_energy']]).rename(columns = {'decoscore':'class_1'})
    data_voxels['energy'] = data_voxels['energy'] * data_voxels['total_energy']

    track_info = data_voxels.groupby('dataset_id', group_keys = False).apply(get_track_info, contiguity, rad, spacing, initial, pos_to_coord, ['xbin', 'ybin', 'zbin', 'energy', 'energy']) 
    data_voxels_thr = data_voxels.groupby('dataset_id', group_keys=False).apply(make_thr_cut_voxel, "zbin")
    track_info_thr = data_voxels_thr.groupby('dataset_id', group_keys = False).apply(get_track_info, contiguity, rad, spacing, initial, pos_to_coord, ['xbin', 'ybin', 'zbin', 'energy', 'energy']) 

    data_bins.to_hdf(savedatafile, key = 'DATASET/BinsInfo', mode = 'a', append= True, complib="zlib", complevel=4)
    data_info.to_hdf(savedatafile, key = 'DATASET/EventInfo', mode = 'a', append= True, complib="zlib", complevel=4)
    track_info.to_hdf(savedatafile, key = 'DATASET/TrackInfo', mode = 'a', append= True, complib="zlib", complevel=4)
    track_info_thr.to_hdf(savedatafile, key = 'DATASET/TrackInfoThr', mode = 'a', append= True, complib="zlib", complevel=4)


