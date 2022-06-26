import numpy    as np
import pandas   as pd
import tables   as tb
import networkx as nx

import itertools

from next_sparseconvnet.utils.data_loaders             import DataGen, LabelType, read_events, read_events_info

def number_of_blobs(event_df, threshold, class_type = 'class_2', max_distance = np.sqrt(3)):
    '''
    For a prediction, returns the number of blobs for data above a threshold using graphs
    '''
    selected_hits = event_df[['xbin', 'ybin', 'zbin']][event_df[class_type]>threshold]
    voxels = [tuple(x) for x in selected_hits.to_numpy()]
    vox_graph = nx.Graph()
    vox_graph.add_nodes_from(voxels)
    for va, vb in itertools.combinations(voxels, 2):
        va_arr, vb_arr = np.array(va), np.array(vb)
        dis = np.linalg.norm(va_arr-vb_arr)
        if dis < max_distance:
            vox_graph.add_edge(va, vb, distance = dis)
    nblobs = nx.algorithms.components.number_connected_components(vox_graph)
    return nblobs

def segmentation_blob_classification(orig_dataset_path, pred_dataset_path, threshold, nevents = None, class_type = 'class_2', max_distance = np.sqrt(3)):
    '''
    Adds a column to the events dataframe where depending on the number of blobs the predicted data had,
    events are classified as signal or background
    '''
    original_events = read_events_info(orig_dataset_path, nevents)
    pred_events = read_events(pred_dataset_path, nevents, table = 'VoxelsPred')

    nblobs = np.array([])
    for idx, event_df in pred_events.groupby('dataset_id'):
        nblobs = np.append(nblobs, number_of_blobs(event_df, threshold, class_type = class_type, max_distance = max_distance))

    #this ==2 is decisive bc puts the ones with more than 2 blobs as bkg, still thinking about it
    original_events = original_events.assign(pred_class = pd.Series(nblobs) == 2)

    original_events.pred_class = original_events.pred_class.astype(int)
    return original_events

def success_rates(true_class, predicted_class):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(true_class)):
        if predicted_class[i] == 1 and true_class[i] == 1:
            TP += 1
        if predicted_class[i] == 0 and true_class[i] == 0:
            TN += 1
        if predicted_class[i] == 1 and true_class[i] == 0:
            FP += 1
        if predicted_class[i] == 0 and true_class[i] == 1:
            FN += 1

    good = TP + TN
    bad = FP + FN
    total = good + bad
    accuracy = good/total

    tpr = TP / (TP + FN)
    fpr = FP / (FP +TN)
    tnr = 1 - fpr
    return accuracy, tpr, tnr
