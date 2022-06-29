import tables as tb
import numpy  as np
import pandas as pd

from typing import Optional

from sklearn.utils.extmath import weighted_mode

from invisible_cities.io import mcinfo_io as mio
from invisible_cities.io import dst_io    as dio

from invisible_cities.core.configure import configure


def get_bin_indices(hits, bins, Rmax=207):
    segclass = 'segclass'
    binclass = 'binclass'
    fiducial_cut = (hits.x**2+hits.y**2)<Rmax**2
    binsX, binsY, binsZ = bins
    boundary_cut = (hits.x>=binsX.min()) & (hits.x<=binsX.max())\
                 & (hits.y>=binsY.min()) & (hits.y<=binsY.max())\
                 & (hits.z>=binsZ.min()) & (hits.z<=binsZ.max())

    hits_act = hits[fiducial_cut & boundary_cut].reset_index(drop = True)
    xbin = pd.cut(hits_act.x, binsX, labels = np.arange(0, len(binsX)-1)).astype(int)
    ybin = pd.cut(hits_act.y, binsY, labels = np.arange(0, len(binsY)-1)).astype(int)
    zbin = pd.cut(hits_act.z, binsZ, labels = np.arange(0, len(binsZ)-1)).astype(int)

    hits_act = hits_act.assign(xbin=xbin, ybin=ybin, zbin=zbin)
    hits_act.event_id = hits_act.event_id.astype(np.int64)

    if segclass not in hits.columns:
        hits_act = hits_act.assign(segclass = -1)
    if binclass not in hits.columns:
        hits_act = hits_act.assign(binclass = -1)

    #outputs df with bins index and energy, and optional label
    out = hits_act.groupby(['xbin', 'ybin', 'zbin', 'event_id']).apply(
        lambda df:pd.Series({'energy':df['energy'].sum(),
                             segclass:int(weighted_mode(df[segclass], df['energy'])[0][0]),
                             binclass:int(df[binclass].unique()[0])})).reset_index()
    out[segclass] = out[segclass].astype(int)
    out[binclass] = out[binclass].astype(int)
    return out


def add_clf_labels(hits, particles):
    clf_labels = particles.groupby('event_id').particle_name.apply(lambda x:sum(x=='e+')).astype(int)
    clf_labels.name = 'binclass'
    return hits.merge(clf_labels, left_index=True, right_index=True).reset_index()[['event_id', 'x', 'y', 'z', 'energy', 'binclass']]


def add_seg_labels(hits, particles, delta_t=None, delta_e=None, label_dict={'track':1, 'blob':2, 'rest':0}):
    label_dict={'track':1, 'blob':2, 'rest':0}
    #hits_par = pd.merge(hits, particles, left_index=True, right_index=True)
    hits_par = pd.merge(hits, particles, on = ['event_id', 'particle_id'])
    per_part_info = hits_par.groupby(['event_id', 'particle_id', 'particle_name', 'creator_proc']).agg(
        {'time':[('timemin',min), ('timemax',max)], 'energy':[('energy', sum)]})
    per_part_info.columns = per_part_info.columns.get_level_values(1)
    per_part_info['DT']   = per_part_info.timemax-per_part_info.timemin
    per_part_info.reset_index(inplace=True)

    #events with positrons and electrons
    positrion_event_ids = per_part_info[per_part_info.particle_name == 'e+'].event_id.unique()
    electron_event_ids  = np.setdiff1d(per_part_info.event_id.unique(), positrion_event_ids)

    #extract particle id that are main track
    #for e+e- that is positron and electron created in conv process
    tracks_pos = per_part_info[(per_part_info.event_id.isin(positrion_event_ids)) &\
                               (per_part_info.particle_name.isin(['e+', 'e-'])    &\
                                (per_part_info.creator_proc == 'conv'))]

    #for no e+ events look for longest electron track
    tracks_el = per_part_info[(per_part_info.event_id.isin(electron_event_ids)) &\
                              (per_part_info.particle_name == 'e-')             &\
                              (per_part_info.creator_proc  == 'compt')]
    tracks_el = tracks_el.loc[tracks_el.groupby('event_id').DT.idxmax()]

    #merge tracks information with segclass to hits_labels
    tracks_info = pd.concat([tracks_el, tracks_pos]).sort_values('event_id')
    tracks_info = tracks_info.assign(segclass = label_dict['track'])

    hits_par = hits_par.reset_index()
    hits_label = hits_par.merge(tracks_info[['event_id', 'particle_id','timemax', 'timemin', 'DT','segclass']],
                                how='outer', on=['event_id', 'particle_id'])
    hits_label = hits_label.fillna(label_dict['rest']) #label all non-tracks as rest
    #add cumsum energy per event per particle inverdet of hits order
    hits_label = hits_label.sort_values(['event_id', 'particle_id', 'hit_id'], ascending=[True, True, False])
    hits_label = hits_label.assign(cumenergy = hits_label.groupby(['event_id', 'particle_id']).energy.cumsum())

    if delta_t is not None:
        #label as blobs hits that are tmax-deltat per particle inside track
        blob_msk = (hits_label.DT<delta_t)
        hits_label.loc[(hits_label.segclass==label_dict['track'])& blob_msk, 'segclass'] = label_dict['blob']
    elif delta_e is not None:
        #label as blobs hits that sum up to delta_e last energy deposition
        blob_msk = (hits_label.cumenergy<delta_e)
        hits_label.loc[(hits_label.segclass==label_dict['track'])& blob_msk, 'segclass'] = label_dict['blob']
    hits_label = hits_label[['event_id', 'x', 'y', 'z', 'energy', 'segclass']].reset_index(drop=True)
    return hits_label



def  add_clf_seg_labels(hits, particles, delta_t=None, delta_e=None, label_dict={'track':1, 'blob':2, 'rest':0}):
    seg_hits = add_seg_labels(hits, particles, delta_t, delta_e, label_dict)
    clf_labels = particles.groupby('event_id').particle_name.apply(lambda x:sum(x=='e+')).astype(int)
    clf_labels.name = 'binclass'
    return seg_hits.merge(clf_labels, on=['event_id'])[['event_id', 'x', 'y', 'z', 'energy', 'binclass', 'segclass']]
