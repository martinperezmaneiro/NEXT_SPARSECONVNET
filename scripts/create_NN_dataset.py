'''
This scripts takes ANY group of voxelized + labelled files and creates a train + valid + test dataset to train 
a net in NEXT_SPARSECONVNET.
'''

import os
import glob
import pandas as pd
import tables as tb
import numpy as np
import invisible_cities.io.dst_io as dio

p = '5bar' # '4bar'
dt = ['0nubb', '1eroi'] # ['208Tl']
prod_type = 'sensim_15mm' #'diffsim_10mm' #'PORT_1a/label/prod'
prod_name = 'sensim_15mm' #'PORT_1a_label
datasets = ['train', 'valid', 'test']

basedir = '/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/{p}/'.format(p=p)
proddir = '{dt}' + '/prod/{prodt}/*'.format(prodt = prod_type)

file_savename = 'dataset_{p}_{prodn}_scn_'.format(p=p, prodn = prod_name) + '{dataset}.h5'
savedir = basedir + file_savename

table_group = 'Sensim' #'Diffsim' # 'DATASET'
table_name  = 'sns_vox_df' #'vox_diff' # 'RecoVoxels'
event_info_tb_name = 'voxel_info' # 'BinsInfo'

file_percentage = 1
id_name = 'event' # 'dataset_id'
coords_name = ['xbin', 'ybin', 'zbin']
ener_name = 'ebin' # 'energy'
label_name = ['segclass', 'decolabel', 'extlabel']
seg_dct = {1:0, 2:1, 3:2} # None # FOR LABELLING FILES, NONE IS OK BECAUSE (0 - GHOST, 1 - OTHER, 2 - TRACK, 3 - BLOB); FOR SENSIM
norm_ener = True

def get_and_split_files(files, train_perc = 0.8, valid_perc = 0.1):
    train_num = int(len(files) * train_perc)
    valid_num = int(len(files) * valid_perc)
    train_files = files[:train_num]
    valid_files = files[train_num:train_num + valid_num]
    test_files  = files[train_num + valid_num:]
    return train_files, valid_files, test_files

def get_dataset_files(directory, dt, perc = 1, train_perc = 0.8, valid_perc = 0.1):
    train, valid, test = [], [], []
    for d in dt:
        files_dir = glob.glob(directory.format(dt = d))
        files = sorted(files_dir, key = lambda x: int(x.split('_')[-2]))
        fil_num = int(len(files) * perc)
        train_files, valid_files, test_files = get_and_split_files(files[:fil_num], train_perc=train_perc, valid_perc=valid_perc)
        train.extend(train_files)
        valid.extend(valid_files)
        test.extend(test_files)
    return train, valid, test


if __name__ == "__main__":
    for dataset, infiles in zip(datasets, get_dataset_files(basedir + proddir, dt, perc = file_percentage)):
        print('Creating ', dataset, ' dataset with ', len(infiles), ' files')
        savefile = savedir.format(dataset = dataset)
        print('Saving it into: ', savefile)
        start_id = 0
        index_count = 0
        for file in infiles:
            pathname, basename = os.path.split(file)
            df = pd.read_hdf(file, table_group + '/' + table_name)

            # This is needed for Sensim because labels are set from 1 to 3 (ghost class never appears, as spurious hits always come from a certain MC hit)
            # This reminds me, now I could "update" the labelling method for Sensim, given that now is quite simple (fewer functions are used)
            # For label, no need because the ghost_class is 0
            if seg_dct and 'segclass' in label_name:
                df = df.assign(segclass = df['segclass'].map(seg_dct))

            # If the file has EventsInfo (they come from labelling), we use directly that table
            # If the file does not have that table (coming from Sensim), we create it and do a mapping
            with tb.open_file(file, 'r') as h5in:
                exist_eventinfo = '/DATASET/EventsInfo' in h5in
                
            if not exist_eventinfo:
                eventInfo = df[[id_name, 'binclass']].drop_duplicates().reset_index(drop=True)
                eventInfo = eventInfo.assign(pathname = pathname, basename = basename)
            else:
                eventInfo = pd.read_hdf(file, 'DATASET/EventsInfo')

            dct_map = {eventInfo.iloc[i][id_name] : i+start_id for i in range(len(eventInfo))}
            df = df[[id_name]+ coords_name + [ener_name] + label_name]
            df = df.assign(dataset_id = df[id_name].map(dct_map))
        
            eventInfo = eventInfo.assign(dataset_id = eventInfo[id_name].map(dct_map))
            eventInfo = eventInfo.merge(df.groupby(id_name)[ener_name].sum().rename('total_energy'), on = id_name)

            eventInfo = eventInfo.assign(
                stridx = df.groupby(id_name).apply(lambda x: x.index[0]).values + index_count,
                endidx   = df.groupby(id_name).apply(lambda x: x.index[-1]).values + 1 + index_count
            )

            if not exist_eventinfo: df = df.drop(id_name, axis = 1)
            
            # Change column names to have the ones read in the NN code
            df = df.rename(columns = {coords_name[0]:'xbin', 
                                    coords_name[1]:'ybin', 
                                    coords_name[2]:'zbin', 
                                    ener_name:'energy'})

            if norm_ener:
                df['energy'] = df.groupby('dataset_id')['energy'].apply(lambda x: x / x.sum())
            start_id += len(eventInfo)
            index_count += len(df)
            df = df.merge(eventInfo[['dataset_id', 'binclass']])
            # Write the voxels of each file
            with tb.open_file(savefile, 'a') as h5out:
                dio.df_writer(h5out, df       , 'DATASET', 'Voxels'    , columns_to_index=['dataset_id'], compression = "ZLIB4")
                dio.df_writer(h5out, eventInfo, 'DATASET', 'EventsInfo', columns_to_index=['dataset_id'], str_col_length=128, compression = "ZLIB4")

        print(start_id, ' events saved')

        # Write the other info
        binsInfo = dio.load_dst(file, table_group, event_info_tb_name)
        # Assure it has nbins info
        if 'nbins_x' not in binsInfo:
            for coord in ['x', 'y', 'z']:
                min_, max_, size = binsInfo['min_' + coord].values[0], binsInfo['max_' + coord].values[0], binsInfo['size_' + coord].values[0]
                binsInfo['nbins_' + coord] = len(np.arange(min_, max_ + size, size)) - 1
        with tb.open_file(savefile, 'a') as h5out:
            dio.df_writer(h5out, binsInfo , 'DATASET', 'BinsInfo', compression = "ZLIB4")