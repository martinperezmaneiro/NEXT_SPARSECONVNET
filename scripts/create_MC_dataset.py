#!/usr/bin/env python
"""
This script creates hdf5 files that contains:
 - DATASET/BinClassHits - voxelized hits table with labels (binary classification)
 - DATASET/SegClassHits - voxelized hits table with labels (segmentation)
 - DATASET/BinInfo      - table that stores info about bins
 - DATASET/EventsInfo   - table that contains EventID and binary classification label
"""
import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from invisible_cities.io import mcinfo_io as mio
from invisible_cities.io import dst_io    as dio

from invisible_cities.core  .configure import configure
# from invisible_cities.cities.components import index_tables
from next_sparseconvnet.utils.data_io   import index_tables
from next_sparseconvnet.data_io_scripts import MC_dataset_table_utils as mcutils


# def get_last_id(filename):
#     try:
#         with tb.open_file(filename, 'r') as tab:
#             #TODO add check for bin info
#             dataset_id = tab.root.DATASET.EventsInfo.cols.dataset_id[-1]
#     except (FileNotFoundError, IOError):
#         print ('Creating new dataset')
#         dataset_id = 0
#     except tb.exceptions.NoSuchNodeError:
#         print('File exists but wrong structure')
#         exit(1)
#     return dataset_id

if __name__ == "__main__":
    config  = configure(sys.argv).as_namespace
    filesin = glob(os.path.expandvars(config.files_in))
    fout = os.path.expandvars(config.file_out)
    start_id = 0
    if os.path.isfile(fout):
        raise Exception('output file exist, please remove it manually')
    for i, f in enumerate(filesin):
        print(i, f)
        eventInfo, binsInfo, hits = mcutils.get_MCtables(f, config, start_id)
        start_id +=len(eventInfo)
        with tb.open_file(fout, 'a') as h5out:
            dio.df_writer(h5out, eventInfo, 'DATASET', 'EventsInfo', columns_to_index=['dataset_id'], str_col_length=64)
            dio.df_writer(h5out, binsInfo , 'DATASET', 'BinsInfo')
            dio.df_writer(h5out, hits     , 'DATASET', 'Voxels', columns_to_index=['dataset_id'])
    index_tables(fout)
