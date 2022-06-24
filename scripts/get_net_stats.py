import sys
import os
import tables as tb
import numpy  as np
import pandas as pd

from glob import glob
from time import time
from invisible_cities.core  .configure  import configure
import invisible_cities.io.dst_io as dio

from next_sparseconvnet.utils.blob_classification_utils import *


if __name__ == "__main__":

    config = configure(sys.argv).as_namespace
    orig_file = os.path.expandvars(config.orig_file)
    pred_file = os.path.expandvars(config.pred_file)
    out_file  = os.path.expandvars(config.out_file)

    if os.path.isfile(out_file):
        raise Exception('output file exists, remove it manually')

    nevents = config.nevents

    start_th, final_th, num_th = config.start_th, config.final_th, config.num_th

    start_dis, final_dis, num_dis = config.start_dis, config.final_dis, config.num_dis

    threshold = np.linspace(start_th, final_th, num_th)
    distances = np.linspace(start_dis, final_dis, num_dis)
    result_df = pd.DataFrame()

    for th in threshold:
        for dis in distances:
            print(round(th, 4), round(dis, 4))
            start_time = time()
            eventInfo_pred = segmentation_blob_classification(orig_file, pred_file, th, nevents = nevents, max_distance = dis)
            acc, tpr, tnr = success_rates(eventInfo_pred['binclass'], eventInfo_pred['pred_class'])
            result_df = result_df.append(pd.DataFrame([[th, dis, acc, tpr, tnr]], columns = ['blobclass_th', 'blobcount_dis', 'acc', 'tpr', 'tnr']))
            print((time() - start_time) / 60, 'mins')

    with tb.open_file(out_file, 'a') as h5out:
        dio.df_writer(h5out, result_df, 'NN_RESULTS', 'Statistics')
