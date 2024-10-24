""" This model contains functions that read data stored on disk and do basic manipulation"""

import numpy as np
import pandas as pd
import tables as tb

def get_mchits(filename, event):
    """ This function returns numpy tensor, where each row represents x, y, z position of a hit and its energy.
    As input it takes filename to read and event number."""
    # TODO: change to new format once data is rerun
    with tb.open_file(filename) as tab:
        indx_tab = tab.root.MC.extents.get_where_list( 'evt_number == {}'.format(event))
        last_hit = tab.root.MC.extents[indx_tab]['last_hit']
        first_hit = tab.root.MC.extents[indx_tab[0]-1]['last_hit'] if indx_tab[0]>0 else -1
        hits_ = tab.root.MC.hits[int(first_hit+1):int(last_hit+1)][['hit_position', 'hit_energy']]
        hits = np.zeros((len(hits_), 4))
        hits[:, :3] = hits_['hit_position']
        hits[:, 3] = hits_['hit_energy']
    return hits



def get_3d_input(filename, eventid, binsX, binsY, binsZ):
    """ This function transform mc hits to 3d voxelized hits"""

    hits = get_mchits(filename, eventid)

    xcord = hits[:,0]
    ycord = hits[:,1]
    zcord = hits[:,2]
    ener  = hits[:,3]

    xdig = np.digitize(xcord,binsX)
    ydig = np.digitize(ycord,binsY)
    zdig = np.digitize(zcord,binsZ)

    return xdig, ydig, zdig, ener

def index_tables(file_out):
    """
    -finds all tables in output_file
    -checks if any columns in the tables have been marked to be indexed by writers
    -indexes those columns
    """
    with tb.open_file(file_out, 'r+') as h5out:
        for table in h5out.walk_nodes(classname='Table'):        # Walk over all tables in h5out
            if 'columns_to_index' not in table.attrs:  continue  # Check for columns to index
            for colname in table.attrs.columns_to_index:         # Index those columns
                table.colinstances[colname].create_index()