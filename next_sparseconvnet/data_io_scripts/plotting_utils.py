import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot  as plt
import matplotlib         as mpl
from   mpl_toolkits.mplot3d import Axes3D
from   matplotlib.patches   import Patch
import itertools

def plot_projections(hits, value='energy', coords = ['x', 'y', 'z'], cmap = mpl.cm.jet, th = 0):
    fig, axs = plt.subplots(nrows=1, ncols=3,
                                        figsize=(12, 6))
    coors_pairs = itertools.combinations(coords, 2)
    cmap.set_under('white')
    for i, coor_pair in enumerate(coors_pairs):
        sel = hits.groupby(list(coor_pair))[value].sum()
        ind0 = sel.index.get_level_values(coor_pair[0])
        ind1 = sel.index.get_level_values(coor_pair[1])
        newind0 = np.arange(ind0.min(), ind0.max()+1)
        newind1 = np.arange(ind1.min(), ind1.max()+1)
        xx, yy = np.meshgrid(newind0, newind1)
        newind = pd.Index(list(zip(xx.flatten(), yy.flatten())), name=tuple(coor_pair))
        sel = sel.reindex(newind,  fill_value=0).reset_index()
        sel = pd.pivot_table(sel, values=value, index=[coor_pair[0]],
                        columns=[coor_pair[1]], aggfunc=np.sum)
        #print((newind0.min(),newind0.max(), newind1.min(),  newind1.max()))
        axs[i].imshow(sel.T, origin='lower', vmin=th+np.finfo(float).eps, extent=(newind0.min(),newind0.max(), newind1.min(),  newind1.max()),
                      cmap=cmap, aspect='auto')
        axs[i].set_xlabel(coor_pair[0])
        axs[i].set_ylabel(coor_pair[1])
    fig.tight_layout()

    plt.show()

def plot_3d_vox(hits_digitized, voxel_size, value='energy', coords = ['x', 'y', 'z'], affluence = (5, 5, 5), th=0, edgecolor='k', linewidth = .3, cmap=mpl.cm.jet, opacity = 1):
    '''
    Function to plot voxels (they have to be normalized)
    Args:
        hits_digitized: DataFrame or list/tuple/array of lists/tuples/arrays in any combination
    Contains the spatial information of the voxels and their content. If we don't use a DataFrame, the input
    must have the structure (x, y, z, content), where content is usually the energy or the segclass. Its shape
    will be (4, N).
        value: STR
    Name of the content column in the DataFrame. Will be also the label of the colorbar.
        coords: LIST
    Name of the coords column in the DataFrame.
        th: FLOAT
    Low threshold of the content of the voxels to plot.
        edgecolor: STR
    Color of the edges of the voxels.
        linewidth: FLOAT
    Width of the edges of the voxels.
        cmap: matplotlib.cm
    Used colormap.
        opacity = FLOAT
    Value from 0 to 1 that indicates the opacity of the voxels.
    '''

    if type(hits_digitized) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(hits_digitized).T
        hits_digitized = pd.DataFrame(coor, columns = coords + [value])

    xcoord  = hits_digitized[coords[0]].values
    ycoord  = hits_digitized[coords[1]].values
    zcoord  = hits_digitized[coords[2]].values
    content = hits_digitized[value].values

    xmin, xmax = min(xcoord), max(xcoord)
    ymin, ymax = min(ycoord), max(ycoord)
    zmin, zmax = min(zcoord), max(zcoord)

    labels, ticks = plot_label_creator((xmin, ymin, zmin), (xmax, ymax, zmax), voxel_size, affluence)

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([xcoord-xmin+1,
                          ycoord-ymin+1,
                          zcoord-zmin+1])

    nonzeros = nonzeros.astype(int)
    xarr[tuple(nonzeros)] = content
    dim     = xarr.shape
    voxels  = xarr > th

    fig  = plt.figure(figsize=(8, 8), frameon=False)
    gs   = fig.add_gridspec(12, 12)
    ax   = fig.add_subplot(gs[0:12, 0:10], projection = '3d')
    axcb = fig.add_subplot(gs[1:11, 11])
    norm = mpl.colors.Normalize(vmin=xarr.min(), vmax=xarr.max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors * opacity, edgecolor=edgecolor, linewidth = linewidth)
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    
    if value == 'energy':
        cb_label = 'Energy (MeV)'    
    else:
        cb_label = value
        
    cb.set_label (cb_label)
    #cb.ax.tick_params(labelsize=13)
    cb.formatter.set_powerlimits((0, 0))

    ax.set_xticklabels(labels[0])
    ax.set_xticks(ticks[0])
    ax.set_yticklabels(labels[1])
    ax.set_yticks(ticks[1])
    ax.set_zticklabels(labels[2])
    ax.set_zticks(ticks[2])

    plt.show()

def plot_3d_hits(hits, value='energy', coords = ['x', 'y', 'z'], cmap = mpl.cm.jet, opacity = 1):
    '''
    Function to plot hits
    Args:
        hits: DataFrame or list/tuple/array of lists/tuples/arrays
    Contains the spatial information of the hits and their content. If we don't use a DataFrame, the input
    must have the structure (x, y, z, content), where content is usually the energy or the segclass. Its shape
    will be (4, N).
        value: STR
    Name of the content column in the DataFrame. Will be also the label of the colorbar.
        coords: LIST
    Name of the coords column in the DataFrame.
        cmap: matplotlib.cm
    Used colormap.
        opacity = FLOAT
    Value from 0 to 1 that indicates the opacity of the hits.
    '''

    if type(hits) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(hits).T
        hits = pd.DataFrame(coor, columns = coords + [value])

    xcoord  = hits[coords[0]].values
    ycoord  = hits[coords[1]].values
    zcoord  = hits[coords[2]].values
    content = hits[value].values

    fig  = plt.figure(figsize=(8, 8), frameon=False)
    gs   = fig.add_gridspec(12, 12)
    ax   = fig.add_subplot(gs[0:12, 0:10], projection = '3d')
    axcb = fig.add_subplot(gs[1:11, 11])
    norm = mpl.colors.Normalize(vmin=min(content), vmax=max(content))

    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = np.asarray(np.vectorize(m.to_rgba)(content))
    colors = np.rollaxis(colors, 0, 2)

    ax.scatter(xcoord, ycoord, zcoord, c=colors * opacity, marker='o')
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='vertical')


    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    cb.set_label (value, size = 15)
    cb.ax.tick_params(labelsize=13)
    cb.formatter.set_powerlimits((0, 0))
    plt.show()


def plot_3d_hits_discrete(labelled_hits, value = 'segclass', coords = ['x', 'y', 'z']):

    color_dict = {1:'deepskyblue', 2:'gold', 3:'tab:red', 4:'deepskyblue', 5:'gold', 6:'tab:red', 7:'tab:green'}

    ax  = plt.figure(figsize=(8, 8), frameon=False).add_subplot(projection='3d')
    hit_color = labelled_hits[value].map(color_dict)
    ax.scatter(labelled_hits[coords[0]], labelled_hits[coords[1]], labelled_hits[coords[2]], c=hit_color, marker='o')
    legend_elements = [Patch(facecolor='deepskyblue', label='other class'),
                       Patch(facecolor='gold',        label='track class'),
                       Patch(facecolor='tab:red',     label='blob class')]
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    ax.legend(handles=legend_elements, fontsize=15)

    plt.show()

def plot_label_creator(min_vals, max_vals, voxel_size, affluence):
    '''
    Function to create ticks and its labels to plot an event.
    Args:
        min_vals: TUPLE
    Contains the minimum values for each coordinate of the position of all the voxels in an event.
        max_vals: TUPLE
    Contains the maximum values for each coordinate of the position of all the voxels in an event.
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate. Used to create the labels.
        affluence: TUPLE
    Separation parameter for the labels in each coordinate. If 1, labels will be plotted for each voxel step,
    if 2, they will be plotted each 2 voxel steps, etc.
    RETURNS:
        labels: LIST
    List of arrays (one per coord.) with the labels we want to use (those consistent with the voxel size in
    each coordinate).
        ticks: LIST
    List of arrays (one per coord.) with the ticks that we want to label.
    '''
    labels = []
    ticks  = []
    for mini, maxi, size, aff in zip(min_vals, max_vals, voxel_size, affluence):
        labels.append(np.arange((mini - mini) * size, (maxi - mini + 2) * size, size * aff))
        ticks.append(np.arange(0, maxi - mini + 2, aff))
    return labels, ticks


#another useful colorbars: viridis, cividis....
def plot_cloud_voxels(labelled_voxels, voxel_size, affluence = (2, 2, 2), value = ['segclass', 'segclass'], coords = ['xbin', 'ybin', 'zbin'], th=0, edgecolor='k', linewidth = .3, cmap = [mpl.cm.coolwarm, mpl.cm.coolwarm], opacity = [1,1]):
    '''
    This function takes all the labelled voxels (the output of one of the label_neighbours function) and plots them
    separately if they are MC voxeles (beersheba voxels that were coincident with those of their MC true event and
    those that also were assigned with the merge_MC_beersh_voxels function) or if they are cloud voxels (beersheba
    voxels that had no coincidence with the MC). This separation is made to personalize for each one the plot:
    different values to plot, colorbars or opacities. Also, you can plot separately and with a solid color the ghost
    voxels to see them clearly.
    Args:
        labelled_voxels: DATAFRAME or list/tuple/array of lists/tuples/arrays in any combination
    Output of one of the label_neighbours function (using DF). Contains all the voxeles labelled as pure MC
    class, their neighbours and the ghost class.
    If we choose not to use a DF we should input (x, y, z, E, segclass) because
    in this function the segclass is always needed to separate the different classes.
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate. Used to create the labels.
        affluence: TUPLE
    Separation parameter for the labels in each coordinate. If 1, labels will be plotted for each voxel step,
    if 2, they will be plotted each 2 voxel steps, etc.
        value: LIST
    List with the wanted features to plot, the first for the MC voxels and the second for the cloud ones: [MC, cloud].
        coords: LIST
    List with the name of the coordinates in the dataframe to plot.
        th: FLOAT
    Minimum value to plot a voxel (usually used with the energy feature, to see the most energetic voxels).
        edgecolor: STR
    Color code of the matplotlib library to plot the edges of the voxel.
        linewidth: FLOAT
    Number to change width edges of the voxels.
        cmap: LIST
    List with the colormap from the matplotlib library to use with each kind of voxels: [MC, cloud].
        opacity: LIST
    List with numbers from 0 to 1 (being 0 transparent and 1 opac), for both kind of voxels: [MC, cloud].
    '''

    if type(labelled_voxels) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_voxels).T
        labelled_voxels = pd.DataFrame(coor, columns = coords + ['energy'] + ['segclass'])

    xcoord  = labelled_voxels[coords[0]].values
    ycoord  = labelled_voxels[coords[1]].values
    zcoord  = labelled_voxels[coords[2]].values
    content = labelled_voxels[value[0]].values

    xmin, xmax = min(xcoord), max(xcoord)
    ymin, ymax = min(ycoord), max(ycoord)
    zmin, zmax = min(zcoord), max(zcoord)

    labels, ticks = plot_label_creator((xmin, ymin, zmin), (xmax, ymax, zmax), voxel_size, affluence)

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2

    mc_label = labelled_voxels[np.isin(labelled_voxels.segclass, (1, 2, 3))]
    cloud    = labelled_voxels[np.isin(labelled_voxels.segclass, (4, 5, 6))]
    ghost    = labelled_voxels[np.isin(labelled_voxels.segclass, 7)]

    #CLOUD
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([cloud[coords[0]].values-xmin,
                          cloud[coords[1]].values-ymin,
                          cloud[coords[2]].values-zmin])
    nonzeros = nonzeros.astype(int)
    xarr[tuple(nonzeros)] = cloud[value[1]].values
    dim     = xarr.shape
    voxels  = xarr > th

    fig  = plt.figure(figsize=(15, 7), frameon=False)
    gs   = fig.add_gridspec(1, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 21])
    norm = mpl.colors.Normalize(vmin=cloud[value[1]].min(), vmax=cloud[value[1]].max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1])

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors * opacity[1], edgecolor=edgecolor, linewidth = linewidth)
    cb_cloud = mpl.colorbar.ColorbarBase(axcb, cmap=cmap[1], norm=norm, orientation='vertical')

    #MC
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([mc_label[coords[0]].values-xmin,
                          mc_label[coords[1]].values-ymin,
                          mc_label[coords[2]].values-zmin])
    nonzeros = nonzeros.astype(int)
    xarr[tuple(nonzeros)] = mc_label[value[0]].values
    dim     = xarr.shape
    voxels  = xarr > th
    axcb  = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=mc_label[value[0]].min(), vmax=mc_label[value[0]].max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap[0])

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors * opacity[0], edgecolor=edgecolor, linewidth = linewidth)
    cb_mc = mpl.colorbar.ColorbarBase(axcb, cmap=cmap[0], norm=norm, orientation='vertical')


    #GHOST
    if ghost.empty == False:
        xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

        nonzeros = np.vstack([ghost[coords[0]].values-xmin,
                              ghost[coords[1]].values-ymin,
                              ghost[coords[2]].values-zmin])
        nonzeros = nonzeros.astype(int)
        xarr[tuple(nonzeros)] = ghost['segclass'].values
        dim     = xarr.shape
        voxels  = xarr > th

        ax.voxels(voxels, facecolors='g', edgecolor=edgecolor, linewidth = linewidth)
        legend_elements = [Patch(facecolor='g', edgecolor='g', label='ghost class')]
        ax.legend(handles=legend_elements)

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')

    ax.set_xticks(ticks[0])
    ax.set_xticklabels(labels[0])
    ax.set_yticks(ticks[1])
    ax.set_yticklabels(labels[1])
    ax.set_zticks(ticks[2])
    ax.set_zticklabels(labels[2])

    if value[0] == 'segclass':
        cb_ticks = np.sort(mc_label.segclass.unique())
        cb_mc.set_ticks(cb_ticks)
        cb_mc.set_ticklabels(cb_ticks)
    cb_mc.set_label(value[0])

    if value[1] == 'segclass':
        cb_ticks = np.sort(cloud.segclass.unique())
        cb_cloud.set_ticks(cb_ticks)
        cb_cloud.set_ticklabels(cb_ticks)
    cb_cloud.set_label('cloud ' + value[1])

    plt.show()


def plot_adaption_hits_to_voxel_scale(event_hits, voxel_size, start_bin, coords = ['x', 'y', 'z']):
    '''
    We want to plot the MC hits in order to visualize better how the neighbour classes went, so this function
    scales the positions for them to coincide.
    Args:
        event_hits: DATAFRAME
    Labelled MC hits of an event.
        voxel_size: TUPLE
    Contains the size of the voxel for each dimension.
        coords: LIST
    List with the coordinate names in the dataframe.
    RETURNS:
        event_scaled_hits: DATAFRAME
    Copy of the input DF but with scaled coordinates to the voxel size.
    '''

    event_scaled_hits = event_hits.copy()
    for (coor, size), start in zip(zip(coords, voxel_size), start_bin):
        event_scaled_hits[coor] = (event_hits[coor] - start) / size
    return event_scaled_hits


def plot_cloud_voxels_and_hits(labelled_voxels, labelled_hits, voxel_size, start_bin, affluence = (2, 2, 2), value = ['segclass', 'segclass', 'segclass'], coords = ['xbin', 'ybin', 'zbin'], coords_mc = ['x', 'y', 'z'], th=0, edgecolor='k', linewidth = .3, cmap = [mpl.cm.coolwarm, mpl.cm.coolwarm, mpl.cm.coolwarm], opacity = [0, 1, 1]):
    '''
    This function is made to plot the neighbour labelled hits (the cloud) with some transparency, and the hits
    inside this cloud, to see how they agree. This is better than using the MC labels because it's difficult to
    see something this way, so these voxels will be mainly transparent when using this.
    Args:
        labelled_voxels: DATAFRAME or list/tuple/array of lists/tuples/arrays in any combination
    Output of one of the label_neighbours function (using DF). Contains all the voxeles labelled as
    pure MC class, their neighbours and the ghost class.
    If we choose not to use a DF we should input (x, y, z, E, segclass) because in this function the
    segclass is always needed to separate the different classes.
        labelled_hits: DATAFRAME or list/tuple/array of lists/tuples/arrays in any combination
    Second output of the labelling_MC function (using DF). Contains the MC hits with their segclass.
    If we don't use a DataFrame, the input must have the structure (x, y, z, content), where content
    is usually the energy or the segclass. Its shape will be (4, N).
        voxel_size: TUPLE
    Contains the voxel size of the detector for each coordinate. Used to create the labels and scale the hits.
        affluence: TUPLE
    Separation parameter for the labels in each coordinate. If 1, labels will be plotted for each voxel step,
    if 2, they will be plotted each 2 voxel steps, etc.
        value: LIST
    List with the wanted features to plot, for the MC and cloud voxels, and the hits: [MC, cloud, hits].
        coords: LIST
    List with the name of the coordinates in the dataframe to plot.
        th: FLOAT
    Minimum value to plot a voxel (usually used with the energy feature, to see the most energetic voxels).
        edgecolor: STR
    Color code of the matplotlib library to plot the edges of the voxel.
        linewidth: FLOAT
    Number to change width edges of the voxels.
        cmap: LIST
    List with the colormap from the matplotlib library to use with each kind of voxels/hits: [MC, cloud, hits].
        opacity: LIST
    List with numbers from 0 to 1 (being 0 transparent and 1 opac), for both kind of voxels: [MC, cloud, hits].
    '''

    #Escojo el minimo y el maximo por coordenada de ambos DF, ya que si no un df se desplaza respecto a otro (probablemente)
    #De esta forma, el ''frame'' que calcula (nbins en cada coord) será el máximo siempre, y a las coordenadas a rellenar
    #le restamos el mínimo de ambos DF ya que si le restamos a cada uno por su cuenta se desplza que era lo que me pasaba
    #Entonces creo que con esto ya está la verdad

    if type(labelled_voxels) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_voxels).T
        labelled_voxels = pd.DataFrame(coor, columns = coords + ['energy'] + ['segclass'])

    if type(labelled_hits) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_hits).T
        labelled_hits = pd.DataFrame(coor, columns = coords_mc + [value[2]])

    xcoord  = labelled_voxels[coords[0]].values
    ycoord  = labelled_voxels[coords[1]].values
    zcoord  = labelled_voxels[coords[2]].values
    content = labelled_voxels[value[0]].values

    xmin, xmax = min(xcoord), max(xcoord)
    ymin, ymax = min(ycoord), max(ycoord)
    zmin, zmax = min(zcoord), max(zcoord)

    labels, ticks = plot_label_creator((xmin, ymin, zmin), (xmax, ymax, zmax), voxel_size, affluence)

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2


    mc_label = labelled_voxels[np.isin(labelled_voxels.segclass, (1, 2, 3))]
    cloud    = labelled_voxels[np.isin(labelled_voxels.segclass, (4, 5, 6))]
    ghost    = labelled_voxels[np.isin(labelled_voxels.segclass, 7)]

    #CLOUD
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([cloud[coords[0]].values-xmin,
                          cloud[coords[1]].values-ymin,
                          cloud[coords[2]].values-zmin])
    nonzeros = nonzeros.astype(int)
    xarr[tuple(nonzeros)] = cloud[value[1]].values
    dim     = xarr.shape
    voxels  = xarr > th

    fig  = plt.figure(figsize=(15, 7), frameon=False)
    gs   = fig.add_gridspec(1, 40)
    ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb = fig.add_subplot(gs[0, 24])
    norm = mpl.colors.Normalize(vmin=cloud[value[1]].min(), vmax=cloud[value[1]].max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap[1])

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors * opacity[1], edgecolor=edgecolor, linewidth = linewidth)
    cb_cloud = mpl.colorbar.ColorbarBase(axcb, cmap=cmap[1], norm=norm, orientation='vertical')

    #MC
    xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

    nonzeros = np.vstack([mc_label[coords[0]].values-xmin,
                          mc_label[coords[1]].values-ymin,
                          mc_label[coords[2]].values-zmin])
    nonzeros = nonzeros.astype(int)
    xarr[tuple(nonzeros)] = mc_label[value[0]].values
    dim     = xarr.shape
    voxels  = xarr > th

    #gs   = fig.add_gridspec(2, 40)
    #ax   = fig.add_subplot(gs[0, 0:16], projection = '3d')
    axcb  = fig.add_subplot(gs[0, 21])
    norm = mpl.colors.Normalize(vmin=mc_label[value[0]].min(), vmax=mc_label[value[0]].max())
    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap[0])

    colors = np.asarray(np.vectorize(m.to_rgba)(xarr))
    colors = np.rollaxis(colors, 0, 4)

    ax.voxels(voxels, facecolors=colors * opacity[0], edgecolor=edgecolor, linewidth = linewidth)
    cb_mc = mpl.colorbar.ColorbarBase(axcb, cmap=cmap[0], norm=norm, orientation='vertical')

    #GHOST
    if ghost.empty == False:
        xarr = np.ones(shape=(nbinsX, nbinsY, nbinsZ))*th

        nonzeros = np.vstack([ghost[coords[0]].values-xmin,
                              ghost[coords[1]].values-ymin,
                              ghost[coords[2]].values-zmin])
        nonzeros = nonzeros.astype(int)
        xarr[tuple(nonzeros)] = ghost['segclass'].values
        dim     = xarr.shape
        voxels  = xarr > th

        ax.voxels(voxels, facecolors='g', edgecolor=edgecolor, linewidth = linewidth)
        legend_elements = [Patch(facecolor='g', edgecolor='g', label='ghost class')]
        ax.legend(handles=legend_elements)

    #HITS
    scaled_hits = plot_adaption_hits_to_voxel_scale(labelled_hits, voxel_size, start_bin)

    axcb = fig.add_subplot(gs[0, 18])
    norm = mpl.colors.Normalize(vmin=scaled_hits.loc[:, value[2]].min(), vmax=scaled_hits.loc[:, value[2]].max())

    m    = mpl.cm.ScalarMappable(norm=norm, cmap=cmap[2])

    colors = np.asarray(np.vectorize(m.to_rgba)(scaled_hits.loc[:, value[2]]))
    colors = np.rollaxis(colors, 0, 2)

    ax.scatter(scaled_hits[coords_mc[0]] - xmin, scaled_hits[coords_mc[1]] - ymin, scaled_hits[coords_mc[2]] - zmin, c=colors * opacity[2], marker='o')
    cb_hits = mpl.colorbar.ColorbarBase(axcb, cmap=cmap[2], norm=norm, orientation='vertical')

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')

    ax.set_xticklabels(labels[0])
    ax.set_xticks(ticks[0])
    ax.set_yticklabels(labels[1])
    ax.set_yticks(ticks[1])
    ax.set_zticklabels(labels[2])
    ax.set_zticks(ticks[2])

    if value[1] == 'segclass':
        cb_cloud.set_ticks([4, 5, 6])
        cb_cloud.set_ticklabels([4, 5, 6])
    cb_cloud.set_label('cloud ' + value[1])

    if value[0] == 'segclass':
        cb_mc.set_ticks([1, 2, 3])
        cb_mc.set_ticklabels([1, 2, 3])
    cb_mc.set_label(value[0])

    if value[2] == 'segclass':
        cb_hits.set_ticks([1, 2, 3])
        cb_hits.set_ticklabels([1, 2, 3])
    cb_hits.set_label('hits ' + value[2])

    plt.show()

def plot_cloud_voxels_and_hits_discrete(labelled_voxels, labelled_hits, voxel_size, start_bin, affluence = (5, 5, 5), value = ['segclass', 'segclass', 'segclass'], coords = ['xbin', 'ybin', 'zbin'], coords_mc = ['x', 'y', 'z'], th=0, edgecolor='k', linewidth = .3, cmap = [mpl.cm.coolwarm, mpl.cm.coolwarm, mpl.cm.coolwarm], opacity = [0, 1, 1]):
    '''
    '''

    color_dict = {1:'deepskyblue', 2:'gold', 3:'tab:red', 4:'deepskyblue', 5:'gold', 6:'tab:red', 7:'tab:green'}

    if type(labelled_voxels) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_voxels).T
        labelled_voxels = pd.DataFrame(coor, columns = coords + ['energy'] + ['segclass'])

    if type(labelled_hits) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_hits).T
        labelled_hits = pd.DataFrame(coor, columns = coords_mc + [value[2]])

    xcoord  = labelled_voxels[coords[0]].values
    ycoord  = labelled_voxels[coords[1]].values
    zcoord  = labelled_voxels[coords[2]].values
    content = labelled_voxels[value[0]].values

    xmin, xmax = min(xcoord), max(xcoord)
    ymin, ymax = min(ycoord), max(ycoord)
    zmin, zmax = min(zcoord), max(zcoord)

    labels, ticks = plot_label_creator((xmin, ymin, zmin), (xmax, ymax, zmax), voxel_size, affluence)

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2


    mc_label = labelled_voxels[np.isin(labelled_voxels.segclass, (1, 2, 3))]
    cloud    = labelled_voxels[np.isin(labelled_voxels.segclass, (4, 5, 6))]
    ghost    = labelled_voxels[np.isin(labelled_voxels.segclass, 7)]


    ax  = plt.figure(figsize=(8, 8), frameon=False).add_subplot(projection='3d')

    #CLOUD
    xarr = np.zeros(shape=(nbinsX, nbinsY, nbinsZ), dtype = 'U16')

    nonzeros = np.vstack([cloud[coords[0]].values-xmin,
                          cloud[coords[1]].values-ymin,
                          cloud[coords[2]].values-zmin])

    xarr[tuple(nonzeros)] = cloud[value[1]].map(color_dict).values

    ax.voxels(xarr, facecolors=xarr, edgecolor=edgecolor, linewidth = linewidth, alpha = opacity[1])

    #MC
    xarr = np.zeros(shape=(nbinsX, nbinsY, nbinsZ), dtype = 'U16')

    nonzeros = np.vstack([mc_label[coords[0]].values-xmin,
                          mc_label[coords[1]].values-ymin,
                          mc_label[coords[2]].values-zmin])

    xarr[tuple(nonzeros)] = mc_label[value[0]].map(color_dict).values


    ax.voxels(xarr, facecolors=xarr, edgecolor=edgecolor, linewidth = linewidth, alpha = opacity[0])

    legend_elements = [Patch(facecolor='deepskyblue', label='other class'),
                       Patch(facecolor='gold',        label='track class'),
                       Patch(facecolor='tab:red',     label='blob class')]

    #GHOST
    if ghost.empty == False:
        xarr = np.zeros(shape=(nbinsX, nbinsY, nbinsZ), dtype = 'U16')

        nonzeros = np.vstack([ghost[coords[0]].values-xmin,
                              ghost[coords[1]].values-ymin,
                              ghost[coords[2]].values-zmin])

        xarr[tuple(nonzeros)] = ghost['segclass'].map(color_dict).values

        ax.voxels(xarr, facecolors=xarr, edgecolor=edgecolor, linewidth = linewidth)

        legend_elements.append(Patch(facecolor='tab:green',   label='ghost class'))


    #HITS
    if not labelled_hits.empty:
        scaled_hits = plot_adaption_hits_to_voxel_scale(labelled_hits, voxel_size, start_bin)

        hit_color = scaled_hits[value[2]].map(color_dict)
        ax.scatter(scaled_hits[coords_mc[0]] - xmin, scaled_hits[coords_mc[1]] - ymin, scaled_hits[coords_mc[2]] - zmin, c=hit_color, marker='o', alpha = opacity[2])


    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')



    ax.legend(handles=legend_elements, fontsize=15)

    ax.set_xticklabels(labels[0])
    ax.set_xticks(ticks[0])
    ax.set_yticklabels(labels[1])
    ax.set_yticks(ticks[1])
    ax.set_zticklabels(labels[2])
    ax.set_zticks(ticks[2])


    plt.show()

def plot_cloud_voxels_and_hits_discrete_blobs(labelled_voxels, labelled_hits, event_blobs, voxel_size, start_bin, affluence = (5, 5, 5), value = ['segclass', 'segclass', 'segclass'], coords = ['xbin', 'ybin', 'zbin'], coords_mc = ['x', 'y', 'z'], th=0, edgecolor='k', linewidth = .3, cmap = [mpl.cm.coolwarm, mpl.cm.coolwarm, mpl.cm.coolwarm], opacity = [0.05, 0.05, 1]):
    '''
    Plots also the extremes in the event_blobs DataFrame
    '''

    color_dict = {1:'deepskyblue', 2:'gold', 3:'tab:red', 4:'deepskyblue', 5:'gold', 6:'tab:red', 7:'tab:green'}

    if type(labelled_voxels) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_voxels).T
        labelled_voxels = pd.DataFrame(coor, columns = coords + ['energy'] + ['segclass'])

    if type(labelled_hits) == type(pd.DataFrame()):
        pass
    else:
        coor = np.array(labelled_hits).T
        labelled_hits = pd.DataFrame(coor, columns = coords_mc + [value[2]])

    xcoord  = labelled_voxels[coords[0]].values
    ycoord  = labelled_voxels[coords[1]].values
    zcoord  = labelled_voxels[coords[2]].values
    content = labelled_voxels[value[0]].values

    xmin, xmax = min(xcoord), max(xcoord)
    ymin, ymax = min(ycoord), max(ycoord)
    zmin, zmax = min(zcoord), max(zcoord)

    labels, ticks = plot_label_creator((xmin, ymin, zmin), (xmax, ymax, zmax), voxel_size, affluence)

    nbinsX = int(np.ceil((xmax-xmin))) + 2
    nbinsY = int(np.ceil((ymax-ymin))) + 2
    nbinsZ = int(np.ceil((zmax-zmin))) + 2


    mc_label = labelled_voxels[np.isin(labelled_voxels.segclass, (1, 2, 3))]
    cloud    = labelled_voxels[np.isin(labelled_voxels.segclass, (4, 5, 6))]
    ghost    = labelled_voxels[np.isin(labelled_voxels.segclass, 7)]


    ax  = plt.figure(figsize=(10, 10), frameon=False).add_subplot(projection='3d')

    #CLOUD
    xarr = np.zeros(shape=(nbinsX, nbinsY, nbinsZ), dtype = 'U16')

    nonzeros = np.vstack([cloud[coords[0]].values-xmin,
                          cloud[coords[1]].values-ymin,
                          cloud[coords[2]].values-zmin])

    xarr[tuple(nonzeros)] = cloud[value[1]].map(color_dict).values

    ax.voxels(xarr, facecolors=xarr, edgecolor=edgecolor, linewidth = linewidth, alpha = opacity[1])

    #MC
    xarr = np.zeros(shape=(nbinsX, nbinsY, nbinsZ), dtype = 'U16')

    nonzeros = np.vstack([mc_label[coords[0]].values-xmin,
                          mc_label[coords[1]].values-ymin,
                          mc_label[coords[2]].values-zmin])

    xarr[tuple(nonzeros)] = mc_label[value[0]].map(color_dict).values


    ax.voxels(xarr, facecolors=xarr, edgecolor=edgecolor, linewidth = linewidth, alpha = opacity[0])

    legend_elements = [Patch(facecolor='deepskyblue', label='other class'),
                       Patch(facecolor='gold',        label='track class'),
                       Patch(facecolor='tab:red',     label='blob class')]

    #GHOST
    if ghost.empty == False:
        xarr = np.zeros(shape=(nbinsX, nbinsY, nbinsZ), dtype = 'U16')

        nonzeros = np.vstack([ghost[coords[0]].values-xmin,
                              ghost[coords[1]].values-ymin,
                              ghost[coords[2]].values-zmin])

        xarr[tuple(nonzeros)] = ghost['segclass'].map(color_dict).values

        ax.voxels(xarr, facecolors=xarr, edgecolor=edgecolor, linewidth = linewidth)

        legend_elements.append(Patch(facecolor='tab:green',   label='ghost class'))


    #HITS
    if not labelled_hits.empty:
        scaled_hits = plot_adaption_hits_to_voxel_scale(labelled_hits, voxel_size, start_bin)

        hit_color = scaled_hits[value[2]].map(color_dict)
        ax.scatter(scaled_hits[coords_mc[0]] - xmin, scaled_hits[coords_mc[1]] - ymin, scaled_hits[coords_mc[2]] - zmin, c=hit_color, marker='o', alpha = opacity[2])

    #ISAURA BLOBS
    blob1_names = ['blob1_x', 'blob1_y', 'blob1_z']
    blob2_names = ['blob2_x', 'blob2_y', 'blob2_z']

    def coord_transformer(coor, voxel_size, start_bin):
        new_coor = [(co - star) / siz for co, (siz, star) in zip(coor, zip(voxel_size, start_bin))]
        return new_coor

    blob1 = coord_transformer((event_blobs[blob1_names[0]], event_blobs[blob1_names[1]], event_blobs[blob1_names[2]]), voxel_size, start_bin)
    blob2 = coord_transformer((event_blobs[blob2_names[0]], event_blobs[blob2_names[1]], event_blobs[blob2_names[2]]), voxel_size, start_bin)

    blob1_plot = ax.scatter3D(blob1[0] - xmin, blob1[1] - ymin, blob1[2] - zmin, s = 300, c = 'forestgreen', marker = '*', label = 'isaura blob')
    blob2_plot = ax.scatter3D(blob2[0] - xmin, blob2[1] - ymin, blob2[2] - zmin, s = 300, c = 'forestgreen', marker = '*', label = 'blob2 isaura')

    legend_elements.append(blob1_plot)
    #legend_elements.append(blob2_plot)

    #BARYCENTERS
    label_blob_names = ['barycenter_x', 'barycenter_y', 'barycenter_z']
    start_track_names = ['track_start_x', 'track_start_y', 'track_start_z']
    blob2_color = 'blue'

    main_blob = event_blobs[event_blobs.elem == '3_0']

    labelblob1 = coord_transformer((main_blob[label_blob_names[0]], main_blob[label_blob_names[1]], main_blob[label_blob_names[2]]), voxel_size, start_bin)

    if len(event_blobs) > 1:
        main_blob = event_blobs[event_blobs.elem == '3_1']
        start_track_names = ['barycenter_x', 'barycenter_y', 'barycenter_z']
        blob2_color = 'm'

    labelblob2 = coord_transformer((main_blob[start_track_names[0]], main_blob[start_track_names[1]], main_blob[start_track_names[2]]), voxel_size, start_bin)

    labelblob1_plot = ax.scatter3D(labelblob1[0] - xmin, labelblob1[1] - ymin, labelblob1[2] - zmin, s = 300, c = 'm', marker = 'X', label = 'label blob')
    labelblob2_plot = ax.scatter3D(labelblob2[0] - xmin, labelblob2[1] - ymin, labelblob2[2] - zmin, s = 300, c = blob2_color, marker = 'X', label = 'start track')

    legend_elements.append(labelblob1_plot)
    if (len(event_blobs) == 1) & (event_blobs.binclass.values[0] == 0):
        legend_elements.append(labelblob2_plot)

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')



    ax.legend(handles=legend_elements, fontsize=15)

    ax.set_xticklabels(labels[0])
    ax.set_xticks(ticks[0])
    ax.set_yticklabels(labels[1])
    ax.set_yticks(ticks[1])
    ax.set_zticklabels(labels[2])
    ax.set_zticks(ticks[2])


    plt.show()