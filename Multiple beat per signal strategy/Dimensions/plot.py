from typing import List, Union, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


def descriptors(descriptors: List[np.ndarray], groupby: int, figsize: Tuple[int,int] = (16,16), varnames: Union[np.ndarray, list] = None, dimnames: Union[np.ndarray, list] = None, same_scale: bool = True, return_axes: bool = False, grid: bool = True, linewidth = 0.75, **kwargs):

    # Nasty fix for descriptors with dim 1
    for i, descriptor in enumerate(descriptors):
        if descriptor.shape[0] == 1:
            descriptors[i] = np.tile(descriptor, (2, 1))

    # Rest
    n_features = len(descriptors)
    n_dimensions = descriptors[0].shape[1]//groupby

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize,**kwargs)
    if ax.ndim == 1:
        ax = ax[:,None]
    for i,des in enumerate(descriptors):
        for j,val in enumerate(des.T):
            col = np.round(1/groupby*(j%groupby),2)
            ax[i,j//groupby].plot(val,color=[col,0,1-col], linewidth=linewidth)
            ax[i,j//groupby].set_xlim([0,val.size-1])
        
    # Set figure options
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]-1) for j in range(ax.shape[1])]
    [ax[i,j].set_yticks([]) for i in range(ax.shape[0])   for j in range(1,ax.shape[1])]
    if varnames is not None:
        [ax[i,0].set_ylabel(f"{varnames[i]}") for i in range(ax.shape[0])]
    else:
        [ax[i,0].set_ylabel(f"Feat. {i+1}") for i in range(ax.shape[0])]
    if dimnames is not None:
        [ax[0,j].set_title(f"{dimnames[j]}") for j in range(ax.shape[1])]
    else:
        [ax[0,j].set_title(f"Dim. {j+1}") for j in range(ax.shape[1])]
    # Set figure ylims
    if same_scale and "sharey" not in kwargs:
        for i in range(ax.shape[0]):
            ylims = [0,0]
            for j in range(ax.shape[1]):
                ylim = ax[i,j].get_ylim()
                ylims[0] = min([ylims[0],ylim[0]])
                ylims[1] = max([ylims[1],ylim[1]])
            
            for j in range(ax.shape[1]):
                ax[i,j].set_ylim(ylims)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    fig.align_ylabels(ax[:,0])

    if return_axes:
        return fig,ax


def path(descriptors: List[np.ndarray], figsize=(16,16), varnames: Union[np.ndarray, list] = None, dimnames: Union[np.ndarray, list] = None):
    # Rest
    n_features = len(descriptors)
    n_dimensions = descriptors[0].shape[1]

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize)
    for i,des in enumerate(descriptors):
        for j,val in enumerate(des.T):
            col = 1/5*(j%5)
            ax[i,j].plot(val,color=[col,0,1-col])
        
    # Set figure options
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]-1) for j in range(ax.shape[1])]
    if varnames is not None:
        [ax[i,0].set_ylabel(f"{varnames[i]}") for i in range(ax.shape[0])]
    else:
        [ax[i,0].set_ylabel(f"Feat. {i+1}") for i in range(ax.shape[0])]
    if dimnames is not None:
        [ax[0,j].set_title(f"Point {dimnames[j]}") for j in range(ax.shape[1])]
    else:
        [ax[0,j].set_title(f"Point {j+1}") for j in range(ax.shape[1])]
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    [ax[i,j].set_yticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01,wspace=0.01)
    fig.align_ylabels(ax[:,0])



