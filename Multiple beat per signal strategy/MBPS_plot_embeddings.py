import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from Dimensions.plot import descriptors
from Dimensions.descriptors import get_variability_descriptors, embedding_self_correlation
from utils import loadMKLdata, seg_loadMKLdata, int_loadMKLdata, colored_line
import pickle

#How many dimensions? If you don't want to enforce a dimension, type None.
dimensions = None
correlation_threshold = 0.90

# Which signal dou you wanna plot?
MKL_run = '1seg'
denoised = True
ds3_index = 4

# Plot the pairgrid?
plotPairGrid = False

# Do you wanna plot the variability descriptors?
plotVarDes = False

coda = MKL_run + '_ds' + str(ds3_index)
os.makedirs('Figures/MKLdimensions/' + MKL_run, exist_ok=True)

# Load the MKL data
if MKL_run == '1seg':
    MKLinfo, MKLinput, MKLoutput = loadMKLdata(ds3_index = ds3_index, denoised=denoised)

elif MKL_run in ['3seg', '6seg']:
    MKLinfo, MKLinput, MKLoutput = seg_loadMKLdata(MKL_run = MKL_run, ds3_index = ds3_index, denoised=denoised)

elif MKL_run in ['P', 'QRS', 'T', 'QT', 'ST']:
    MKLinfo, MKLinput, MKLoutput = int_loadMKLdata(MKL_run = MKL_run, ds3_index = ds3_index, denoised=denoised)

signalnames = [s['signalname'] for s in loadMKLdata(ds3_index=1, denoised=denoised)[0]]

lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Take one beat per signal, the most representative one...
# Load the MKL_info containing the most representative beat per signal, computed by DS3=1
BEATon_ds1 = {s['signalname']: s['BEATon'] for s in loadMKLdata(denoised=denoised, ds3_index=1, K_NN=None)[0]}

# Check if beat in MKLoutput has same BEATon as the most representative beat per signal
matches = []
for i, beat in enumerate(MKLinfo):
    if beat['BEATon'] == BEATon_ds1[beat['signalname']]:
        matches.append(i)
MKLinfo = [MKLinfo[i] for i in matches]
MKLoutput = MKLoutput[matches, :]
MKLinput = [s[:, matches] for s in MKLinput]

# Compute the dimension number with x% of info, trim the MKLoutput dimensions.
if dimensions is None:
    corr, dims = embedding_self_correlation(MKLoutput, correlation_threshold=correlation_threshold)
    dims = 3 if dims < 3 else dims
else:
    dims = dimensions
MKLoutput = MKLoutput[:, :dims]
print('DIMS: ' + str(dims))

# Plot the PairGrid
if plotPairGrid:

    df = pd.DataFrame({f"Dim. {d + 1}": MKLoutput[:, d] for d in range(dims)})
    df.index = [s['signalname'] for s in loadMKLdata(denoised=denoised, ds3_index=1, K_NN=None)[0]]

    g = sn.PairGrid(df, height=3.25)
    g.fig.set_size_inches(20.99, 29.7)
    g.map_upper(sn.scatterplot, size=0.75, alpha=0.8)
    g.map_diag(sn.kdeplot, shade=True)
    g.set(xticklabels=[]).set(yticklabels=[])
    g.add_legend()
    [g.axes[i, j].remove() for i, j in zip(*np.tril_indices_from(g.axes, -1))]
    [g.axes[i, i].set_xlabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
    [g.axes[i, i].set_ylabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
    g.fig.subplots_adjust(top=.95)
    g.fig.suptitle(coda, fontsize=20)
    plt.show()
    plt.close()

# Plot the Variability Descriptors
if plotVarDes:

    # Get the leads variablity
    vd = get_variability_descriptors(MKLoutput, MKLinput, dims)
    groupby = 5

    # Load the resampling lengths
    lengths = pickle.load(open('_Intermediates/lengths' + ('_d' if denoised else '') + '.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    # If signals are kept with shared resize lengths:
    if MKL_run not in ['1seg', '3seg']:
        fig, ax = descriptors(vd, groupby, return_axes=True, linewidth=0.5, figsize=(20.99, 29.7),
                              varnames=lead_names)
        fig.align_ylabels(ax[:, 0])
        # fig.suptitle(MKL_run, fontsize=20)
        fig.tight_layout()
        # fig.subplots_adjust(top=0.94)
        fig.savefig('Figures/MKLdimensions/' + MKL_run + '/VarDes_matched' + '.svg', dpi=1000)
        fig.show()

    if MKL_run in ['1seg', '3seg']:
        fig, ax = plt.subplots(18, dims, figsize=(20.99, 29.7), sharey='row')
        for i, des in enumerate(vd[:12]):
            for j, val in enumerate(des.T):
                col = np.round(1 / groupby * (j % groupby), 2)
                ax[i, j // groupby].plot(val, color=[col, 0, 1 - col], linewidth=0.5)
                ax[i, j // groupby].set_xlim([0, val.size - 1])

        for i in range(6):
            for j in range(dims):
                des = vd[12][i, :]
                x = [i for i in range(groupby)]
                y = des[j * groupby:(j + 1) * groupby]
                fig, ax = colored_line(x, y, fig=fig, ax=ax, col=i + 12, row=j, linewidth=0.05)

                # rg = max(dim[0, :]) - min(dim[0, :])
                # ax[i+12, j].set_ylim((min(dim[0, :]) - rg / 10, max(dim[0, :]) + rg / 10))

        ylabels = lead_names + ['P length', 'PQ length', 'QRS length', 'ST length', 'T length', 'TP length']
        [ax[i, 0].set_ylabel(ylabels[i]) for i in range(18)]

        [ax[i, j].set_xticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
        [ax[0, j].set_title(f"Dim. {j + 1}") for j in range(dims)]

        fig.align_ylabels(ax[:, 0])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.01)
        fig.savefig('Figures/MKLdimensions/' + MKL_run + '/VarDes_matched' + '.svg', dpi=1000)
        plt.show()