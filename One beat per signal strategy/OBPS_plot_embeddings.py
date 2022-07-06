import numpy as np
import seaborn as sn
import pickle
import pandas as pd
from utils import loadMKLdata, seg3_loadMKLdata, seg6_loadMKLdata, int_loadMKLdata, colored_line
from matplotlib import pyplot as plt
from Dimensions.descriptors import get_variability_descriptors, embedding_self_correlation
from Dimensions.plot import descriptors
import sak

MKL_run = '1seg' #Options: '1seg', '3seg', 'P', 'QRS', 'QT', 'ST', 'T'
outlier_detection = False

# Plot the variability descriptors?
plotPairGrid = True
plotVarDes = True

# Define some parameters
lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Load the MKL data
if MKL_run == '1seg':
    MKLinfo, MKLinput, MKLoutput = loadMKLdata()

elif MKL_run == '3seg':
    MKLinfo, MKLinput, MKLoutput = seg3_loadMKLdata()

elif MKL_run == '6seg':
    MKLinfo, MKLinput, MKLoutput = seg6_loadMKLdata()

elif MKL_run in ['P', 'PQ', 'QRS', 'T', 'QT', 'ST']:
    MKLinfo, MKLinput, MKLoutput = int_loadMKLdata(int_name=MKL_run)

# Compute the dimension number with 95% of info, trim the MKLoutput dimensions.
corr, dims = embedding_self_correlation(MKLoutput, correlation_threshold=0.95)
dims = 3 if dims < 3 else dims
MKLoutput = MKLoutput[:,:dims]
print('DIMS: ' + str(dims))

# If there is a HUGE outlier, discard it:
if outlier_detection:
    to_discard = set([])
    for dim in range(dims):
        mean = np.mean(MKLoutput[:,dim])
        std = np.std(MKLoutput[:,dim])
        outliers = (MKLoutput[:,dim] < mean-4*std) | (MKLoutput[:,dim] > mean+4*std)
        if sum(outliers) != 0:
            outliers = np.where(outliers)[0].tolist()
            for outlier in outliers:
                to_discard.add(outlier)
            print(dim, outliers)

    MKLoutput = np.delete(MKLoutput, list(to_discard), axis =0)
    for idx in to_discard:
        del MKLinfo[idx]
    for i, feature in enumerate(MKLinput):
        MKLinput[i] = np.delete(feature, list(to_discard), axis =1)

# Load the resampling lengths
lengths = pickle.load(open('_Intermediates/lengths.pckl', 'rb'))
lengths = [0] + [sum(lengths[:i+1]) for i in range(len(lengths))]


# Plot the PairGrid
if plotPairGrid:

    df = pd.DataFrame({f"Dim. {d + 1}": MKLoutput[:, d] for d in range(dims)})
    df.index = [s['signalname'] for s in MKLinfo]

    g = sn.PairGrid(df, height=3.25)
    g.fig.set_size_inches(20.99, 29.7)
    g.map_upper(sn.scatterplot, size=0.3, alpha=0.8)
    g.map_diag(sn.kdeplot, shade=True)
    g.set(xticklabels=[]).set(yticklabels=[])
    [g.axes[i, j].remove() for i, j in zip(*np.tril_indices_from(g.axes, -1))]
    [g.axes[i, i].set_xlabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
    [g.axes[i, i].set_ylabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
    plt.show()
    plt.close()

# Plot the Variablity Descriptors
if plotVarDes:

    # Get the leads variablity
    vd = get_variability_descriptors(MKLoutput, MKLinput, dims, NN_dims=20)
    groupby = 5

    # If signals are kept with shared resize lengths:
    if MKL_run not in ['1seg', '3seg']:
        fig, ax = descriptors(vd, groupby, return_axes=True, linewidth=0.5, figsize=(20.99, 29.7), varnames=lead_names)
        fig.align_ylabels(ax[:, 0])
        fig.tight_layout()
        fig.savefig('Figures/MKLoutputs/' + MKL_run + '/VarDes' + '.svg', dpi=1000)
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
                des= vd[12][i,:]
                x = [i for i in range(groupby)]
                y = des[j * groupby:(j + 1) * groupby]
                fig, ax = colored_line(x, y, fig=fig, ax=ax, col=i+12, row=j, linewidth=0.05)

        ylabels = lead_names + ['P length', 'PQ length', 'QRS length', 'ST length', 'T length', 'TP length']
        [ax[i, 0].set_ylabel(ylabels[i]) for i in range(18)]

        [ax[i, j].set_xticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
        [ax[0, j].set_title(f"Dim. {j + 1}") for j in range(dims)]

        fig.align_ylabels(ax[:, 0])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.01)
        plt.show()