import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import silhouette_score
from Dimensions.descriptors import get_variability_descriptors, embedding_self_correlation
from sklearn.cluster import KMeans
from utils import loadMKLdata, seg_loadMKLdata, int_loadMKLdata, colored_line

#How many dimensions and clusters? If you don't want to enforce a dimension or k number, type None.
k = None
dimensions = None
correlation_threshold = 0.90

# Which signal dou you wanna plot?
MKL_run = '1seg'
denoised = True
ds3_index = 4

# Plot k-Means results?
plotKMeans = True
plotClusterMeans = True

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

# Run K-Means
if k is not None:
    kmeans = KMeans(n_clusters=k, random_state=56)
    labels = kmeans.fit_predict(MKLoutput[:,:2])

else:
    # Compute the silhouette score to select k clusters:
    silhouettes = np.zeros((2, 17)); silhouette_best = 0
    for k in range(2, 19):
        clusterer = KMeans(n_clusters=k, random_state=5675678)
        cluster_labels = clusterer.fit_predict(MKLoutput[:, :dims])
        silhouette = silhouette_score(MKLoutput[:, :dims], cluster_labels)
        if silhouette > silhouette_best:
            silhouette_best = silhouette
            kmeans = clusterer
            labels = cluster_labels
        silhouettes[:, k - 2] = [k, silhouette]

    k = int(silhouettes[0, np.argmax(silhouettes[1, :])])
    plt.plot(silhouettes[0, :], silhouettes[1, :])
    plt.vlines(k, ymin=0, ymax=max(silhouettes[1, :]) + max(silhouettes[1, :]) / 6, linestyles="dotted")
    plt.show(); plt.close()

print('k: ' + str(k))

# Plot the K-Means PairGrid
if plotKMeans:
    df = pd.DataFrame({f"Dim. {d + 1}": MKLoutput[:, d] for d in range(dims)})
    df["Clusters"] = labels
    hue_order = [0, 1]
    palette = "rainbow"
    g = sn.PairGrid(df, hue="Clusters", hue_order=hue_order, height=3.25, palette=palette)
    g.map_upper(sn.scatterplot, size=0.75, alpha=0.8); g.map_diag(sn.kdeplot, shade=True); g.set(xticklabels=[]).set(yticklabels=[]); g.add_legend();
    [g.axes[i, j].remove() for i, j in zip(*np.tril_indices_from(g.axes, -1))]
    [g.axes[i, i].set_xlabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
    [g.axes[i, i].set_ylabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
    g.fig.subplots_adjust(top=.95)
    g.fig.savefig('Figures/MKLdimensions/' + MKL_run + '/kPairGrid' + '.svg', dpi=1000)
    plt.show(); plt.close()

# Plot the average ECG leads in each cluster:
if plotClusterMeans:

    N = len(MKLinfo)
    Features = MKLinput
    u_clusters = np.unique(labels)

    # Regress in original space
    f, ax = plt.subplots(nrows=13, ncols=u_clusters.size, figsize=(20.99/1.2, 29.7/1.2), num=1, clear=True)
    f.tight_layout()
    f.subplots_adjust(wspace=0.01, hspace=0.01)

    ylims = []
    for i in range(12):
        ylims.append([0, 0])
        for j, ix_cluster in enumerate(u_clusters):
            filt_cluster = (labels == ix_cluster)
            mean_shape = np.mean(Features[i][:, filt_cluster], axis=-1)
            diff_shape = Features[i][:, filt_cluster] - mean_shape[:, None]
            ax[i, ix_cluster].plot(mean_shape)
            ax[i, ix_cluster].fill_between(np.arange(Features[i].shape[0]),
                                           mean_shape + np.percentile(diff_shape, 5, axis=1),
                                           mean_shape + np.percentile(diff_shape, 95, axis=1), alpha=0.25)
            ax[i, ix_cluster].fill_between(np.arange(Features[i].shape[0]),
                                           mean_shape + np.percentile(diff_shape, 25, axis=1),
                                           mean_shape + np.percentile(diff_shape, 75, axis=1), alpha=0.75)
            ylims[-1][0] = min([ylims[-1][0], ax[i, j].get_ylim()[0]])
            ylims[-1][1] = max([ylims[-1][1], ax[i, j].get_ylim()[1]])

    # Assuming you have time as a vector of 6 elements
    segment_names = ['P', 'PQ', 'QRS', 'ST',  'T', 'TP']
    lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'lengths (ms)']
    for j, ix_cluster in enumerate(u_clusters):
        filt_cluster = (labels == ix_cluster)
        ax[-1, j].boxplot(Features[-1][:, filt_cluster].tolist())
        ax[-1, j].set_xticklabels(segment_names)
        [ax[i, 0].set_ylabel(list(lead_names)[i]) for i in range(13)]

    # Prettify - this could have been done much more efficiently
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            if j != 0:
                ax[i, j].set_yticks([])
            if i != 12:
                ax[i, j].set_xticks([])

            if i != 12:
                lmin, lmax = np.ceil(ylims[i][0] * 2) / 2, np.floor(ylims[i][1] * 2) / 2
                for c in np.arange(lmin, lmax + 0.5, 0.5):
                    ax[i, j].axhline(c, linewidth=0.75, color='red', alpha=0.5)
                for c in np.arange(0, Features[i].shape[0], 100):
                    ax[i, j].axvline(c, linewidth=0.75, color='red', alpha=0.5)

            ax[i, j].set_xlim([0, Features[i].shape[0]])
            if i != 12:
                ax[i, j].set_ylim(ylims[i])
            else:
                ax[i, j].set_xlim([0, 7])

            if i == 0:
                ax[i, j].set_title('Cluster ' + str(j))

                # Set background color
                ax[i, j].patch.set_facecolor('red')
                ax[i, j].patch.set_alpha(0.01)

    f.align_ylabels(ax[:, 0])
    plt.tight_layout()
    plt.show()