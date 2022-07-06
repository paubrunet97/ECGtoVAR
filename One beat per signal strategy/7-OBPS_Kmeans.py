import pickle
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from Dimensions.descriptors import embedding_self_correlation, get_variability_descriptors
from Dimensions.plot import descriptors
from itertools import compress
from tqdm import tqdm
from utils import loadMKLdata, seg3_loadMKLdata, seg6_loadMKLdata, int_loadMKLdata, compute_axis

# Which parameters will we use for generating a kmeans:
MKL_run = 'T' #Options: '1seg', '3seg', 'P', 'QRS', 'QT', 'ST', 'T'
outlier_detection = False
plotClusterMeans= True
plotBoxPlots = True
plotFeatures = False

# Load the MKL data
if MKL_run == '1seg':
    MKLinfo, MKLinput, MKLoutput = loadMKLdata()

elif MKL_run == '3seg':
    MKLinfo, MKLinput, MKLoutput = seg3_loadMKLdata()

elif MKL_run == '6seg':
    MKLinfo, MKLinput, MKLoutput = seg6_loadMKLdata()

elif MKL_run in ['P', 'QRS', 'T', 'QT', 'ST']:
    MKLinfo, MKLinput, MKLoutput = int_loadMKLdata(int_name=MKL_run)

# Compute the relevant dimensions
corr, dims = embedding_self_correlation(MKLoutput, correlation_threshold=0.90)
dims = 3 if dims < 3 else dims
print('DIMS: ' + str(dims))

# Open the lenghths dictionary from the resampled beats
P_length, PQ_length, Q_length, ST_length, T_length, TP_length = pickle.load(open('_Intermediates/lengths_d.pckl', 'rb'))

# Compute fiducials
P_on = 0; P_off = P_on + P_length; Q_on = P_off + PQ_length
Q_off = Q_on + Q_length; T_on = Q_off + ST_length; T_off = T_on + T_length

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

# Compute the silhouette score to select k clusters:
silhouettes = np.zeros((2, 18)); silhouette_best = 0

for k in range(2, 20):
    clusterer = KMeans(n_clusters=k, random_state=1)
    cluster_labels = clusterer.fit_predict(MKLoutput[:, :dims])
    silhouette = silhouette_score(MKLoutput[:, :dims], cluster_labels)
    if silhouette > silhouette_best:
        silhouette_best = silhouette
        kmeans = clusterer
        labels = cluster_labels
    silhouettes[:, k-2] = [k, silhouette]

n_clusters = int(silhouettes[0, np.argmax(silhouettes[1,:])])


# Plot the silhouette results
plt.plot(silhouettes[0, :], silhouettes[1, :])
plt.vlines(n_clusters, ymin = 0, ymax = max(silhouettes[1,:]) + max(silhouettes[1,:])/6, linestyles="dotted")
plt.xlabel('Number of clusters')
plt.ylabel('Average silhouette width')
plt.title('Optimal number of clusters')
plt.xticks([i for i in range(2, 20)])
plt.show(); plt.close()
print('N_CLUSTERS: ' + str(n_clusters))

# Plot the average ECG leads in each cluster:
if plotClusterMeans:

    N = len(MKLinfo)
    clusters = kmeans.labels_
    if outlier_detection:
        lens = loadMKLdata()[1][12]
        lens = np.delete(lens, list(to_discard), axis=1)
        Features = MKLinput[:12] + [lens]
    else:
        Features = MKLinput[:12] + [loadMKLdata()[1][12]]
    u_clusters = np.unique(clusters)

    # Regress in original space
    f, ax = plt.subplots(nrows=13, ncols=u_clusters.size, figsize=(20.99/1.2, 29.7/1.2), num=1, clear=True)
    f.tight_layout()
    f.subplots_adjust(wspace=0.01, hspace=0.01)

    ylims = []
    for i in range(12):
        ylims.append([0, 0])
        for j, ix_cluster in enumerate(u_clusters):
            filt_cluster = (clusters == ix_cluster)
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
        filt_cluster = (clusters == ix_cluster)
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

# Plot the results:
dataframe = pd.DataFrame({f"Dim. {d + 1}": MKLoutput[:, d] for d in range(dims)})
dataframe["Clusters"] = labels
hue_order = [False, True]
palette = "rainbow"
g = sn.PairGrid(dataframe, hue="Clusters", hue_order=hue_order, height=3.25, palette=palette)
g.map_upper(sn.scatterplot, size=0.75, alpha=0.8); g.map_diag(sn.kdeplot, shade=True); g.set(xticklabels=[]).set(yticklabels=[]); g.add_legend();
[g.axes[i, j].remove() for i, j in zip(*np.tril_indices_from(g.axes, -1))]
[g.axes[i, i].set_xlabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
[g.axes[i, i].set_ylabel(f"Dim. {i + 1}") for i in range(g.axes.shape[0])]
plt.show(); plt.close()

# Make subsets according to clusters
K_means_clusters={'Cluster ' + str(i): {} for i in range(n_clusters)}
K_means_clusters['kmeans'] = kmeans

vd_list=[]
for i in tqdm(range(n_clusters)):
    in_cluster = labels == i
    K_means_clusters['Cluster ' + str(i)]['MKLoutput'] = MKLoutput[in_cluster, :]
    K_means_clusters['Cluster ' + str(i)]['MKLinput_list'] = [feature[:, in_cluster] for feature in MKLinput]
    K_means_clusters['Cluster ' + str(i)]['MKL_info'] = list(compress(MKLinfo, in_cluster))

# Save the clustered MKLouput and features in a pickle
pickle.dump(K_means_clusters, open('_Intermediates/MKLclusters_' + MKL_run + '.pckl', 'wb'))

if plotBoxPlots:

    # Load the resampling lengths
    lengths = pickle.load(open('_Intermediates/lengths_d.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    # Open the dictionary with the RR lengths, compute the cQT lengths
    RR_len = pickle.load(open('_Intermediates/Fiducials/HeartRate_d.pckl', 'rb'))
    QTc_len = {}; QRS_angles = {}
    if MKL_run in ['1seg', '3seg', '6seg']:
        for beat in MKLinfo:
            QTc_len[beat['signalname']] = 1000 * (sum(beat['lengths'][2:5])/1000) / np.sqrt(sum(beat['lengths'])/1000)
            QRS_angles[beat['signalname']] = [compute_axis(s['resampled_beat'][Q_on:Q_off]) for s in MKLinfo]

    elif MKL_run in ['P', 'QRS', 'T', 'QT', 'ST']:
        for beat in loadMKLdata()[0]:
            QTc_len[beat['signalname']] = 1000 * (sum(beat['lengths'][2:5])/1000) / np.sqrt(sum(beat['lengths'])/1000)
            QRS_angles[beat['signalname']] = [compute_axis(s['resampled_beat'][Q_on:Q_off]) for s in MKLinfo]

    df = pd.DataFrame(list(zip(labels, QTc_len.values(), QRS_angles.values())), columns=['Cluster', 'QTc length (ms)', 'QRS angle'],
                      index=[s['signalname'] for s in MKLinfo])

    # Boxplots of QTc lengths vs cluster
    sns.set_theme(style="whitegrid")
    if n_clusters == 3:
        palette = ["#9200F7", "#82FFB9", "#FE0024"]
    elif n_clusters ==4:
        palette = ["#9200F7", "#6ad9db", "#d7dc8b", "#FE0024"]
    else:
        palette = 'rainbow'
    sns.set(rc={'figure.figsize': (22.7/2, 9.7/2)})
    sns.set_style("whitegrid")
    g = sn.boxplot(data=df, x="Cluster", y='QTc length (ms)', palette=palette)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.show()




