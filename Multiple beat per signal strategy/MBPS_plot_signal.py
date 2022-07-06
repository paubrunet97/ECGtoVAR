import pickle
import scipy.io as spio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from utils import load_segmentations, onoff_pairs, matobj_to_dict

basedir = "Directory with the denoised signals"

# Do you want to plot BeatOn?
plotBeatON = True

# Do you want to plot the representative beats?
plotBatchBeats = True
plotIncrementalBeats = True

# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
Pon, Poff, QRSon, QRSoff, Ton, Toff = load_segmentations()

# Create the legend
pop_a = mpatches.Patch(facecolor='r', alpha=0.15, label='P wave')
pop_b = mpatches.Patch(facecolor='g', alpha=0.15, label='QRS complex' )
pop_c = mpatches.Patch(facecolor='b', alpha=0.15, label='T wave')

# Open the Beats Dictionary containing info on all valid beats for a given signal.
BeatsDic = pickle.load(open('_Intermediates/BeatsDic.pckl', 'rb'))
AcceptableBeatsON = dict.fromkeys(BeatsDic.keys())
for signalname in BeatsDic:
    AcceptableBeatsON[signalname] = [int(key) for key in list(BeatsDic[signalname].keys())]

# Open the RepresentativeBeats List.
if plotBatchBeats:
    BatchBeatsON = {s:[] for s in AcceptableBeatsON}
    batch_indices = matobj_to_dict(spio.loadmat('4-MBPS_subset_selection/_Intermediates/ds3_indices1.mat', squeeze_me=True, struct_as_record=False)['python_indices'])
    for signalname in AcceptableBeatsON:
        for idx in batch_indices[signalname]:
            BatchBeatsON[signalname].append(AcceptableBeatsON[signalname][idx])
    pop_batch = mpatches.Patch(color='gold', alpha=1, hatch = '///', edgecolor= 'red', label='Most representative beat')

if plotIncrementalBeats:
    IncrementalBeatsON = {s: [] for s in AcceptableBeatsON}
    incremental_indices = matobj_to_dict(spio.loadmat('4-MBPS_subset_selection/_Intermediates/ds3_indices4.mat', squeeze_me=True, struct_as_record=False)['python_indices'])
    for signalname in AcceptableBeatsON:
        for idx in incremental_indices[signalname]:
            IncrementalBeatsON[signalname].append(AcceptableBeatsON[signalname][idx])
    pop_incremental = mpatches.Patch(color='gold', alpha=0.3, label='Morphologically-rich beats')


# Loop over all signals
for signalname in tqdm(list(Pon)):
    print(signalname)
    # Load signal
    signal = pd.read_csv(basedir + signalname + '.csv', sep=',', header=None).to_numpy()
    lead_names = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Computes the on-off fiducial pairs (which off is between two on's?)
    ponoff = onoff_pairs(Pon[signalname], Poff[signalname])
    qrsonoff = onoff_pairs(QRSon[signalname], QRSoff[signalname])
    tonoff = onoff_pairs(Ton[signalname], Toff[signalname])

    # Create a subplot, set its title
    fig, axs = plt.subplots(4, 3, figsize=(29.7/1.4, 20.99/1.4))
    fig.legend(handles=[pop_a, pop_b, pop_c])
    fig.suptitle(signalname, size=20)

    for lead in range(0,12):

        #Set subplots titles with leads
        axs[lead//3, lead%3].set_title(lead_names[lead])

        #Plot the signal lines
        axs[lead//3, lead%3].plot(signal[:, lead], color='b', linewidth=0.75)

        # Plot the fids
        for (pon, poff) in ponoff:
            axs[lead//3, lead%3].axvspan(pon, poff, alpha=0.15, color='r')

        for (qrson, qrsoff) in qrsonoff:
            axs[lead//3, lead%3].axvspan(qrson, qrsoff, alpha=0.15, color='g')

        for (ton, toff) in tonoff:
            axs[lead//3, lead%3].axvspan(ton, toff, alpha=0.15, color='b')

        # Plot the BeatON
        if plotBeatON and signalname in BeatsDic:
            for beat in [int(key) for key in list(BeatsDic[signalname].keys())]:
                axs[lead//3, lead%3].vlines(
                    x=beat, ymin=signal[:, lead].min() - 0.2, ymax=signal[:, lead].max() + 0.2,
                    linestyles="dotted", colors="r", linewidth=0.8)

        # Plot the batch beats
        if plotBatchBeats and signalname in BatchBeatsON:
            for beat in BatchBeatsON[signalname]:
                axs[lead//3, lead%3].axvspan(
                    int(BeatsDic[signalname][str(beat)]['BEATon']), int(BeatsDic[signalname][str(beat)]['BEAToff']),
                    alpha=0.5, hatch='///', edgecolor='red', facecolor='gold')

        # Plot the incremental beats
        if plotIncrementalBeats and signalname in IncrementalBeatsON:
            for beat in IncrementalBeatsON[signalname]:
                axs[lead//3, lead%3].axvspan(
                    int(BeatsDic[signalname][str(beat)]['BEATon']), int(BeatsDic[signalname][str(beat)]['BEAToff']),
                    alpha=0.3, facecolor='gold')

        if plotBatchBeats or plotIncrementalBeats:
            fig.legend(handles=[pop_batch, pop_incremental], loc='upper left')


    # Saves the figures to folder figs
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()
    plt.cla()
