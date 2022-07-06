import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
from utils import load_segmentations, onoff_pairs
from CardioSoftECGXMLReader import CardioSoftECGXMLReader
import matplotlib.patches as mpatches

basedir = 'Directory with filtered signals'

# Do you want to plot BeatOn?
plotBeatON = True

# Do you want to plot BestBeat?
plotBatchBeats = True

# Do you want to plot the RepresentativeBeats?
plotIncrementalBeats = False

# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
Pon, Poff, QRSon, QRSoff, Ton, Toff = load_segmentations(denoised=denoised)

# Create the legend
pop_a = mpatches.Patch(facecolor='r', alpha=0.15, label='P wave')
pop_b = mpatches.Patch(facecolor='g', alpha=0.15, label='QRS complex' )
pop_c = mpatches.Patch(facecolor='b', alpha=0.15, label='T wave')

# Open the Beats Dictionary containing info on all valid beats for a given signal.
BeatsDic = pickle.load(open('_Intermediates/BeatsDic.pckl', 'rb'))

# Open the BatchBeats Dictionary.
if plotBatchBeats:
    BatchBeats = pickle.load(open('7-mainMKL/MKL_info.pckl', 'rb'))
    BatchBeats = {s['signalname']: s for s in BatchBeats}
    pop_batch = mpatches.Patch(color='gold', alpha=1, hatch = '///', edgecolor= 'red', label='Most representative beat')

# Loop over all signals
for signalname in tqdm(list(Pon)[:]):
    print(signalname)
    # Load signal
    signal = pd.read_csv(basedir + signalname + '.csv', sep=',', header=None).to_numpy()

    # Computes the on-off fiducial pairs (which off is between two on's?)
    ponoff = onoff_pairs(Pon[signalname], Poff[signalname])
    qrsonoff = onoff_pairs(QRSon[signalname], QRSoff[signalname])
    tonoff = onoff_pairs(Ton[signalname], Toff[signalname])

    # Create a subplot, set its title
    fig, axs = plt.subplots(4, 3, figsize=(29.7/1.4, 20.99/1.4)) #(29.7/1.4, 20.99/1.4)
    fig.legend(handles=[pop_a, pop_b, pop_c])
    if denoised:
        fig.suptitle(signalname + '. Filtered.')
    else:
        fig.suptitle(signalname + '. Raw.')

    for lead in range(0,12):
        #Set subplots titles with leads
        axs[lead//3, lead%3].set_title(file.Leads[lead])

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
                    linestyles="dotted", colors="r", linewidth=0.8
                )

        # Plot the batch beat
        if plotBatchBeats and signalname in BatchBeats:
            axs[lead//3, lead%3].axvspan(
                int(BatchBeats[signalname]['BEATon']), int(BatchBeats[signalname]['BEAToff']),
                alpha=0.5, facecolor='gold', edgecolor='r', hatch='///'
            )

        if plotBatchBeats:
            fig.legend(handles=[pop_batch], loc=2)

    # Saves the figures to folder figs
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()
    plt.cla()