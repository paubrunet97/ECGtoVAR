import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
from utils import load_segmentations, onoff_pairs
from CardioSoftECGXMLReader import CardioSoftECGXMLReader
import matplotlib.patches as mpatches


# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
Pon, Poff, QRSon, QRSoff, Ton, Toff = load_segmentations()
lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Which lead do you wanna plot?
lead ='II'
lead_num = [i for i, elem in enumerate(lead_names) if lead == elem][0]

# For a given time interval
on = 1490
off = 2490

# For a given signal name
plotLegend= True
signal_names = [signal.split('#')[0] for signal in list(Pon.keys())]
signal_indices = [i for i, elem in enumerate(signal_names) if signal == elem]

# Create the legend
pop_a = mpatches.Patch(facecolor='r', alpha=0.15, label='P wave')
pop_b = mpatches.Patch(facecolor='g', alpha=0.15, label='QRS complex')
pop_c = mpatches.Patch(facecolor='b', alpha=0.15, label='T wave')

# Open the Beats Dictionary containing info on all valid beats for a given signal.
BeatsDic = pickle.load(open('_Intermediates/BeatsDic.pckl', 'rb'))

# Loop over all signals, if you want to plot signal 3319_hr -> database.index.tolist()[3318:3319]
for idx in signal_indices:

    signalname = list(Pon.keys())[idx]

    # Load signal
    signal = pd.read_csv(basedir + signalname + '.csv', sep=',', header=None).to_numpy()

    # Computes the on-off fiducial pairs (which off is between two on's?)
    ponoff = onoff_pairs(Pon[signalname], Poff[signalname])
    qrsonoff = onoff_pairs(QRSon[signalname], QRSoff[signalname])
    tonoff = onoff_pairs(Ton[signalname], Toff[signalname])

    # Create a subplot, set its title
    plt.clf()
    plt.figure(figsize=(9, 3), dpi=500)
    plt.plot(signal[:, lead_num], linewidth=0.75)
    if plotLegend:
         plt.legend(handles=[pop_a, pop_b, pop_c])
    plt.title(signalname + '. Filtered lead ' + lead + '.')

    # Plot the fids
    for (pon, poff) in ponoff:
        plt.axvspan(pon, poff, alpha=0.15, color='r')

    for (qrson, qrsoff) in qrsonoff:
        plt.axvspan(qrson, qrsoff, alpha=0.15, color='g')

    for (ton, toff) in tonoff:
        plt.axvspan(ton, toff, alpha=0.15, color='b')

    plt.xlim((on,off))
    plt.ylim((min(signal[on:off, lead_num])-0.2, max(signal[on:off, lead_num])+0.2))

    # Saves the figures to folder figs
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()
    plt.cla()
