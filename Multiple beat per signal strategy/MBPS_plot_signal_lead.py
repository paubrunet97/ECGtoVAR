import matplotlib.pyplot as plt
import pickle
import pandas as pd
from utils import load_segmentations, onoff_pairs
import matplotlib.patches as mpatches

# What do you wanna plot?
signal = 'Adol70'
lead= 'I'
on = 12800
off = on + 1000
denoised = True
raw = False
fids = True
plotLegend= True

basedir = "Directory with the denoised signals"

# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
if fids:
    Pon, Poff, QRSon, QRSoff, Ton, Toff = load_segmentations()

lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead_num = [i for i, elem in enumerate(lead_names) if lead == elem][0]

# Create the legend
pop_a = mpatches.Patch(facecolor='r', alpha=0.15, label='P wave')
pop_b = mpatches.Patch(facecolor='g', alpha=0.15, label='QRS complex')
pop_c = mpatches.Patch(facecolor='b', alpha=0.15, label='T wave')

# Open the Beats Dictionary containing info on all valid beats for a given signal.
f = open('_Intermediates/BeatsDic.pckl', 'rb')
BeatsDic = pickle.load(f)
f.close()


# Computes the on-off fiducial pairs (which off is between two on's?)
ponoff = onoff_pairs(Pon[signal], Poff[signal])
qrsonoff = onoff_pairs(QRSon[signal], QRSoff[signal])
tonoff = onoff_pairs(Ton[signal], Toff[signal])

# Create a plot, set its title
plt.figure(figsize=(9, 3), dpi=500)
signal = pd.read_csv(basedir + signal + '.csv', sep=',', header=None).to_numpy()
plt.plot(signal[:, lead_num], linewidth=0.75)

if plotLegend:
    plt.legend(handles=[pop_a, pop_b, pop_c])
plt.title(signal + '. Filtered lead ' + lead + '.')

# Plot the fids
if fids:
    for (pon, poff) in ponoff:
        plt.axvspan(pon, poff, alpha=0.15, color='r')

    for (qrson, qrsoff) in qrsonoff:
        plt.axvspan(qrson, qrsoff, alpha=0.15, color='g')

    for (ton, toff) in tonoff:
        plt.axvspan(ton, toff, alpha=0.15, color='b')


# Saves the figures to folder figs
plt.tight_layout()
plt.show()
plt.close()
plt.clf()
plt.cla()
