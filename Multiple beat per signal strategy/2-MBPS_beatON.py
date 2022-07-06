import csv
from tqdm import tqdm
from utils import load_segmentations


# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
Pon, Poff, QRSon, QRSoff, Ton, Toff = load_segmentations()

# Creates a dictionary with beats for each signal, by looking if there is any Pon between two QRSon:
# a) There are no Pon: Use the second QRSon value.
# b) There is one Pon: Use that value.
# c) There are multiple Pon: Use the last Pon value.

BEATon = {}
for signalname in tqdm(QRSon):
    BEATon[signalname] = []

    for qrson_fid1, qrson_fid2 in zip(QRSon[signalname][:-1], QRSon[signalname][1:]):

        correspondence_p = (Pon[signalname] > qrson_fid1) & (Pon[signalname] < qrson_fid2)
        if sum(correspondence_p) == 1:
            BEATon[signalname].append(int(Pon[signalname][correspondence_p]))

        elif sum(correspondence_p) > 1:
            BEATon[signalname].append(int(Pon[signalname][correspondence_p][-1]))

    # Ckeck that the first and last Pon have been added to the BeatON list.
    if len(Pon[signalname]) != 0:
        if int(Pon[signalname][0]) not in BEATon[signalname]:
            BEATon[signalname].insert(0, int(Pon[signalname][0]))
        if int(Pon[signalname][-1]) not in BEATon[signalname]:
            BEATon[signalname].append(int(Pon[signalname][-1]))

# Save teh BeatON fids as a csv file
with open('Fiducials/BEATon.csv', 'w', newline='') as f:
    csv = csv.writer(f)
    for key in sorted(BEATon.keys()):
        csv.writerow([key] + BEATon[key])


