import numpy as np
import scipy as sp
from utils import loadMKLdata, seg_loadMKLdata, int_loadMKLdata

''' For ds3 runs with multiple beats per signal, checks if the signals n-th closer to the centroid of the signal's 
embedding is the representative one '''

# Which signal dou you wanna plot?
MKL_run = 'ST'
ds3_index = 4

coda = MKL_run+ '_ds' + str(ds3_index)

# Load the MKL data
if MKL_run == '1seg':
    MKLinfo, MKLinput, MKLoutput = loadMKLdata(ds3_index = ds3_index)

elif MKL_run in ['3seg', '6seg']:
    MKLinfo, MKLinput, MKLoutput = seg_loadMKLdata(MKL_run = MKL_run, ds3_index = ds3_index)

elif MKL_run in ['P', 'QRS', 'T', 'QT', 'ST']:
    MKLinfo, MKLinput, MKLoutput = int_loadMKLdata(MKL_run = MKL_run, ds3_index = ds3_index)

# Compute the dimension number with n% of info, trim the MKLoutput dimensions.
#dims = embedding_self_correlation(MKLoutput, correlation_threshold=0.9)[1]
#dims = 3 if dims < 3 else dims
dims = 6
MKLoutput = MKLoutput[:,:dims]

print('DIMS: ' + str(dims) + '\n')

# List signalnames and beatons in the dataset
signalnames = [s['signalname'] for s in loadMKLdata(ds3_index=1)[0]]
BEATon_ds1 = {s['signalname']: s['BEATon'] for s in loadMKLdata(ds3_index=1)[0]}

# Create a dictionary in which all the output points for all the beats of a same signal are listed.
beats_in_signal = {signalname:[] for signalname in signalnames}
center_of_mass = {signalname:[] for signalname in signalnames}
for i, beat in enumerate(MKLinfo):
    beats_in_signal[beat['signalname']].append(MKLoutput[i,:])

# Compute the center of mass of the output points for all the beats of a same signal.
center_of_mass = {signalname:[] for signalname in signalnames}
for signalname in beats_in_signal:
    center_of_mass[signalname] = np.average(beats_in_signal[signalname], axis=0)

# Check on which index of beats_in_signal the representative beat is
representative_idx = {signalname:[] for signalname in signalnames}
for i, beat in enumerate(MKLinfo):
    if beat['BEATon'] == BEATon_ds1[beat['signalname']]:
        representative_idx[beat['signalname']].append(True)
    else:
        representative_idx[beat['signalname']].append(False)

for signalname in representative_idx:
    representative_idx[signalname] = [i for i, x in enumerate(representative_idx[signalname]) if x][0]

# Compute the Euclidean Distances between each signal's beat and CoG, rank the distances
distance_to_CoG = {signalname:[] for signalname in signalnames}
ranking_to_CoG = {signalname:[] for signalname in signalnames}
for signalname in beats_in_signal:
    distance_to_CoG[signalname] = np.linalg.norm(beats_in_signal[signalname] - center_of_mass[signalname], axis=1)
    ranking_to_CoG[signalname] = sp.stats.rankdata(distance_to_CoG[signalname])

# Which ranking has the most representative?
most_representative_ranking = []
for signalname in ranking_to_CoG:
    most_representative_ranking.append(int(ranking_to_CoG[signalname][representative_idx[signalname]]))
    if int(ranking_to_CoG[signalname][representative_idx[signalname]]) == 6:
        print(signalname)

values, counts = np.unique(most_representative_ranking, return_counts=True)
for i in range(len(values)):
    print('Most representative in ranking ' + str(values[i]) + ': ' + str(counts[i]))

