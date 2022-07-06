import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import colors
from Dimensions.descriptors import embedding_self_correlation
from utils import loadMKLdata, ordered_positions, MKLoutput_distance, seg_loadMKLdata, int_loadMKLdata
import scipy as sp


# MKL_A...
MKL_runA = 'ST' # OPTIONS: 'main', '3seg', 'P', 'Q', 'T', 'QT', 'ST'
ds3_A = 4

#MKL_B
MKL_runB = 'T' # OPTIONS: 'main', '3seg', 'P', 'Q', 'T', 'QT', 'ST'
ds3_B = 4

# How many dimensions to compare?
dims = 151

for ESC in [0.9, 0.95, None]:

    # MKL Output #1:
    codaA = coda = MKL_runA + '_ds' + str(ds3_A)

    if MKL_runA == '1seg':
        MKLinfoA, MKLinputA, MKLoutputA = loadMKLdata(ds3_index = ds3_A)

    elif MKL_runA in ['3seg', '6seg']:
        MKLinfoA, MKLinputA, MKLoutputA = seg_loadMKLdata(MKL_run = MKL_runA, ds3_index = ds3_A)

    elif MKL_runA in ['P', 'QRS', 'T', 'QT', 'ST']:
        MKLinfoA, MKLinputA, MKLoutputA = int_loadMKLdata(MKL_run = MKL_runA, ds3_index = ds3_A)


    # Get the most representative if ds3 greater than 1
    if ds3_A != 1:

        # Load the MKL_info containing the most representative beat per signal, computed by DS3=1
        BEATon_ds1 = {s['signalname']: s['BEATon'] for s in loadMKLdata(ds3_index=1)[0]}

        # Check if beat in MKLoutput has same BEATon as the most representative beat per signal
        matches = []
        for i, beat in enumerate(MKLinfoA):
            if beat['BEATon'] == BEATon_ds1[beat['signalname']]:
                matches.append(i)
        MKLinfoA = [MKLinfoA[i] for i in matches]
        MKLoutputA = MKLoutputA[matches, :]
        MKLinputA = [s[:, matches] for s in MKLinputA]


    # MKLOutput #2
    codaB = MKL_runB + '_ds' + str(ds3_B)

    if MKL_runB == '1seg':
        MKLinfoB, MKLinputB, MKLoutputB = loadMKLdata(ds3_index = ds3_B)

    elif MKL_runB in ['3seg', '6seg']:
        MKLinfoB, MKLinputB, MKLoutputB = seg_loadMKLdata(MKL_run = MKL_runB, ds3_index = ds3_B)

    elif MKL_runB in ['P', 'QRS', 'T', 'QT', 'ST']:
        MKLinfoB, MKLinputB, MKLoutputB = int_loadMKLdata(MKL_run = MKL_runB, ds3_index = ds3_B)

    # Get the most representative if ds3 greater than 1
    if ds3_B != 1:

        # Load the MKL_info containing the most representative beat per signal, computed by DS3=1
        BEATon_ds1 = {s['signalname']: s['BEATon'] for s in loadMKLdata(ds3_index=1)[0]}

        # Check if beat in MKLoutput has same BEATon as the most representative beat per signal
        matches = []
        for i, beat in enumerate(MKLinfoB):
            if beat['BEATon'] == BEATon_ds1[beat['signalname']]:
                matches.append(i)
        MKLinfoB = [MKLinfoB[i] for i in matches]
        MKLoutputB = MKLoutputB[matches, :]
        MKLinputB = [s[:, matches] for s in MKLinputB]

    #Trim MKLoutput?
    if ESC is None:
        MKLoutputA = MKLoutputA[:,:dims]
        MKLoutputB = MKLoutputB[:,:dims]

    else:
        MKLoutputA = MKLoutputA[:,:embedding_self_correlation(MKLoutputA, correlation_threshold=ESC)[1]]
        MKLoutputB = MKLoutputB[:,:embedding_self_correlation(MKLoutputB, correlation_threshold=ESC)[1]]

    # Compute euclidean distances between ordered sets of MKLoutputs, if not already computed
    distancesA = MKLoutput_distance(MKLoutputA)
    distancesB = MKLoutput_distance(MKLoutputB)

    # Rank the distances from each point
    ordered_positionsA = ordered_positions(distancesA)
    ordered_positionsB = ordered_positions(distancesB)

    # Discard the diagonal values (each point is closest to itself)
    x = ordered_positionsA[~np.eye(ordered_positionsA.shape[0],dtype=bool)].reshape(ordered_positionsA.shape[0],-1).flatten()
    y = ordered_positionsB[~np.eye(ordered_positionsA.shape[0],dtype=bool)].reshape(ordered_positionsA.shape[0],-1).flatten()


    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
    print(MKL_runA + ' vs ' + MKL_runB + ' ;' +  str(ESC) + ' ESC: ' + str(r_value*r_value))

    #Plot that
    fig = plt.figure(figsize=(21, 13.5))
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
    plt.title('Ranked neighbors: ' + codaA + ' vs ' + codaB + '. $R^{2}$=' + str(r_value*r_value), fontsize=25)
    plt.hist2d(x, y, bins=(int(MKLoutputA.shape[0]), int(MKLoutputB.shape[0])), cmap="RdYlGn_r", norm = colors.LogNorm())
    plt.xlabel(codaA, fontsize=20)
    plt.ylabel(codaB, fontsize=20)
    plt.tight_layout()
    plt.show()
