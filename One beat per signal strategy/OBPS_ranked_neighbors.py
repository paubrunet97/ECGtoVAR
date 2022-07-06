import os
import pickle
import numpy as np
import scipy.io as spio
import scipy.stats as ss
from tqdm import tqdm
from collections import Counter
from matplotlib import colors
import matplotlib.pyplot as plt
from Dimensions.descriptors import embedding_self_correlation
from utils import loadMKLdata, seg3_loadMKLdata, seg6_loadMKLdata, int_loadMKLdata, ordered_positions, MKLoutput_distance
import scipy as sp

denoised = True
ESC = 0.95

# MKL_A...
MKL_runA = '1seg' # OPTIONS: 'main', '3seg', 'P', 'Q', 'T', 'QT', 'ST'

#MKL_B
MKL_runB = 'QRS'  # OPTIONS: 'main', '3seg', 'P', 'Q', 'ST', 'QT', 'T'

# How many dimensions to compare?
dims = 786

# MKL Output #1:
if MKL_runA == '1seg':
    MKLinfoA, MKLinputA, MKLoutputA = loadMKLdata()

elif MKL_runA == '3seg':
    MKLinfoA, MKLinputA, MKLoutputA = seg3_loadMKLdata()

elif MKL_runA == '6seg':
    MKLinfoA, MKLinputA, MKLoutputA = seg6_loadMKLdata()

elif MKL_runA in ['P', 'QRS', 'T', 'QT', 'ST']:
    MKLinfoA, MKLinputA, MKLoutputA = int_loadMKLdata(int_name=MKL_runA)

MKLoutputA = MKLoutputA[:,:embedding_self_correlation(MKLoutputA, correlation_threshold=ESC)[1]]

# MKLOutput #2
if MKL_runB == '1seg':
    MKLinfoB, MKLinputB, MKLoutputB = loadMKLdata()

elif MKL_runB == '3seg':
    MKLinfoB, MKLinputB, MKLoutputB = seg3_loadMKLdata()

elif MKL_runB == '6seg':
    MKLinfoB, MKLinputB, MKLoutputB = seg6_loadMKLdata()

elif MKL_runB in ['P', 'QRS', 'T', 'QT', 'ST']:
    MKLinfoB, MKLinputB, MKLoutputB = int_loadMKLdata(int_name=MKL_runB)

MKLoutputB = MKLoutputB[:,:embedding_self_correlation(MKLoutputB, correlation_threshold=ESC)[1]]


#Trim MKLoutput?
MKLoutputA = MKLoutputA[:,:dims]
MKLoutputB = MKLoutputB[:,:dims]


# Compute euclidean distances between ordered sets of MKLoutputs, if not already computed
distancesA = MKLoutput_distance(MKLoutputA)
distancesB = MKLoutput_distance(MKLoutputB)

# Rank the distances from each point
ordered_positionsA = ordered_positions(distancesA)
ordered_positionsB = ordered_positions(distancesB)

# Discard the diagonal values (each point is closest to itself)
x = ordered_positionsA[~np.eye(ordered_positionsA.shape[0],dtype=bool)].reshape(ordered_positionsA.shape[0],-1).flatten()
y = ordered_positionsB[~np.eye(ordered_positionsA.shape[0],dtype=bool)].reshape(ordered_positionsA.shape[0],-1).flatten()

#Plot that
fig = plt.figure(figsize=(21, 13.5))
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
plt.title('Ranked neighbors: ' + MKL_runA + ' vs ' + MKL_runB + ' (' + str(ESC) + '% ESC) . $R^{2}$=' + str(r_value*r_value), fontsize=25)
plt.hist2d(x, y, bins=(int(MKLoutputA.shape[0]/5), int(MKLoutputB.shape[0]/5)), cmap="RdYlGn_r", norm = colors.LogNorm())
plt.xlabel(MKL_runA + ' MKL output neighbour ranking', fontsize=20)
plt.ylabel(MKL_runB + ' MKL output neighbour ranking', fontsize=20)
plt.tight_layout()
plt.show()
