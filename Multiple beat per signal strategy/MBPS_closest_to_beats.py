import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import loadMKLdata, MKLoutput_distance, ordered_positions, seg_loadMKLdata, int_loadMKLdata
from Dimensions.descriptors import embedding_self_correlation

''' For DS3 runs with multiple beats per signal, checks if the signals n-th closer to each beat are other beats of 
the same or coming from a different signal '''

# Which embedding dou you wanna scan?
MKL_run = 'ST'
ds3_index = 4

coda = MKL_run + '_ds' + str(ds3_index)

# Load the MKL data
if MKL_run == '1seg':
    MKLinfo, MKLinput, MKLoutput = loadMKLdata(ds3_index = ds3_index)

elif MKL_run in ['3seg', '6seg']:
    MKLinfo, MKLinput, MKLoutput = seg_loadMKLdata(MKL_run = MKL_run, ds3_index = ds3_index)

elif MKL_run in ['P', 'QRS', 'T', 'QT', 'ST']:
    MKLinfo, MKLinput, MKLoutput = int_loadMKLdata(MKL_run = MKL_run, ds3_index = ds3_index)


# How many neighbor positions wanna check?
pos = ds3_index*5
signalnames = [s['signalname'] for s in MKLinfo]
N = len(signalnames)

# Compute the dimension number with n% of info, trim the MKLoutput dimensions.
corr, dims = embedding_self_correlation(MKLoutput, correlation_threshold=0.9)
dims = 3 if dims < 3 else dims
MKLoutput = MKLoutput[:,:dims]
print('DIMS: ' + str(dims) + '\n')

# Compute the matrix and ordered distances for the MKLoutput
distance_matrix = MKLoutput_distance(MKLoutput)
ordered_distances = ordered_positions(distance_matrix)

# Create a dataframe to store the neighbors, ordered:
df_names = pd.DataFrame(np.zeros((N, N-1)), index=signalnames, columns=['Position ' + str(i+1) for i in range(N-1)])
for i in tqdm(range(N)):
    for j in range(N):
        if i != j:
            df_names.iloc[i,ordered_distances[i, j]-1] = signalnames[j]

# Put a 0 in that dataframe is the signal in i-th closest neighor is the same signal.
df_bool = np.zeros((N, N-1))
for i in tqdm(range(N)):
    for j in range(N-1):
        if df_names.iloc[i, j] == signalnames[i]:
            df_bool[i, j] = True
        else:
            df_bool[i, j] = False

df_bool = pd.DataFrame(df_bool, index=signalnames, columns=['Position ' + str(i+1) for i in range(N-1)]).iloc[:,:pos]

# Create a DataFrame with proportions:
df_proportions = pd.DataFrame(np.zeros((pos,2)), columns = ['Self', 'Other'], index =['Position ' + str(i+1) for i in range(pos)])

for i in range(pos):
    df_proportions.iloc[i, 0] = sum(df_bool.iloc[:, i])
    df_proportions.iloc[i, 1] = len(df_bool.iloc[:, i]) - sum(df_bool.iloc[:, i])

df_proportions.plot(kind='bar', stacked=True, color=['r', 'b'])
plt.title(MKL_run)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
