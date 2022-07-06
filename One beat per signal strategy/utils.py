import numpy as np
import scipy as sp
import pickle
import os
import re
import sak
import wfdb
from scipy.integrate import trapz
import scipy.io as spio
import matplotlib.pyplot as plt
from numpy import transpose
import scipy.stats as stats

# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
def load_segmentations():
    Pon = sak.load_data(os.path.join("_Intermediates/Fiducials", "Ponsets.csv"))
    Poff = sak.load_data(os.path.join("_Intermediates/Fiducials", "Poffsets.csv"))
    QRSon = sak.load_data(os.path.join("_Intermediates/Fiducials", "QRSonsets.csv"))
    QRSoff = sak.load_data(os.path.join("_Intermediates/Fiducials", "QRSoffsets.csv"))
    Ton = sak.load_data(os.path.join("_Intermediates/Fiducials", "Tonsets.csv"))
    Toff = sak.load_data(os.path.join("_Intermediates/Fiducials", "Toffsets.csv"))

    return Pon, Poff, QRSon, QRSoff, Ton, Toff


# Computes the on-off fiducial pairs (which off is between two on's?)
def onoff_pairs(on, off):
    on = np.append(on, float('inf'))

    correspondence_matrix = (on[None, :-1] < off[:, None]) & (on[None, 1:] > off[:, None])
    on_index, off_index = np.where(correspondence_matrix)

    onoff_pairs = []
    for i in range(on_index.size):
        onoff_pairs.append((on[on_index[i]], off[off_index[i]]))

    return onoff_pairs


# Comptes the tempolar length in miliseconds of an on-off pair.
def onoff_to_ms(onoff_pairs, fs):
    durantions = []
    for (on, off) in onoff_pairs:
        durantions.append(1000*(off-on)/fs)

    return durantions


# Returns a list of fids of a given signal in temporal order.
def temporal_fids(Pon, Poff, QRSon, QRSoff, Ton, Toff):

    pon = [(fid, 'pon') for fid in Pon]
    poff = [(fid, 'poff') for fid in Poff]
    qrson = [(fid, 'qrson') for fid in QRSon]
    qrsoff = [(fid, 'qrsoff') for fid in QRSoff]
    ton = [(fid, 'ton') for fid in Ton]
    toff = [(fid, 'toff') for fid in Toff]
    fids = pon + poff + qrson + qrsoff + ton + toff
    fids.sort(key=lambda x: x[0])

    return fids


# Filter the signals
def soft_filter(signal, fs):
    try:
        signal = sp.signal.filtfilt(*sp.signal.butter(2, 0.5 / fs, 'high'), signal, axis=0)
    except:
        print('Error filtering' + '\n')
    try:
        signal = sp.signal.filtfilt(*sp.signal.butter(2, 125.0 / fs, 'low'), signal, axis=0)
    except:
        print('Error filtering'  + '\n')
    try:
        signal = sp.signal.lfilter(*sp.signal.iirnotch(50, 20.0, fs), signal, axis=0)
    except:
        print('Error filtering'  + '\n')
    try:
        signal = sp.signal.lfilter(*sp.signal.iirnotch(60, 20.0, fs), signal, axis=0)
    except:
        print('Error filtering'  + '\n')
    return signal


# Converts the matobj variable to a dictionary variable, while undoing the character changes to make MATLAB happy.
def matobj_to_dict(matobj):

    dict = {}
    for key in matobj._fieldnames:
        if type(matobj.__dict__[key]) == int:
            dict[key.replace("z","#")] = [matobj.__dict__[key]]
        else:
            dict[key.replace("z","#")] = matobj.__dict__[key]

    return dict


# Prepare the input for MKL (1seg)
def prepareMKLinput(beat_list, resizeasvector=True):

    lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Preinitialize the dictionary which will be MKLinput
    MKLinput = {}
    for lead in range(12):
        MKLinput[lead_names[lead]] = np.empty((len(beat_list), beat_list[0]['resampled_beat'].shape[0]))
    MKLinput['lengths'] = np.empty((len(beat_list), 6))

    # Loop for all beats in beat_list
    for i, beat in enumerate(beat_list):

        MKLinput['lengths'][i,:] = beat['lengths']

        for lead in range(12):
            MKLinput[lead_names[lead]][i, :] = beat['resampled_beat'][:, lead]

    # Add lengths as 1D features
    if not resizeasvector:
        MKLinput['P_length'] = MKLinput['lengths'][:, 0]
        MKLinput['PQ_length'] = MKLinput['lengths'][:, 1]
        MKLinput['QRS_length'] = MKLinput['lengths'][:, 2]
        MKLinput['ST_length'] = MKLinput['lengths'][:, 3]
        MKLinput['T_length'] = MKLinput['lengths'][:, 4]
        MKLinput['TP_length'] = MKLinput['lengths'][:, 5]
        MKLinput.pop('lengths', None)

    return MKLinput

# Load the MKL data (1seg)
def loadMKLdata():

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('6-OBPS_intersignal_MKLs/MKL_info.pckl', 'rb'))

    # Load the MKL input as a list:
    MKLinput = prepareMKLinput(MKLinfo)
    MKLinput_list = [transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else transpose(MKLinput[feature]) for feature in MKLinput]

    # Load the output matrix from MKL_incremental.
    MKLoutput = spio.loadmat('6-OBPS_intersignal_MKLs/MKLoutput.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput

# Prepare the input for MKL (3seg)
def seg3_prepareMKLinput(beat_list, resizeasvector=True):

    lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    segment_names = ['PQ', 'QRS', 'SP']
    length_indices = [0, 2, 3, 6]

    # Load the resampling lengths
    lengths = pickle.load(open('_Intermediates/lengths.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    MKLinput = {}
    # Preinitialize the dictionary which will be MKLinput
    for lead in lead_names:
        for n, segment in enumerate(segment_names):
            MKLinput[lead + '_' + segment] = \
                np.empty((len(beat_list), lengths[length_indices[n+1]]-lengths[length_indices[n]]))

    MKLinput['lengths'] = np.empty((len(beat_list), 6))

    # Loop for all beats, leads and segments in beat_list
    for i, beat in enumerate(beat_list):
        MKLinput['lengths'][i,:] = beat['lengths']

        for j, lead in enumerate(lead_names):
            for n, segment in enumerate(segment_names):
                MKLinput[lead + '_' + segment][i, :] = \
                    beat['resampled_beat'][lengths[length_indices[n]]:lengths[length_indices[n+1]], j]

    # If resize is not asked to be a vector, we'll decompress it and delete the resize key
    if not resizeasvector:
        MKLinput['P_length'] = MKLinput['lengths'][:, 0]
        MKLinput['PQ_length'] = MKLinput['lengths'][:, 1]
        MKLinput['QRS_length'] = MKLinput['lengths'][:, 2]
        MKLinput['ST_length'] = MKLinput['lengths'][:, 3]
        MKLinput['T_length'] = MKLinput['lengths'][:, 4]
        MKLinput['TP_length'] = MKLinput['lengths'][:, 5]
        MKLinput.pop('lengths', None)


    return MKLinput

# Load the MKL data (3seg)
def seg3_loadMKLdata():

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('7-mainMKL/MKL_info.pckl', 'rb'))

    # Load the MKL input as a list:
    MKLinput = prepareMKLinput(MKLinfo)
    MKLinput_list = [np.transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else np.transpose(MKLinput[feature]) for feature in MKLinput]

    # Load the output matrix from MKL_incremental.
    MKLoutput = spio.loadmat('6-OBPS_intersignal_MKLs/MKLoutput_3seg.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput

# Prepare the input for MKL (interval)
def int_prepareMKLinput(beat_list, int_name:str):
    lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Load those segment lengths, at which indices does the selected interval stop?
    lengths = pickle.load(open('_Intermediates/lengths.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    if int_name == 'P':
        int_ind = (0, 1)

    elif int_name == 'PQ':
        int_ind = (0, 2)

    elif int_name == 'QRS':
        int_ind = (2, 3)

    elif int_name == 'T':
        int_ind = (4, 5)

    elif int_name == 'QT':
        int_ind = (2, 5)

    elif int_name == 'ST':
        int_ind = (3, 5)

    # Preinitialize the dictionary which will be MKLinput
    MKLinput = {}
    for lead_num in range(12): # or augmented_less
        MKLinput[lead_names[lead_num]] = np.empty((len(beat_list), int(lengths[int_ind[1]]- lengths[int_ind[0]])))

    MKLinput['lengths'] = np.empty((len(beat_list), 6))

    # Loop for all beats in beat_list
    for i, beat in enumerate(beat_list):
        MKLinput['lengths'][i, :] = beat['lengths']
        for lead_num in range(12):
            MKLinput[lead_names[lead_num]][i, :] = beat['resampled_beat'][lengths[int_ind[0]]:lengths[int_ind[1]], lead_num]

    # Use only lengths included in the interval, but delete them 😋
    MKLinput['lengths'] = MKLinput['lengths'][:, int_ind[0]:int_ind[1]]

    assert len(beat_list) == MKLinput['lengths'].shape[0]
    assert len(beat_list) == MKLinput['I'].shape[0]

    MKLinput.pop('lengths', None)

    return MKLinput

# Load the MKL data (interval)
def int_loadMKLdata(int_name:str):

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('6-OBPS_intersignal_MKLs/MKL_info.pckl', 'rb'))

    # Load the MKL input as a list:
    MKLinput = int_prepareMKLinput(MKLinfo, int_name=int_name)
    MKLinput_list = [transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else transpose(MKLinput[feature]) for feature in MKLinput]

    MKLoutput = spio.loadmat('6-OBPS_intersignal_MKLs/MKLoutput_' + int_name +'.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput


# Load clustered MKLs
def loadMKLclustered(MKL_run):

    K_means_clusters = pickle.load(open('_Intermediates/MKLclusters_' + MKL_run + '.pckl', 'rb'))

    return K_means_clusters


# Compute the distance matrix between signals of a given MKLoutputs
def MKLoutput_distance(signalsN):
    distances = np.zeros((len(signalsN), len(signalsN)))

    for i in range(len(signalsN)):
        for j in range(i + 1, len(signalsN)):
            distances[i, j] = np.linalg.norm(signalsN[i] - signalsN[j])
    distances += np.triu(distances, 1).T

    return distances

# Given a distance matrix, order the position, index (i,j)=n -> signal j is the n-th closest to signal i,
# e.g., if (531,0)=9 -> signal 0 is the 9th closest neighbor to signal 531
def ordered_positions(distances):
    ordered_position = np.zeros((distances.shape[0], distances.shape[0]))

    for i in range(distances.shape[0]):
        ordered_position[i, :] = stats.rankdata(distances[i,:])

    return (ordered_position - 1).astype(int)

# Compute the QRS axis
def compute_axis(QRS_segment):

    # Get amplitude leads I and III
    amplitude_I = trapz(QRS_segment[:, 0], list(range(QRS_segment.shape[0])))
    amplitude_aVF = trapz(QRS_segment[:, 5], list(range(QRS_segment.shape[0])))

    # X and Y amplitudes are...
    x = 2*amplitude_aVF
    y = np.sqrt(3)*amplitude_I

    return  -np.rad2deg(np.arctan2(y, x))


def colored_line(x, y, fig, ax, col, row, z=None, linewidth=1, MAP='seismic'):
    xl = len(x)
    [xs, ys, zs] = [np.zeros((xl, 2)), np.zeros((xl, 2)), np.zeros((xl, 2))]

    # z is the line length drawn or a list of vals to be plotted
    if z == None:
        z = [0]

    for i in range(xl - 1):
        # Make a vector to thicken our line points
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        perp = np.array([-dy, dx])
        unit_perp = (perp / np.linalg.norm(perp)) * linewidth

        # Need to make 4 points for quadrilateral
        xs[i] = [x[i], x[i] + unit_perp[0]]
        ys[i] = [y[i], y[i] + unit_perp[1]]
        xs[i + 1] = [x[i + 1], x[i + 1] + unit_perp[0]]
        ys[i + 1] = [y[i + 1], y[i + 1] + unit_perp[1]]

        if len(z) == i + 1:
            z.append(z[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        # Set z values
        zs[i] = [z[i], z[i]]
        zs[i + 1] = [z[i + 1], z[i + 1]]

    cm = plt.get_cmap(MAP)
    ax[col, row].pcolormesh(xs, ys, zs, shading='gouraud', cmap=cm)

    return fig, ax
