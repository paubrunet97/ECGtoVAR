import numpy as np
import pickle
import os
import scipy.stats as stats
import sak
import scipy.io as spio
from numpy import transpose
from matplotlib import pyplot as plt


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
    off = np.asarray(off)

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


# Returns the similarity matrix for a list of beats of same length. If regularize, it has values between 0 and 1.
def similarity_matrix(resampled_beats, lengths, regularize=True):
    n_beats = len(lengths)
    xcorr_matrix_leads = np.zeros((n_beats, n_beats))
    xcorr_matrix_lengths = np.zeros((n_beats, n_beats))

    for i in range(n_beats):
        for j in range(i+1, n_beats):
            corr_12leads = []
            for lead in range(resampled_beats[0].shape[1]):
                corr = sak.signal.xcorr(resampled_beats[i][:, lead], resampled_beats[j][:, lead], maxlags=0)[0]
                corr_12leads.append(float(corr))
            xcorr_matrix_leads[i, j] = np.mean(corr_12leads)
    xcorr_matrix_leads += np.triu(xcorr_matrix_leads, 1).T

    for i in range(n_beats):
        for j in range(i+1, n_beats):
            xcorr_matrix_lengths[i, j] = sak.signal.xcorr(lengths[i], lengths[j], maxlags=0)[0]
    xcorr_matrix_lengths += np.triu(xcorr_matrix_lengths, 1).T
    xcorr_matrix = (xcorr_matrix_leads + xcorr_matrix_lengths) / 2

    if regularize:
        xcorr_matrix_leads = 0.5 + xcorr_matrix_leads/2
        xcorr_matrix_lengths = 0.5 + xcorr_matrix_lengths/2

        xcorr_matrix = (xcorr_matrix_leads + xcorr_matrix_lengths)/2
        np.fill_diagonal(xcorr_matrix, 1)

    return xcorr_matrix


# Converts the matobj variable to a dictionary variable, while undoing the character changes to make MATLAB happy.
def matobj_to_dict(matobj):

    dict = {}
    for key in matobj._fieldnames:
        if type(matobj.__dict__[key]) == int:
            dict[key] = [matobj.__dict__[key]]
        else:
            dict[key] = matobj.__dict__[key]

    return dict


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

# Prepare the input for MKL (1seg)
def prepareMKLinput(beat_list):

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

    return MKLinput

# Load the MKL data (1seg)
def loadMKLdata(ds3_index: int):

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('6-OBPS_intersignal_MKLs/MKL_info_ds' + str(ds3_index) + '.pckl', 'rb'))


    # Load the MKL input as a list:
    MKLinput = prepareMKLinput(MKLinfo)
    MKLinput_list = [transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else transpose(MKLinput[feature]) for feature in MKLinput]

    # Load the output matrix from MKL_incremental.
    MKLoutput = spio.loadmat('6-OBPS_intersignal_MKLs/MKLoutput_ds' + str(ds3_index) + '.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput

# Prepare the input for MKL (3seg)
def seg3_prepareMKLinput(beat_list):

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

    return MKLinput

# Prepare the input for MKL (6seg)
def seg6_prepareMKLinput(beat_list):
    lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    segment_names = ['P', 'PQ', 'QRS', 'ST', 'T', 'TP']
    length_indices = [0, 1, 2, 3, 4, 5, 6]

    # Load the resampling lengths
    lengths = pickle.load(open('_Intermediates/lengths.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    MKLinput = {}
    # Preinitialize the dictionary which will be MKLinput
    for lead in lead_names:
        for n, segment in enumerate(segment_names):
            MKLinput[lead + '_' + segment] = \
                np.empty((len(beat_list), lengths[length_indices[n + 1]] - lengths[length_indices[n]]))

    MKLinput['lengths'] = np.empty((len(beat_list), 6))

    # Loop for all beats, leads and segments in beat_list
    for i, beat in enumerate(beat_list):
        MKLinput['lengths'][i, :] = beat['lengths']

        for j, lead in enumerate(lead_names):
            for n, segment in enumerate(segment_names):
                MKLinput[lead + '_' + segment][i, :] = \
                    beat['resampled_beat'][lengths[length_indices[n]]:lengths[length_indices[n + 1]], j]

    return MKLinput

# Load the MKL data (6seg)
def seg_loadMKLdata(MKL_run:str, ds3_index:int):

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('6-OBPS_intersignal_MKLs/MKL_info_ds' + str(ds3_index) + '.pckl', 'rb'))

    # Load the MKL input as a list:
    MKLinput = prepareMKLinput(MKLinfo)
    MKLinput_list = [np.transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else np.transpose(MKLinput[feature]) for feature in MKLinput]

    # Load the output matrix from MKL_incremental.
    MKLoutput = spio.loadmat('6-OBPS_intersignal_MKLs/MKLoutput_' + MKL_run + '_ds' + str(ds3_index) + '.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput


# Prepare the input for MKL (interval)
def int_prepareMKLinput(beat_list, int_name:str):
    lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Load those segment lengths, at which indices does the selected interval stop?
    lengths = pickle.load(open('_Intermediates/lengths.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    if int_name == 'P':
        int_ind = (0, 1)

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


    MKLinput.pop('lengths', None)

    return MKLinput


# Load the MKL data (interval)
def int_loadMKLdata(MKL_run:str, ds3_index:int):

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('6-OBPS_intersignal_MKLs/MKL_info_ds' + str(ds3_index) + '.pckl', 'rb'))

    # Load the MKL input as a list:
    MKLinput = int_prepareMKLinput(MKLinfo, MKL_run)
    MKLinput_list = [transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else transpose(MKLinput[feature]) for feature in MKLinput]

    MKLoutput = spio.loadmat('6-OBPS_intersignal_MKLs/MKLoutput_' + MKL_run + '_ds' + str(ds3_index) + '.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput


def colored_line(x, y, fig, ax, col, row, z=None, linewidth=1, MAP='seismic'):
    # this uses pcolormesh to make interpolated rectangles
    xl = len(x)
    [xs, ys, zs] = [np.zeros((xl, 2)), np.zeros((xl, 2)), np.zeros((xl, 2))]

    # z is the line length drawn or a list of vals to be plotted
    if z == None:
        z = [0]

    for i in range(xl - 1):
        # make a vector to thicken our line points
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        perp = np.array([-dy, dx])
        unit_perp = (perp / np.linalg.norm(perp)) * linewidth

        # need to make 4 points for quadrilateral
        xs[i] = [x[i], x[i] + unit_perp[0]]
        ys[i] = [y[i], y[i] + unit_perp[1]]
        xs[i + 1] = [x[i + 1], x[i + 1] + unit_perp[0]]
        ys[i + 1] = [y[i + 1], y[i + 1] + unit_perp[1]]

        if len(z) == i + 1:
            z.append(z[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        # set z values
        zs[i] = [z[i], z[i]]
        zs[i + 1] = [z[i + 1], z[i + 1]]

    cm = plt.get_cmap(MAP)
    ax[col, row].pcolormesh(xs, ys, zs, shading='gouraud', cmap=cm)

    return fig, ax

def ECG_to_VCG(ecg, method):
    dower_matrix = np.array([[-0.172, -0.074, 0.122, 0.231, 0.239, 0.194, 0.156, -0.010],
                             [0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227, 0.887],
                             [-0.229, -0.310, -0.246, -0.063, 0.055, 0.108, 0.022, 0.102]])

    kors_matrix = np.array([[-0.13, 0.05, -0.01, 0.14, 0.06, 0.54, 0.38, -0.07],
                            [0.06, -0.02, -0.05, 0.06, -0.17, 0.13, -0.07, 0.93],
                            [-0.43, -0.06, -0.14, -0.20, -0.11, 0.31, 0.11, -0.23]])

    ecg_ordered = np.zeros((ecg.shape[0], 8))
    ecg_ordered[:, 0:6] = ecg[:, 6:12]
    ecg_ordered[:, 6:8] = ecg[:, 0:2]
    if method == 'dower':
        vcg = np.matmul(ecg_ordered, dower_matrix.T)
    elif method == 'kors':
        vcg = np.matmul(ecg_ordered, kors_matrix.T)

    return vcg

def seg1_VCGprepareMKLinput(beat_list:list, method:str):

    lead_names = ['X', 'Y', 'Z']

    MKLinput = {}

    # Preinitialize the dictionary which will be MKLinput
    for lead in lead_names:
        MKLinput[lead] = np.empty((len(beat_list), beat_list[0]['resampled_' + method].shape[0]))

    MKLinput['lengths'] = np.empty((len(beat_list), 3))

    # Loop for all beats in beat_list
    for i, beat in enumerate(beat_list):
        MKLinput['lengths'][i,:] = beat['lengths'][2:5]

        for lead in range(len(lead_names)):
            MKLinput[lead_names[lead]][i, :] = beat['resampled_' + method][:, lead]

    MKLinput.pop('lengths', None)


    return MKLinput

def seg2_VCGprepareMKLinput(beat_list:list, method:str):

    lead_names = ['X', 'Y', 'Z']
    segment_names = ['QRS', 'SP']
    length_indices = [2, 3, 5]

    # Load the resampling lengths
    lengths = pickle.load(open('_Intermediates/lengths.pckl', 'rb'))
    lengths = [0] + [sum(lengths[:i + 1]) for i in range(len(lengths))]

    MKLinput = {}
    # Preinitialize the dictionary which will be MKLinput
    for lead in lead_names:
        for n, segment in enumerate(segment_names):
            MKLinput[lead + '_' + segment] = np.empty((len(beat_list), lengths[n+1]-lengths[n]))

    MKLinput['lengths'] = np.empty((len(beat_list), 6))

    # Loop for all beats, leads and segments in beat_list
    for i, beat in enumerate(beat_list):
        MKLinput['lengths'][i,:] = beat['lengths']

        for j, lead in enumerate(lead_names):
            for n, segment in enumerate(segment_names):
                MKLinput[lead + '_' + segment][i, :] = \
                    beat['resampled_' + method][lengths[n]:lengths[n+1], j]

    MKLinput.pop('lengths', None)


    return MKLinput

def VCGloadMKLdata(MKL_run):

    # Open the MKL info pickle
    MKLinfo = pickle.load(open('_Intermediates/MKLinfo_' + MKL_run.split('_')[2] + '.pckl', 'rb'))

    # Load the MKL input as a list:
    MKLinput = seg1_VCGprepareMKLinput(MKLinfo, method = MKL_run.split('_')[1])
    MKLinput_list = [np.transpose(MKLinput[feature][:, np.newaxis]) if MKLinput[feature].ndim == 1
                     else np.transpose(MKLinput[feature]) for feature in MKLinput]

    # Load the output matrix from MKL_incremental.
    MKLoutput = spio.loadmat('4-MKL/MKLoutput_' + MKL_run + '.mat')['MKLoutput']

    return MKLinfo, MKLinput_list, MKLoutput