import scipy.io as spio
import pickle
import os
from utils import seg3_prepareMKLinput, seg6_prepareMKLinput, int_prepareMKLinput, prepareMKLinput, matobj_to_dict

# Open the SCP_bestbeats dictionary of resized representative beats created in PTB_bestbeat.py function.
BeatsDic = pickle.load(open('_Intermediates/BeatsDic.pckl', 'rb'))

# Load .mat containing representative indices from its respective SCP_bestbeats lists.
mat = spio.loadmat('4-OBPS_subset_selection/_Intermediates/ds3_indices1.mat', squeeze_me=True, struct_as_record=False)
most_representative_indices = matobj_to_dict(mat['python_indices'])
MostRepresentativeBeats = []

# Create lists with beats info on selected beats by ds3_lambada1 -> MostRepresentativeBeats
for signalname in BeatsDic:
    signal_beats = list(BeatsDic[signalname].values())
    MostRepresentativeBeats.append(signal_beats[most_representative_indices[signalname][0]])

# Save the list of signals in MKLinputs in pickle
pickle.dump(MostRepresentativeBeats, open('6-OBPS_intersignal_MKLs/MKL_info.pckl', 'wb'))

# Prepare the input for main MKL. 13 features (one for each lead), with one beat per row + pre-resampling lengths.
spio.savemat('6-OBPS_intersignal_MKLs/MKLinput.mat', prepareMKLinput(MostRepresentativeBeats))

# Prepare the input for 3seg & 6seg auxiliary MKLs.
spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_3seg.mat', seg3_prepareMKLinput(MostRepresentativeBeats))
spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_6seg.mat', seg6_prepareMKLinput(MostRepresentativeBeats))

# Prepare the input for intervals auxiliary MKLs.
for int in ['P', 'PQ', 'QRS', 'T', 'QT', 'ST']:
    spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_' + int + '.mat', int_prepareMKLinput(MostRepresentativeBeats, int_name=int))
