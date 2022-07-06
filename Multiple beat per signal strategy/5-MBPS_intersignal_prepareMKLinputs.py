import scipy.io as spio
import numpy as np
import pickle
from utils import prepareMKLinput, seg3_prepareMKLinput, seg6_prepareMKLinput, int_prepareMKLinput, matobj_to_dict

'''After DS3 returns a list of indices for the representative subset selection for each SCP, this translates indices to 
beats, stacking them for MKL-DR, one for each lead. Additionaly, computes the reescaling performed for each 
segment as additional input'''

# What do you wanna create?

for ds3_index in [1, 4]:

    # Open the BeatsDic dictionary of resized representative beats created in BeatDissimilarityMatrix function.
    BeatsDic = pickle.load(open('_Intermediates/BeatsDic' + '.pckl', 'rb'))

    # Load .mat containing representative indices of each signal acceptable beats.
    representative_indices = spio.loadmat('4-MBPS_subset_selection/_Intermediates/ds3_indices' + str(ds3_index) + '.mat', squeeze_me=True, struct_as_record=False)['python_indices']
    representative_indices = matobj_to_dict(representative_indices)
    RepresentativeBeats = []

    # Check that segment in ds3_index=1 is included, if not, add it:
    most_representative = spio.loadmat('4-MBPS_subset_selection/_Intermediates/ds3_indices1.mat', squeeze_me=True, struct_as_record=False)['python_indices']
    most_representative = matobj_to_dict(most_representative)
    for signalname in representative_indices:
        if most_representative[signalname][0] not in representative_indices[signalname]:
            representative_indices[signalname] = np.append(representative_indices[signalname],
                                                           most_representative[signalname][0])

    # Create lists with beats info on selected beats by DS3
    for signalname in BeatsDic:
        signal_beats = list(BeatsDic[signalname].values())
        for idx in representative_indices[signalname]:
            RepresentativeBeats.append(signal_beats[idx])

    # Some metrics on the Acceptable Beats
    lengths = [s['lengths'] for s in RepresentativeBeats]
    lengths = list(map(list, zip(*lengths)))
    num_beats = [len(s) for s in representative_indices.values()]

    print('Number of acceptable beats: ' + str(len(lengths[0])))
    print('Acceptable beats per signal: ' + str(round(np.mean(num_beats), 2)) + '±' + str(
        round(np.std(num_beats), 2)))

    for i, segment in enumerate(['P', 'PQ', 'QRS', 'ST', 'T', 'TP']):
        print('Average ' + segment + ' length: ' + str(round(np.mean(lengths[i]), 2)) + '±' + str(round(np.std(lengths[i]), 2)))


    # Save the list of signals in MKLinputs in pickle
    pickle.dump(RepresentativeBeats, open('6-OBPS_intersignal_MKLs/MKL_info_ds'+ str(ds3_index) + '.pckl', 'wb'))

    # Prepare the input for main MKL. 13 features (one for each lead), with one beat per row + pre-resampling lengths.
    spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_ds'+ str(ds3_index) + '.mat', prepareMKLinput(RepresentativeBeats))

    # Prepare the input for 3seg & 6seg auxiliary MKLs.
    spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_3seg_ds'+ str(ds3_index) + '.mat', seg3_prepareMKLinput(RepresentativeBeats, denoised=denoised))
    spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_6seg_ds'+ str(ds3_index) + '.mat', seg6_prepareMKLinput(RepresentativeBeats, denoised=denoised))

    # Prepare the input for intervals auxiliary MKLs.
    for int in ['P', 'QRS', 'T', 'QT', 'ST']:
        spio.savemat('6-OBPS_intersignal_MKLs/MKLinput_' + int + '_ds'+ str(ds3_index) + '.mat', int_prepareMKLinput(RepresentativeBeats, int_name=int, denoised=denoised))