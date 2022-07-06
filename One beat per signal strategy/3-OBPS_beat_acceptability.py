import sak
import pickle
import pandas as pd
import numpy as np
import scipy.io as spio
from tqdm import tqdm
from utils import load_segmentations, temporal_fids, onoff_pairs, soft_filter

basedir = 'Directory with the filtered signals'

# Load segmentations as dictionary of key (signal, str) and value (fiducials, np.ndarray)
Pon, Poff, QRSon, QRSoff, Ton, Toff = load_segmentations()

# Load beats as dictionary of key (signal, str) and value (fiducials, np.ndarray).
BEATon = sak.load_data('_Intermediates/Fiducials/BEATon.csv')

# Perform tests to each beat of the signal. Results are stored in a dictionary, with a key for each signal and beat.
BeatsDic = {}; OverlapDic = {}
OverlapDic['NOoverlap']       = []
OverlapDic['PQoverlap']       = []
OverlapDic['PQ+SToverlap']    = []
OverlapDic['PQ+TPoverlap']    = []
OverlapDic['SToverlap']       = []
OverlapDic['ST+TPoverlap']    = []
OverlapDic['TPoverlap']       = []
OverlapDic['PQ+ST+TPoverlap'] = []
OverlapDic['LackingP']        = []
OverlapDic['Weirdos']         = []
OverlapDic['Weirdests']       = []
OverlapDic['Pwonder']         = []
OverlapDic['QRSwonder']       = []
OverlapDic['TPwonder']        = []
OverlapDic['Weirdos']         = []
OverlapDic['Weirdests']       = []

P_lengths = []; PQ_lengths = []; Q_lengths = []; ST_lengths = []; T_lengths = []; TP_lengths = []
# Loop over all signals in the database
for signalname in tqdm(list(Pon)):

    fids = temporal_fids(Pon[signalname], Poff[signalname], QRSon[signalname], QRSoff[signalname],
                         Ton[signalname], Toff[signalname])

    BeatsDic[signalname] = {}
    # Loop over all beats (excluding first & last), set values of tests in each beat to False.
    for beaton, beatoff in zip(BEATon[signalname][1:-1], BEATon[signalname][2:]):

        basedic = BeatsDic[signalname][str(beaton)] = {}

        basedic['AcceptableBeat'] = True

        basedic['PQoverlap'] = False
        basedic['SToverlap'] = False
        basedic['TPoverlap'] = False

        basedic['signalname'] = signalname
        basedic['fs'] = 500 # Write here the signal fs.
        basedic['BEATon'] = beaton
        basedic['BEAToff'] = beatoff

        # Takes fiducials in the beat.
        beat_fids = []
        for fid in fids:
            if beaton <= fid[0] < beatoff:
                beat_fids.append(fid)

        basedic['beat_fids'] = beat_fids

        # Discard the last two fids in case (1) they are repeated pon, poff in AV node dysfunctions or
        # (2) marked as lacking P and QRSon set as BEATon, but still there are some Pon.
        while len(beat_fids) >= 2 and (beat_fids[-2][1], beat_fids[-1][1]) == ('pon', 'poff'):
            beat_fids = beat_fids[:-2]

        # Discard the second fid if it is a toff (beat before that had TP overlap)
        if len(beat_fids) >= 2 and (beat_fids[0][1], beat_fids[1][1]) == ('pon', 'toff'):
            beat_fids.pop(1)

        # List fid type of fiducials in the beat
        beat_fids_types = [beat_fid_name[1] for beat_fid_name in beat_fids]

        if len(beat_fids_types) in [5,6]:

            # 'No overlaps'
            if beat_fids_types == ['pon', 'poff', 'qrson', 'qrsoff', 'ton', 'toff']:

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[1][0])
                basedic['QRSon']  = int(beat_fids[2][0])
                basedic['QRSoff'] = int(beat_fids[3][0])
                basedic['Ton']    = int(beat_fids[4][0])
                basedic['Toff']   = int(beat_fids[5][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], basedic['QRSon'] - basedic['Poff'],
                                               basedic['QRSoff'] - basedic['QRSon'], basedic['Ton'] - basedic['QRSoff'],
                                               basedic['Toff'] - basedic['Ton'], beatoff - basedic['Toff']]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(basedic['QRSon'] - basedic['Poff'])
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(basedic['Ton'] - basedic['QRSoff'])
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(beatoff - basedic['Toff'])

                OverlapDic['NOoverlap'].append((signalname, beaton, beat_fids))

            # 'PQ overlap'
            elif beat_fids_types == ['pon', 'qrson', 'poff', 'qrsoff', 'ton', 'toff']:

                basedic['PQoverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[2][0])
                basedic['QRSon']  = int(beat_fids[1][0])
                basedic['QRSoff'] = int(beat_fids[3][0])
                basedic['Ton']    = int(beat_fids[4][0])
                basedic['Toff']   = int(beat_fids[5][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], 0,
                                               basedic['QRSoff'] - basedic['QRSon'], basedic['Ton'] - basedic['QRSoff'],
                                               basedic['Toff'] - basedic['Ton'], beatoff - basedic['Toff']]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(0)
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(basedic['Ton'] - basedic['QRSoff'])
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(beatoff - basedic['Toff'])

                OverlapDic['PQoverlap'].append((signalname, beaton, beat_fids))

            # 'PQ + ST overlap'
            elif beat_fids_types == ['pon', 'qrson', 'poff', 'ton', 'qrsoff', 'toff']:

                basedic['PQoverlap'] = True
                basedic['SToverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[2][0])
                basedic['QRSon']  = int(beat_fids[1][0])
                basedic['QRSoff'] = int(beat_fids[4][0])
                basedic['Ton']    = int(beat_fids[3][0])
                basedic['Toff']   = int(beat_fids[5][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], 0,
                                               basedic['QRSoff'] - basedic['QRSon'], 0,
                                               basedic['Toff'] - basedic['Ton'], beatoff - basedic['Toff']]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(0)
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(0)
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(beatoff - basedic['Toff'])

                OverlapDic['PQ+SToverlap'].append((signalname, beaton, beat_fids))

            # 'PQ + TP overlap'
            elif beat_fids_types == ['pon', 'qrson', 'poff', 'qrsoff', 'ton']:

                basedic['PQoverlap'] = True
                basedic['TPoverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[2][0])
                basedic['QRSon']  = int(beat_fids[1][0])
                basedic['QRSoff'] = int(beat_fids[3][0])
                basedic['Ton']    = int(beat_fids[4][0])
                basedic['Toff']   = int([pair[1] for pair in onoff_pairs(Ton[signalname], Toff[signalname]) if int(pair[0]) == int(basedic['Ton'])][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], 0,
                                               basedic['QRSoff'] - basedic['QRSon'], basedic['Ton'] - basedic['QRSoff'],
                                               basedic['Toff'] - basedic['Ton'], 0]).dot(1000/basedic['fs'])


                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(0)
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(basedic['Ton'] - basedic['QRSoff'])
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(0)

                OverlapDic['PQ+TPoverlap'].append((signalname, beaton, beat_fids))

            # 'ST overlap'
            elif beat_fids_types == ['pon', 'poff', 'qrson', 'ton', 'qrsoff', 'toff']:

                basedic['SToverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[1][0])
                basedic['QRSon']  = int(beat_fids[2][0])
                basedic['QRSoff'] = int(beat_fids[4][0])
                basedic['Ton']    = int(beat_fids[3][0])
                basedic['Toff']   = int(beat_fids[5][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], basedic['QRSon'] - basedic['Poff'],
                                               basedic['QRSoff'] - basedic['QRSon'], 0,
                                               basedic['Toff'] - basedic['Ton'], beatoff - basedic['Toff']]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(basedic['QRSon'] - basedic['Poff'])
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(0)
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(beatoff - basedic['Toff'])

                OverlapDic['SToverlap'].append((signalname, beaton, beat_fids))

            # 'ST + TP overlap'
            elif beat_fids_types == ['pon', 'poff', 'qrson', 'ton', 'qrsoff']:

                basedic['SToverlap'] = True
                basedic['TPoverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[1][0])
                basedic['QRSon']  = int(beat_fids[2][0])
                basedic['QRSoff'] = int(beat_fids[4][0])
                basedic['Ton']    = int(beat_fids[3][0])
                tonoff_pairs = onoff_pairs(Ton[signalname], Toff[signalname])
                basedic['Toff']   = int([pair[1] for pair in tonoff_pairs if int(pair[0]) == int(basedic['Ton'])][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], basedic['QRSon'] - basedic['Poff'],
                                               basedic['QRSoff'] - basedic['QRSon'], 0,
                                               basedic['Toff'] - basedic['Ton'], 0]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(basedic['QRSon'] - basedic['Poff'])
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(0)
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(0)

                OverlapDic['ST+TPoverlap'].append((signalname, beaton, beat_fids))

            # 'TP overlaps'
            elif beat_fids_types == ['pon', 'poff', 'qrson', 'qrsoff', 'ton']:

                basedic['TPoverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[1][0])
                basedic['QRSon']  = int(beat_fids[2][0])
                basedic['QRSoff'] = int(beat_fids[3][0])
                basedic['Ton']    = int(beat_fids[4][0])
                tonoff_pairs = onoff_pairs(Ton[signalname], Toff[signalname])
                basedic['Toff']   = int([pair[1] for pair in tonoff_pairs if int(pair[0]) == int(basedic['Ton'])][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], basedic['QRSon'] - basedic['Poff'],
                                               basedic['QRSoff'] - basedic['QRSon'], basedic['Ton'] - basedic['QRSoff'],
                                               basedic['Toff'] - basedic['Ton'], 0]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(basedic['QRSon'] - basedic['Poff'])
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(basedic['Ton'] - basedic['QRSoff'])
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(0)

                OverlapDic['TPoverlap'].append((signalname, beaton, beat_fids))

            # 'PQ + ST + TP overlaps'
            elif beat_fids_types == ['pon', 'qrson', 'poff', 'ton', 'qrsoff']:

                basedic['PQoverlap'] = True
                basedic['SToverlap'] = True
                basedic['TPoverlap'] = True

                basedic['Pon']    = int(beat_fids[0][0])
                basedic['Poff']   = int(beat_fids[2][0])
                basedic['QRSon']  = int(beat_fids[1][0])
                basedic['QRSoff'] = int(beat_fids[4][0])
                basedic['Ton']    = int(beat_fids[3][0])
                tonoff_pairs = onoff_pairs(Ton[signalname], Toff[signalname])
                basedic['Toff']   = int([pair[1] for pair in tonoff_pairs if int(pair[0]) == int(basedic['Ton'])][0])

                basedic['lengths'] = np.array([basedic['Poff'] - basedic['Pon'], 0,
                                               basedic['QRSoff'] - basedic['QRSon'], 0,
                                               basedic['Toff'] - basedic['Ton'], 0]).dot(1000/basedic['fs'])

                P_lengths.append(basedic['Poff'] - basedic['Pon'])
                PQ_lengths.append(0)
                Q_lengths.append(basedic['QRSoff'] - basedic['QRSon'])
                ST_lengths.append(0)
                T_lengths.append(basedic['Toff'] - basedic['Ton'])
                TP_lengths.append(0)

                OverlapDic['PQ+ST+TPoverlap'].append((signalname, beaton, beat_fids))

            else:
                basedic['AcceptableBeat'] = False
                OverlapDic['Weirdos'].append((signalname, beaton, beat_fids))

        else:
            basedic['AcceptableBeat'] = False
            OverlapDic['Weirdests'].append((signalname, beaton, beat_fids))

# Compute the 90th percentile of all lengths (so minimum amount of info gets lost)
P_length  = int(np.percentile(P_lengths, 90))
PQ_length = int(np.percentile(PQ_lengths, 90))
Q_length  = int(np.percentile(Q_lengths, 90))
ST_length = int(np.percentile(ST_lengths, 90))
T_length  = int(np.percentile(T_lengths, 90))
TP_length = int(np.percentile(TP_lengths, 90))
Total_length = P_length + PQ_length + Q_length + ST_length + T_length + TP_length

# Saves a pickle file with the ReprentativeBeats dictionary
pickle.dump([P_length,PQ_length,Q_length,ST_length,T_length,TP_length], open('_Intermediates/lengths.pckl', 'wb'))
spio.savemat('_Intermediates/lengths.mat', {'lengths': [P_length,PQ_length,Q_length,ST_length,T_length,TP_length]})

WorseSignals = []; DissimilarityMatrices = {}
# Resample all the acceptable beats, if there are more than 3 acceptable beats. Delete unacceptable beats.
for signalname in tqdm(list(BeatsDic)):

    signal = pd.read_csv(basedir + signalname + '.csv', sep=',', header=None).to_numpy()
    fs = 500 # Write here your signal sampling frequency

    # Resample each segment to have an equal length, if overlapping pad with overlapping value
    resampled_beats = []
    for beaton, beatoff in zip(BEATon[signalname][1:-1], BEATon[signalname][2:]):

        basedic = BeatsDic[signalname][str(beaton)]
        if basedic['AcceptableBeat']:

            # P segment
            if basedic['Poff'] - basedic['Pon'] == 1:
                P_seg = np.tile(signal[basedic['Pon'], :], (P_length, 1))
            else:
                P_seg = sak.signal.interpolate.interp1d(signal[basedic['Pon']:basedic['Poff'], :], P_length, axis=0)

            # PQ segment
            if basedic['PQoverlap'] or basedic['QRSon'] - basedic['Poff'] == 1:
                PQ_seg = np.tile(signal[basedic['QRSon'], :], (PQ_length, 1))
            else:
                PQ_seg = sak.signal.interpolate.interp1d(signal[basedic['Poff']:basedic['QRSon'], :], PQ_length, axis=0)

            # Q segment
            if basedic['QRSoff'] - basedic['QRSon'] == 1:
                Q_seg = np.tile(signal[basedic['QRSoff'], :], (Q_length, 1))
            else:
                Q_seg = sak.signal.interpolate.interp1d(signal[basedic['QRSon']:basedic['QRSoff'], :], Q_length, axis=0)

            # ST segment
            if basedic['SToverlap'] or basedic['Ton'] - basedic['QRSoff'] == 1:
                ST_seg = np.tile(signal[basedic['Ton'], :], (ST_length, 1))
            else:
                ST_seg = sak.signal.interpolate.interp1d(signal[basedic['QRSoff']:basedic['Ton'], :], ST_length, axis=0)

            # T segment
            if basedic['Toff'] - basedic['Ton'] == 1:
                T_seg = np.tile(signal[basedic['Toff'], :], (T_length, 1))
            else:
                T_seg = sak.signal.interpolate.interp1d(signal[basedic['Ton']:basedic['Toff'], :], T_length, axis=0)

            # TP segment
            if basedic['TPoverlap'] or beatoff - basedic['Toff'] == 1:
                TP_seg = np.tile(signal[beatoff, :], (TP_length, 1))
            else:
                TP_seg = sak.signal.interpolate.interp1d(signal[basedic['Toff']:beatoff, :], TP_length, axis=0)

            # Concatenate all segments and save it to BeatsDic
            resampled_beat = np.concatenate((P_seg, PQ_seg, Q_seg, ST_seg, T_seg, TP_seg), axis=0)
            assert len(resampled_beat) == Total_length, "Signal" + signalname + str(beaton)

            basedic['resampled_beat'] = resampled_beat

        # If the beat is not acceptable, remove it from BeatsDic
        else:
            del BeatsDic[signalname][str(beaton)]

    # If the signal has less than three acceptable beats, remove it (too noisy)
    if len(BeatsDic[signalname]) < 3:
        del BeatsDic[signalname]
        WorseSignals.append(signalname)

# Saves a pickle file with the Beats dictionary.
pickle.dump(BeatsDic, open('_Intermediates/BeatsDic.pckl', 'wb'))
