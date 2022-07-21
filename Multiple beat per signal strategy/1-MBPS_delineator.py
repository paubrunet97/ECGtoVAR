from functools import partial
import sak
import sak.signal
import pandas as pd
import scipy as sp
from scipy import signal
import numpy as np
import skimage
import torch
import os.path
import math
import dill
import glob
from tqdm import tqdm

def predict(signal, model, window_size=2048, stride=256, threshold_ensemble: float = 0.5, thr_dice=0.9, percentile=95,
            ptg_voting=0.5, batch_size=16, normalize=True, norm_threshold=1e-6, filter=False):
    # Preprocess signal
    signal = np.copy(signal).squeeze()
    if signal.ndim == 0:
        return np.array([])
    elif signal.ndim == 1:
        signal = signal[:, None]
    elif signal.ndim == 2:
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
    else:
        raise ValueError("2 dims max allowed")

    # Pad if necessary
    if signal.shape[0] < window_size:
        padding = math.ceil(signal.shape[0] / window_size) * window_size - signal.shape[0]
        signal = np.pad(signal, ((0, padding), (0, 0)), mode='edge')
    if (signal.shape[0] - window_size) % stride != 0:
        padding = math.ceil((signal.shape[0] - window_size) / stride) * stride - (signal.shape[0] % window_size)
        signal = np.pad(signal, ((0, padding), (0, 0)), mode='edge')

    # Get dimensions
    N, L = signal.shape

    # (Optional) Normalize amplitudes
    if normalize:
        # Get signal when it's not flat zero
        norm_signal = signal[np.all(np.abs(np.diff(signal, axis=0, append=0)) >= norm_threshold, axis=1), :]

        # High pass filter normalized signal to avoid issues with baseline wander
        norm_signal = sp.signal.filtfilt(*sp.signal.butter(2, 0.5 / 250., 'high'), norm_signal, axis=0)

        # Compute amplitude for those segments
        amplitude = np.array(sak.signal.moving_lambda(
            norm_signal,
            256,
            partial(sak.signal.amplitude, axis=0),
            axis=0
        ))
        amplitude = amplitude[np.all(amplitude > norm_threshold, axis=1),]
        amplitude = np.percentile(amplitude, percentile, axis=0)

        # Apply normalization
        signal = signal / amplitude[None, :]

    # (Optional) Filter signal
    if filter:
        signal = sp.signal.filtfilt(*sp.signal.butter(2, 0.5 / 250., 'high'), signal, axis=0)
        signal = sp.signal.filtfilt(*sp.signal.butter(2, 125.0 / 250., 'low'), signal, axis=0)
        signal = sp.signal.lfilter(*sp.signal.iirnotch(50, 20.0, 250.), signal, axis=0)
        signal = sp.signal.lfilter(*sp.signal.iirnotch(60, 20.0, 250.), signal, axis=0)

    # Avoid issues with negative strides due to filtering:
    if np.any(np.array(signal.strides) < 0):
        signal = signal.copy()

    # Data structure for computing the segmentation
    windowed_signal = skimage.util.view_as_windows(signal, (window_size, 1), (stride, 1))

    # Flat batch shape
    new_shape = (windowed_signal.shape[0] * windowed_signal.shape[1], *windowed_signal.shape[2:])
    windowed_signal = np.reshape(windowed_signal, new_shape)

    # Exchange channel position
    windowed_signal = np.swapaxes(windowed_signal, 1, 2)

    # Output structures
    windowed_mask = np.zeros((windowed_signal.shape[0], 3, windowed_signal.shape[-1]), dtype=int)

    # Check device for segmentation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute segmentation for all leads independently
    with torch.no_grad():
        if isinstance(model, list):
            for m in model:
                m = m.to(device)
                for i in range(0, windowed_signal.shape[0], batch_size):
                    inputs = {"x": torch.tensor(windowed_signal[i:i + batch_size]).float().to(device)}
                    outputs = m(inputs)["sigmoid"].cpu().detach().numpy()  #python quits here
                    windowed_mask[i:i + batch_size] += outputs > thr_dice
            windowed_mask = windowed_mask >= len(model) * threshold_ensemble
        else:
            model = model.to(device)
            for i in range(0, windowed_signal.shape[0], batch_size):
                inputs = {"x": torch.tensor(windowed_signal[i:i + batch_size]).to(device).float()}
                outputs = model(inputs)["sigmoid"].cpu().detach().numpy()
                windowed_mask[i:i + batch_size] = outputs > thr_dice

    # Retrieve mask as 1D
    counter = np.zeros((N), dtype=int)
    segmentation = np.zeros((3, N))

    # Iterate over windows
    for i in range(0, windowed_mask.shape[0], L):
        counter[(i // L) * stride:(i // L) * stride + window_size] += 1
        segmentation[:, (i // L) * stride:(i // L) * stride + window_size] += windowed_mask[i:i + L].sum(0)
    segmentation = ((segmentation / counter) >= (signal.shape[-1] * ptg_voting))

    # Correct padding
    segmentation = segmentation[:, :-padding]

    return segmentation


# Load models
models = []
for i in tqdm(range(5)):
    path = os.path.join("Directory with ECGDelNet models", f'model.{i + 1}')
    models.append(torch.load(path, pickle_module=dill).eval().float())

# Load signalpaths
basedir = "Directory with the denoised signals"
signalpaths = glob.glob(os.path.join(basedir, "*.csv"))

# Creates an ordered dictionary with the file names (e.g. SJD0001#2017_02_16#10_54_10) and its directory
signals = {}
for signalpath in signalpaths:
    signals[os.path.splitext(os.path.split(signalpath)[1])[0]] = signalpath
signals = {k: v for k, v in sorted(signals.items(), key=lambda s: s)}

# Initialize DataFrames to store fiducials
Ponset = []
Poffset = []
QRSonset = []
QRSoffset = []
Tonset = []
Toffset = []

for signalname in tqdm(list(signals)):

    #Load the signal and its sampling rate
    fs = 500 # Write the sampling frequency of the dataset
    signal = pd.read_csv(basedir + signalname + '.csv', sep=',', header=None).to_numpy()

    # Downsample signal to 250Hz item
    if fs % 250. == 0:
       signal_downsampled = sp.signal.decimate(signal, int(fs // 250), axis=0)
    else:
       signal_downsampled = sak.signal.interpolate.interp1d(signal, math.ceil(signal.shape[0] / fs * 250.), axis=0)

    # Predict signal
    segmentation = predict(signal, models, filter=not denoised)

    # Upsample segmentation to original signal shape
    segmentation = sak.signal.interpolate.interp1d(segmentation, signal.shape[0], axis=1, kind="nearest")

    # Retrieve P, QRS, T onsets/offsets
    pon, poff = sak.signal.get_mask_boundary(segmentation[0, :])
    qrson, qrsoff = sak.signal.get_mask_boundary(segmentation[1, :])
    ton, toff = sak.signal.get_mask_boundary(segmentation[2, :])

    # Add fiducials values to its respective dataframes
    pon.insert(0, signalname); Ponset.append(pon)
    poff.insert(0, signalname); Poffset.append(poff)
    qrson.insert(0, signalname); QRSonset.append(qrson)
    qrsoff.insert(0, signalname); QRSoffset.append(qrsoff)
    ton.insert(0, signalname); Tonset.append(ton)
    toff.insert(0, signalname); Toffset.append(toff)

# Finally, export dataframes with fiducials into CSV
pd.DataFrame(Ponset).to_csv('_Intermediates/Fiducials/Ponsets.csv', header=False, index=False)
pd.DataFrame(Poffset).to_csv('_Intermediates/Fiducials/Poffsets.csv', header=False, index=False)
pd.DataFrame(QRSonset).to_csv('_Intermediates/Fiducials/QRSonsets.csv', header=False, index=False)
pd.DataFrame(QRSoffset).to_csv('_Intermediates/Fiducials/QRSoffsets.csv', header=False, index=False)
pd.DataFrame(Tonset).to_csv('_Intermediates/Fiducials/Tonsets.csv', header=False, index=False)
pd.DataFrame(Toffset).to_csv('_Intermediates/Fiducials/Toffsets.csv', header=False, index=False)

