# ECGtoVAR: Quantification of 12-lead ECG dataset's patterns

## General info
ECGtoVAR is a pipeline for quantification of the main patterns of morphological variability within a 12-lead ECG dataset, resulting in the automatic feature extraction of its most pointant characteristics and possibilitating its clustering into morphological phenogroups.

In short, after delineation, for each ECG signal, at least one cardiac cycle containing representative P, QRS, and ST periods of the signal’s cardiac cycles is selected. On those beats, a dimensionality reduction algorithm is run on the whole signal (PP’) and on its isolated segments (P, QRS, QT, ST, and T) to identify the more salient patterns of morphological change, concerning the whole cardiac cycle or its constituent segments, respectively. An embedding is obtained for each MKL run, with coordinates quantifying the automatically-extracted morphological features of each beat. In the end, one embedding coordinate per patient is kept in each MKL embedding: on those coordinates, the correlation of clinical variables with morphological features of the ECG can be studied. Additionally, using K-Means, it is possible to cluster patients into well-differentiated ECG morphological phenogroups.

This is the result of my end-of-master thesis, and this project wouldn't have been possible without the collaboration of Guillermo Jiménez-Pérez. For  a detailed explanation on the steps involved in the pipeline, see the thesis document in the following link:
https://docs.google.com/document/d/11uOL-1D3-hZ3KOdIz8jXvwOl58944teF/edit?usp=sharing&ouid=117927211295833133548&rtpof=true&sd=true

### One or multiple beats per signal?
The two strategies result from the fact that MKL ought to be provided with as much morphological beat diversity as possible to have well-informed embedding spaces. Given your computational capacity for MKL to take in N cardiac cycles, if the dataset contains approximately N signals, picking only the most representative beat per signal is the goal (one beat per signal strategy). On the contrary, if the dataset contains fewer than N signals, selecting a subset of multiple morphologically-rich beats per signal is possible, in addition to the single most representative beat of the signal (multiple beats per signal strategy). After running the MKL and obtaining a more informed embedding thanks to higher beat variability, only the embeddings of representative beats may be kept, thus getting one embedding per signal.

## Requirments
To run this project, installation of the following packages is required:

* Swiss Army Knife Package for Python: 
https://github.com/guillermo-jimenez/sak

* CVX, a MATLAB-based convex modeling framework:
http://cvxr.com

### Execution
The C scripts (computeENERGY.c, computeSWA.c and computeSWB.c) need to be compiled in MATLAB. To do so just write in the Matlab command line: 

```javascript
mex computeENERGY
```
```javascript
mex computeSWA
```
```javascript
mex computeSWB
```
```javascript
mex projL1inf
```

## Technologies
Parts of the following packages have been used for the ellaboration of ECGtoVAR repository:
* Zheng, J., Zhang, J., Danioko, S. et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Sci Data 7, 48 (2020): https://github.com/zheng120/ECGDenoisingTool

* Jimenez-Perez, G., Acosta, J., Alcaine, A., & Camara, O. (2021). Generalizing electrocardiogram delineation: Training convolutional neural networks with synthetic data augmentation: https://github.com/guillermo-jimenez/ECGDelNet

* S. Sanchez-Martinez, N. Duchateau, T. Erdei, A.G. Fraser, B.H. Bijnens, and G. Piella. Characterization of myocardial motion patterns by unsupervised multiple kernel learning. Medical Image Analysis, 35:70-82, 2017: https://github.com/bcnmedtech/unsupervised_multiple_kernel_learning

* PyMKL: https://github.com/guillermo-jimenez/PyMKL
