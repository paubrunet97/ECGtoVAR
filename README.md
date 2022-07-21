# ECGtoVAR: Automatic phenotyping and phenogrouping of ECG datasets

## General info
ECGtoVAR is a pipeline for the automatic extraction of phenotypes and phenogroups in a 12-lead ECG dataset.

In short, after delineation, for each ECG signal, at least one cardiac cycle containing representative P, QRS, and ST periods of the signal’s cardiac cycles is selected. On those beats, a dimensionality reduction algorithm is run on the whole signal (PP’) and on its isolated segments (P, QRS, QT, ST, and T) to identify the more salient patterns of morphological change, concerning the whole cardiac cycle or its constituent segments, respectively. An embedding is obtained for each MKL run, with coordinates quantifying the automatically-extracted phenotypipical traits of each beat. Signals displaying similar phenotypic traits are grouped into phenogroups. On the resulting phenotypes and phenogroups, the correlation of clinical variables with the morphological features of the 12-lead ECG can be studied.

This is the result of my end-of-master thesis, in collaboration with Guillermo Jiménez-Pérez. For a detailed explanation on the steps involved in the pipeline, see the thesis document in the following link:
https://shorturl.at/oq359

### One or multiple beats per signal?
The two strategies result from the fact that MKL ought to be provided with as much morphological beat diversity as possible to have well-informed embedding spaces. Given your computational capacity for MKL to take in N cardiac cycles, if the dataset contains approximately N signals, picking only the most representative beat per signal is the goal (one beat per signal strategy). On the contrary, if the dataset contains fewer than N signals, selecting a subset of multiple morphologically-rich beats per signal is possible, in addition to the single most representative beat of the signal (multiple beats per signal strategy). After running the MKL and obtaining a more informed embedding thanks, only the embeddings of representative beats may be kept, thus getting one phenotypic quantization per signal.

## Requirments
To run this project, installation of the following packages is required:

* Swiss Army Knife Package for Python: 
https://github.com/guillermo-jimenez/sak

* ECGDelNET models [soo to be released at]:
https://github.com/guillermo-jimenez

* CVX, a MATLAB-based convex modeling framework:
http://cvxr.com

### Execution
The C scripts (projL1inf.c, computeENERGY.c, computeSWA.c and computeSWB.c) need to be compiled in MATLAB. To do so just write in the MATLAB command line: 

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

* Jimenez-Perez, G et al. PyMKL Package: https://github.com/guillermo-jimenez/PyMKL
