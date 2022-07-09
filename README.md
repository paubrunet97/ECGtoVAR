# ECGtoVAR

## General info
In short, after delineation, for each ECG signal, at least one cardiac cycle containing representative P, QRS, and ST periods of the signal’s cardiac cycles was selected. On those beats, MKL was run on the whole signal (PP’) and on its isolated segments (P, QRS, QT, ST, and T) to identify the more salient patterns of morphological change concerning the whole cardiac cycle or its constituent segments. An embedding is obtained for each MKL run, with coordinates quantifying the automatically-extracted morphological features of each beat. In the end, one embedding coordinate per patient is kept in each MKL embedding: on those coordinates, the correlation of clinical variables with morphological features of the ECG can be studied. Additionally, using K-Means, it is possible to cluster patients into well-differentiated ECG morphological phenogroups.

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
