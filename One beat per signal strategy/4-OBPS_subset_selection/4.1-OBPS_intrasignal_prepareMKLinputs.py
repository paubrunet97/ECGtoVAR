import pickle
from utils import seg3_prepareMKLinput
import scipy.io as spio
from tqdm import tqdm

basedic = '/Users/paubrunet/Google Drive/Documents/TFM/LongQT/_Intermediates/'
denoised = True

BeatsDic = pickle.load(open('../_Intermediates/BeatsDic.pckl', 'rb'))

for signalname in tqdm(BeatsDic):
    MKLinput = seg3_prepareMKLinput(list(BeatsDic[signalname].values()))
    spio.savemat('_Intermediates/' + signalname + '.mat', MKLinput)
