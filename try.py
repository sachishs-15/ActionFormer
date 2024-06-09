# for trying random stuff

from data.datafile import THUMOS
from data.utils import get_data_loader

import numpy as np

file =  "/home/sachishs/Documents/ActionFormer/DATA/thumos/i3d_features/video_validation_0000906.npy"  
file = np.load(file)
print(file.shape)