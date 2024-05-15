import numpy as np
import pandas as pd
from glob import glob
import sys
import os
import matplotlib.pyplot as plt

conf_cutoff = 0.8
x_cutoff = 640
y_cutoff = 480
pad_nan = False

def conf_limit(array):
    _array = array.transpose(1,0)
    mask = _array[:, 2] < conf_cutoff
    _array[mask] = [np.nan, np.nan, np.nan]
    return _array.transpose(1,0)

def coord_limit(array):
    _array = array.transpose(1,0)
    mask = (_array[:, 0] > x_cutoff) | (_array[:, 0] < 0) | (_array[:, 1] > y_cutoff) | (_array[:, 1] < 0)
    _array[mask] = [np.nan, np.nan, np.nan]
    return _array.transpose(1,0)

# def secure_row(array):
#     mask = (array[:,:, 2] == np.nan)
#     nan_len = array.shape[2]
#     fillNan = np.array([[np.nan, np.nan, np.nan]])
#     for _ in range(nan_len-1):
#         fillNan = np.concatenate((fillNan, [[np.nan, np.nan, np.nan]]))
#     fillNan = fillNan.reshape(3, -1)
#     print(fillNan)
#     print(fillNan.shape)
#     print(array.shape)
#     array[mask] = fillNan
#     return array

def secure_row(array):
    if not pad_nan:
        return array
    mask = np.isnan(array[:,:, 2])
    fillNan = np.full(array[mask].shape, np.nan)
    array[mask] = fillNan
    return array

# hdf_path = "/Users/marksong/Library/CloudStorage/ProtonDrive-2d39y@pm.me/HengenLab/data_converter/data_organized/all/JMG1XXXXX/20221121T083305-093306.h5"

def processing(path):
    hdf5_set = pd.read_hdf(path, 'df_with_missing')
    num_keypoints = len(hdf5_set.keys())/3
    if not num_keypoints.is_integer():
        raise ValueError('Number of keys is not divisible by 3 (x, y, likelihood)')
    else:
        num_keypoints = int(num_keypoints)
    np_array = []
    for key in hdf5_set.keys():
        np_array.append(list(hdf5_set[key]))
    np_array = np.array(np_array)
    np_array = np_array.reshape(num_keypoints, 3, -1)
    cutted_np_array = []
    for i in np_array:
        cutted_np_array.append(conf_limit(coord_limit(i)))
    cutted_np_array = np.array(cutted_np_array)
    cutted_np_array = secure_row(cutted_np_array.transpose(2, 1, 0))
    percent_of_valid = np.count_nonzero(~np.isnan(cutted_np_array[:,:, 2])) / cutted_np_array[:,:, 2].size
    cutted_np_array = cutted_np_array[:,:2,:]
    
    return cutted_np_array, percent_of_valid

def hdf_filename_to_save_filename(hdf_filename):
    return hdf_filename.replace('all', 'feature').replace('.h5', '.npy')

hdf_root = "data_converter/data_organized/all"

hdf_dir = glob(hdf_root+"/*")

for i in hdf_dir:
    if not os.path.exists(i.replace('all', 'feature')):
        os.makedirs(i.replace('all', 'feature'))
    for j in glob(i+"/*.h5"):
        np_array, percent_of_valid = processing(j)
        np.save(hdf_filename_to_save_filename(j), np_array)
        print(f"{j}\t{hdf_filename_to_save_filename(j)}\t{percent_of_valid*100}")