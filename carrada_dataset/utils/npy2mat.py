# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:05:08 2021

@author: zlw
"""

import numpy as np
import scipy.io as io
import os

def npy2mat(npy_path, mat_path):
    npy_file_list = os.listdir(npy_path)
    if not os.path.exists(mat_path):
        os.makedirs(mat_path)
    for npy_name in npy_file_list:
        npy_name = os.path.join(npy_path, npy_name)
        tmp_name = npy_name[:-4]
        tmp_name = tmp_name[-6:]
        mat_name = tmp_name + '.mat'
        mat_name = os.path.join(mat_path, mat_name)
        data = np.load(npy_name)
        io.savemat(mat_name, {'data':data})

if __name__ == '__main__':
    json_name = r'/data/zhangxinyan/dataset/Carrada/2019-09-16-12-52-12/range_doppler_numpy'
    mat_name = r'/data/zhangxinyan/dataset/Carrada/2019-09-16-12-52-12/range_doppler_mat'
    npy2mat(json_name, mat_name)