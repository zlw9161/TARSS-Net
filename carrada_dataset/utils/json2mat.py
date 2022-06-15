# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:05:08 2021

@author: zxy
"""
import json
import numpy as np
import scipy.io as io
import os

from utils import CARRADA_HOME
from utils.configurable import Configurable


def process(json_name):
    with open(json_name, 'r') as fp:
        res = json.load(fp)
    sequences = res.keys()

    annotations = dict()
    for sequence in sequences:
        frames = res[sequence].keys()
        annotations[sequence] = dict()
        for frame in frames:
            instances = res[sequence][frame].keys()
            # if not instances:
            #     continue
            annotations[sequence][frame] = dict()
            # print(frames)
            for instance in instances:
                annotations[sequence][frame][instance] = dict()
                sparses = res[sequence][frame][instance]['range_angle']['sparse']
                boxes = res[sequence][frame][instance]['range_angle']['box']
                labels = res[sequence][frame][instance]['range_angle']['label']
                sparses, boxes = anno_filter(sparses,boxes)
                annotations[sequence][frame][instance]['sparse'] = sparses
                annotations[sequence][frame][instance]['box'] = boxes
                annotations[sequence][frame][instance]['label'] = labels
    with open(r'/data/zhangxinyan/dataset/Carrada/anno_process.json', 'w') as fp:
        json.dump(annotations, fp)



def anno_filter(sparses, boxes):
    for sparse in sparses:
        if boxes[0][0] <= sparse[0] <= boxes[1][0] and boxes[0][1] <= sparse[1] <= boxes[1][1]:
            continue
        else:
            sparses.remove(sparse)
    filter_box = dict()

    # print(sparses, boxes)
    return sparses,boxes


def json2mat(json_name, mat_path):
    if not os.path.exists(mat_path):
        os.makedirs(mat_path)
    with open(json_name, 'r') as fp:
        data = json.load(fp)
    print(type(data))
    # npy_name = r'/data/zhangxinyan/dataset/Carrada/annotations_frame_oriented.npy'
    # np.save(npy_name, data)
    # np_data = np.load(npy_name,allow_pickle=True)
    mat_name = json_name[:-5]
    mat_name = mat_name[-10:]
    mat_name = mat_name + '.mat'
    mat_name = os.path.join(mat_path, mat_name)
    print(mat_name)
    io.savemat(mat_name, data)


if __name__ == '__main__':
    json_name = r'/data/zhangxinyan/dataset/Carrada/anno_trans.json'
    mat_path = r'/data/zhangxinyan/dataset/Carrada'
    # process(json_name)
    json2mat(json_name,mat_path)

