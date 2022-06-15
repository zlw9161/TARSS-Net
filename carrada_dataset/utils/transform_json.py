import json
import numpy as np
import scipy.io as io
import os

def transform_json(json_name):
    with open(json_name, 'r') as fp:
        json_data = json.load(fp)
    sequences = json_data.keys()
    annotation_list = list()
    for sequence in sequences:
        frames = json_data[sequence].keys()
        frame_list = list()
        for frame in frames:
            instances = json_data[sequence][frame].keys()
            instance_list = list()
            for instance in instances:
                sparses = json_data[sequence][frame][instance]['sparse']
                boxes = json_data[sequence][frame][instance]['box']
                label = json_data[sequence][frame][instance]['label']
                instance_dict = {'instance_id': instance, 'sparse': sparses, 'box': boxes, 'label': label}
                instance_list.append(instance_dict)
            frame_dict = {'frame_id':frame,'instance':instance_list}
            frame_list.append(frame_dict)
        annotation = {'date': sequence, 'frame': frame_list}
        annotation_list.append(annotation)
    data = {'data':annotation_list}
    return data

if __name__ == '__main__':
    json_name = r'/data/zhangxinyan/dataset/Carrada/anno_process.json'
    save_path = r'/data/zhangxinyan/dataset/Carrada/anno_trans.json'
    data = transform_json(json_name)
    with open(save_path, 'w') as fp:
        json.dump(data, fp)