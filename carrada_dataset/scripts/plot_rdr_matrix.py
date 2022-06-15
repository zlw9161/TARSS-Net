"""Script to generate RD points """
import os
import json
import time
import glob
import numpy as np
import pandas as pd

from carrada_dataset.utils import CARRADA_HOME
from carrada_dataset.utils.configurable import Configurable
from carrada_dataset.utils.visualize_signal import SignalVisualizer

from carrada_dataset.annotation_generators.rd_points_generator import RDPointsGenerator
# Class DataLoader(Configurable)
#
#
# def plt_matrix()


def main():
    print('***** Step 1/4: Plot Rang-Doppler Matrixes *****')
    time1 = time.time()
    config_path = os.path.join(CARRADA_HOME, 'config.ini')
    config = Configurable(config_path).config
    warehouse = config['data']['warehouse']
    carrada = os.path.join(warehouse, 'Carrada')
    with open(os.path.join(carrada, 'data_seq_ref.json'), 'r') as fp:
        ref_data = json.load(fp)
    with open(os.path.join(carrada, 'validated_seqs.txt')) as fp:
        seq_names = fp.readlines()
    seq_names = [seq.replace('\n', '') for seq in seq_names]
    for seq_name in seq_names:
        print('*** Processing sequence {} ***'.format(seq_name))
        instances = ref_data[seq_name]['instances']
        n_points = 1
        time_window = 10
        generator = RDPointsGenerator(seq_name, n_points, instances, time_window)
        _, _ = generator.get_rd_points(save_rd_imgs=False,
                                       save_points=False,
                                       save_clb_points=True,
                                       save_points_coordinates=False,
                                       save_world_points=False)
    print('***** Execution Time for Step 2/4:'
          ' {} secs. *****'.format(round(time.time() - time1, 2)))

if __name__ == '__main__':
    main()
