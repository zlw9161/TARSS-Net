# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:29:58 2021
Threshold-based R-D filtering for CARRADA
@author: zlw
"""
import numpy as np
import math as mt
import cv2
import os
import scipy.stats as st

def threshold_filtering(rd_matrix, top_val0 = 100, top_val1 = 10):
    rng_dim = rd_matrix.shape[0]
    dop_dim = rd_matrix.shape[1]
    zero_dop_idx = mt.ceil(dop_dim/2)
    rd_vec = np.sort(rd_matrix.reshape(rng_dim * dop_dim))
    vth0 = rd_vec[-1 * top_val0]
    vth1 = rd_vec[-1 * top_val1]
    rd_matrix_thf = np.zeros([rng_dim, dop_dim])
    for i in range(rng_dim):
        for j in range(dop_dim):
            if rd_matrix[i][j] > vth0:
                rd_matrix_thf[i][j] = rd_matrix[i][j]    
    for i in range(rng_dim):
        if rd_matrix[i][zero_dop_idx] > vth1:
            rd_matrix_thf[i][zero_dop_idx] = rd_matrix[i][zero_dop_idx]
    return rd_matrix_thf, zero_dop_idx
    
def gen_rd_proposals(rd_matrix):
    rng_dim = rd_matrix.shape[0]
    dop_dim = rd_matrix.shape[1]
    rd_matrix_thf, zero_dop_idx = threshold_filtering(rd_matrix, 100, 10)
    nonzero_coords = []
    for i in range(rng_dim):
        for j in range(dop_dim):
            if rd_matrix_thf[i][j] > 0:
                nonzero_coords.append([i, j])
    # print('nonzero_coords:', nonzero_coords)
    rd_proposals = []
    for i in range(len(nonzero_coords)):
        coord = [nonzero_coords[i][0], nonzero_coords[i][1]]
        rd_proposals.append(coord)
    return rd_proposals, zero_dop_idx
    
def compute_range_velocity(rd_coord, zero_dop_idx):
    DOPPLER_RES = 0.41968030701528203
    RANGE_RES = 0.1953125
    rd_prop_range = (rd_coord[0]+1) * RANGE_RES
    rd_prop_velocity = (rd_coord[1] - zero_dop_idx) * DOPPLER_RES
    return rd_prop_range, rd_prop_velocity

def compute_match_score(a, b):
    score = -1 * np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return score

def rd_coord_calibration(rd_proposals, 
                         zero_dop_idx,
                         img_range_coord, 
                         img_velocity_coord):
    img_coord = [img_range_coord, img_velocity_coord]
    match_scores = []
    for coord in rd_proposals:
        match_scores.append(compute_match_score(coord, img_coord))
    matched_index = match_scores.index(max(match_scores))
    calibrated_coord = rd_proposals[matched_index]
    _range, _velocity = compute_range_velocity(calibrated_coord,
                                               zero_dop_idx)
    calibrated_range = _range
    calibrated_velocity = _velocity
    # print('prop:', rd_proposals)
    # print('clb:', calibrated_coord)
    # print('img:', img_range_coord, img_velocity_coord)
    return calibrated_coord, calibrated_range, calibrated_velocity




#zxy@2022060603


# configs
GUARD_CELLS = 5
BG_CELLS = 2
ALPHA = 1.3
CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
HALF_CFAR_UNITS = int(CFAR_UNITS / 2) + 1

# path
OUTPUT_IMG_DIR = "./test_out/"
INPUT_IMG_DIR = "./test_input/"
root = './range_angle_numpy/'


# 2D-CA-CFAR
def gen_rd_proposals_cfar(rd_matrix):

    inputImg = rd_matrix
    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1]))

    # search
    for i in range(inputImg.shape[0] - CFAR_UNITS):
        center_cell_x = i + BG_CELLS + GUARD_CELLS
        for j in range(inputImg.shape[1] - CFAR_UNITS):
            center_cell_y = j + BG_CELLS + GUARD_CELLS
            average = 0
            for k in range(CFAR_UNITS):
                for l in range(CFAR_UNITS):
                    if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (
                            l < (CFAR_UNITS - BG_CELLS)):
                        continue
                    average += inputImg[i + k, j + l]
            average /= (CFAR_UNITS * CFAR_UNITS) - (((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1))

            if inputImg[center_cell_x, center_cell_y] > (average * ALPHA):
                estimateImg[center_cell_x, center_cell_y] = inputImg[center_cell_x,center_cell_y]

    rd_proposals = []
    #rd_filtering
    estimateImg_thf, zero_dop_idx = rd_threshold_filtering_cfar(estimateImg,top_val0=100,top_val1=5)
    for i in range(estimateImg_thf.shape[0]):
        for j in range(estimateImg_thf.shape[1]):
            if estimateImg_thf[i][j]>0:
                rd_proposals.append([i,j])

    return rd_proposals,zero_dop_idx,estimateImg_thf


def rd_threshold_filtering_cfar(rd_matrix, top_val0 = 100, top_val1 = 10):
    rng_dim = rd_matrix.shape[0]
    dop_dim = rd_matrix.shape[1]
    zero_dop_idx = mt.ceil(dop_dim / 2)
    rd_vec = np.sort(rd_matrix.reshape(rng_dim * dop_dim))
    vth0 = rd_vec[-1 * top_val0]
    vth1 = rd_vec[-1 * top_val1]
    rd_matrix_thf = np.zeros([rng_dim, dop_dim])
    for i in range(rng_dim):
        for j in range(dop_dim):
            if rd_matrix[i][j] > vth0:
                rd_matrix_thf[i][j] = rd_matrix[i][j]
    for i in range(rng_dim):
        if rd_matrix[i][zero_dop_idx] < vth1:
            rd_matrix_thf[i][zero_dop_idx] = 0
    return rd_matrix_thf, zero_dop_idx


def compute_match_score_zxy(a,b,rd_matrix_thf):
    score = 1 / np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    t_score = score * rd_matrix_thf[a[0]][a[1]]
    return t_score


def rd_coord_calibration_zxy(rd_proposals,
                         rd_matrix_thf,
                         zero_dop_idx,
                         img_range_coord,
                         img_velocity_coord):
    img_coord = [img_range_coord, img_velocity_coord]
    match_scores = []
    for coord in rd_proposals:
        match_scores.append(compute_match_score_zxy(coord, img_coord,rd_matrix_thf))
    matched_index = match_scores.index(max(match_scores))
    calibrated_coord = rd_proposals[matched_index]
    _range, _velocity = compute_range_velocity(calibrated_coord,
                                               zero_dop_idx)
    calibrated_range = _range
    calibrated_velocity = _velocity
    # print('prop:', rd_proposals)
    # print('clb:', calibrated_coord)
    # print('img:', img_range_coord, img_velocity_coord)
    return calibrated_coord, calibrated_range, calibrated_velocity


def rd_gaussian_calibration(rd_proposals,
                         rd_matrix_thf,
                         zero_dop_idx,
                         img_range_coord,
                         img_velocity_coord):

    img_map = gaussian_distribution_img(img_range_coord,img_velocity_coord)
    radar_map = gaussian_distribution_radar(rd_proposals)
    map = img_map * radar_map

    return map


def gaussian_distribution_img(img_range_coord,img_velocity_coord):
    mean = np.array([img_range_coord, img_velocity_coord])
    cov = np.array([[10, 0], [0, 2]])  # 参数设定
    x, y = np.mgrid[0:256, 0:64]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = st.multivariate_normal(mean, cov)  # 生成多元正态分布
    # print(rv)       # <scipy.stats._multivariate.multivariate_normal_frozen object at 0x08EDDDB0> 只是生成了一个对象，并没有生成数组
    map = rv.pdf(pos)

    return map


def gaussian_distribution_radar(rd_proposals):
    c_map = np.zeros((len(rd_proposals),256,64))
    i = 0
    for coord in rd_proposals:
        mean = np.array([coord[0], coord[1]])
        cov = np.array([[0.5, 0], [0, 1]])  # 参数设定
        x, y = np.mgrid[0:256, 0:64]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = st.multivariate_normal(mean, cov)  # 生成多元正态分布
        c_map[i] = rv.pdf(pos)
        i = i + 1
    map = maximized_gaussian(c_map)
    return map


def maximized_gaussian(c_map):
    map = np.zeros((256, 64))
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            map[i][j] = max(c_map[k][i][j] for k in range(c_map.shape[0]))

    # Normalize
    sum = np.sum(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            map[i][j] = map[i][j] / sum
    return map