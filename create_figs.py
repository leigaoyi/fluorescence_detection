# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:07:13 2018

@author: kasy
"""

import os
import numpy as np
from skimage import io, measure, transform
from tqdm import tqdm
import glob

pre_process_dir = './data/pre_process_1/*.jpg'
pre_process_paths = glob.glob(pre_process_dir)

store_path = './data/tmp/'
label_count = 'a'

window_size = 400
predict_area = 50

if not os.path.exists(store_path):
    os.makedirs(store_path)
fig_count = 15
for fig_name in pre_process_paths[15:]:
    fig_read = io.imread(fig_name)
    h, w = fig_read.shape[0], fig_read.shape[1]
    
    mean_fig = np.mean(fig_read[..., 1])
    std_fig = np.std(fig_read[..., 1])
    
    label_fig = np.zeros_like(fig_read[..., 1])
    print(fig_name)
    for row_pixel in tqdm(range(h)):
        for col_pixel in range(w):
            if fig_read[row_pixel, col_pixel, 1] - mean_fig > 3*std_fig:
                label_fig[row_pixel, col_pixel] = 1
                
    contours = measure.find_contours(label_fig, 0.5)
    container_point_list = []
    print('finish counters')
    for contour in contours:
        row_list = contour[:, 0]
        h_min = np.min(row_list)
        h_max = np.max(row_list)
        
        col_list = contour[:, 1]
        w_min = np.min(col_list)
        w_max = np.max(col_list)

        if ((h_max-h_min)*(w_max-w_min)) > predict_area  :    
            container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])   
    count_a = 0
    point_list = []
    fig_count += 1
    print('runing fig ', fig_count)
    for green_point in container_point_list:
        #print('1')
        h_, w_ = int(green_point[0]), int(green_point[1])
        if (h_-16)>0 and (h_+16)<h and (w_-16)>0 and (w_+16)<w: 
            crop_fig = fig_read[(h_-16):(h_+16), (w_-16):(w_+16), :]
            crop_fig = transform.resize(crop_fig, [64, 64, 3], mode='reflect')
            count_a += 1
            io.imsave(store_path+'{0}_{1}.jpg'.format(label_count+str(fig_count),count_a), crop_fig) 
            point_list.append(green_point)  
                        