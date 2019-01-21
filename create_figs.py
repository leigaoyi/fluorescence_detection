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

pre_process_dir = './data/pre_process_3/*.jpg'
pre_process_paths = glob.glob(pre_process_dir)

store_path = './data/tmp/'
label_dir = './data/labels/'
label_count = 'bz'

window_size = 400
predict_area = 80

if not os.path.exists(store_path):
    os.makedirs(store_path)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)    
    
fig_count = 31
for fig_name in pre_process_paths:
    fig_read = io.imread(fig_name)
    fig_base_name = os.path.basename(fig_name)
    h, w = fig_read.shape[0], fig_read.shape[1]  
    label_fig = np.zeros_like(fig_read[..., 1])
                
    h_count = h//window_size
    w_count = w//window_size
    
    for i in tqdm(range(h_count)):
        for j in range(w_count):
            crop_fig = fig_read[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size, :]
            mean_crop = np.mean(crop_fig[..., 1])
            std_crop = np.std(crop_fig[..., 1])
            
            for row_crop in range(window_size):
                for col_crop in range(window_size):
                    if crop_fig[row_crop, col_crop, 1] - mean_crop > 3*std_crop and int(crop_fig[row_crop, col_crop, 1])>int(crop_fig[row_crop, col_crop, 2])+10:
                        label_fig[i*window_size+row_crop, j*window_size+col_crop] = 1
                           
    io.imsave(label_dir+fig_base_name, 255*label_fig)
    label_read = io.imread(label_dir + fig_base_name)
    contours = measure.find_contours(label_read, 0.5)
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
                        