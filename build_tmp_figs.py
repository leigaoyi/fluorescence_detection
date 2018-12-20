# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:20:12 2018

@author: kasy
"""

import numpy as np
import os
import imageio
from skimage import transform, color, measure
from tqdm import tqdm
import glob
import pickle

dir_path = './data/prepare/'
label_path = './data/prepare/label/'
tmp_path = './data/tmp/'
fig_path_list = glob.glob(dir_path+'*.jpg')
label_count = 6
if not os.path.exists(label_path):
    os.makedirs(label_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)    
#
#for fig_name in fig_path_list[2:]:
#    fig = imageio.imread(fig_name)#[2000:3000, 2000:3000, :]
#    h, w, _ = fig.shape
#    
#    green_fig = np.zeros_like(fig[..., 0])
#    for i in tqdm(range(h)):
#        for j in range(w):
#            pixel = fig[i, j, :]
#            green_fig[i, j] = (int(pixel[1]) - int(pixel[2])) > 40
#            # green 40, blue 50 , 90(liquid)
#    green_fig *= 1
#    #imageio.imsave('./data/3.jpg', fig)
#    imageio.imsave(label_path+os.path.basename(fig_name), 255*green_fig)    

label_path_list = glob.glob(label_path+'*.jpg')

for label_name in label_path_list:
    fig_label = imageio.imread(label_name)

    contours = measure.find_contours(fig_label, 0.5)
    
    #fig, (ax0,ax1) = plt.subplots(1,2,figsize=(20,20))
    #ax0.imshow(fig_label,plt.cm.gray)
    #ax1.imshow(fig_label,plt.cm.gray)
    count = 0
    container_point_list = []
    task = 'label'
    cell_area = 175
    predict_area = 512
    for contour in contours:
        h_min = np.min(contour, axis=0)[0]
        h_max = np.max(contour, axis=0)[0]
        
        col_list = contour[:, 1]
        w_min = np.min(col_list)
        w_max = np.max(col_list)
        # cell 180, predict 512
        if task == 'cell':
            if ((h_max-h_min)*(w_max-w_min)) > cell_area  :  
                count += 1
              #  container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])
            if ((h_max-h_min)*(w_max-w_min)) > cell_area*2 and ((h_max-h_min)*(w_max-w_min))<cell_area*5 :
                count += ((h_max-h_min)*(w_max-w_min))//cell_area-1
        else :
            if ((h_max-h_min)*(w_max-w_min)) > predict_area  :  
                count += 1    
                container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])   
    point_list = container_point_list
    count = 0
    fig_data = imageio.imread(dir_path+os.path.basename(label_name))
    for i in point_list:
        #print('1')
        h, w = int(i[0]), int(i[1])
        crop_fig = fig_data[(h-16):(h+16), (w-16):(w+16), :]
        crop_fig = transform.resize(crop_fig, [64, 64])
        imageio.imsave(tmp_path+'{0}_{1}.jpg'.format(label_count,count), crop_fig)
        count += 1            
    label_count += 1

            