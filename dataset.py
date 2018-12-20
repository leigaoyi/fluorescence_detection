# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:40:41 2018

@author: kasy
"""

import numpy as np
import imageio
from skimage import transform, color
from tqdm import tqdm
from opt import elastic_transform
#==============save crop figs=====
#fig_path = './data/3_ori.jpg'
#fig = imageio.imread(fig_path)
#
#point_list = np.loadtxt('./data/point_list.txt')
#count = 0
#for i in point_list:
#    h, w = int(i[0]), int(i[1])
#    crop_fig = fig[(h-16):(h+16), (w-16):(w+16), :]
#    crop_fig = transform.resize(crop_fig, [64, 64])
#    imageio.imsave('./data/tmp/3_{}.jpg'.format(count), crop_fig)
#    count += 1




#==============================
#===========create mask=======
#fig_path = './data/blue_cell_num.jpg'
#fig = imageio.imread(fig_path)#[2000:2500, 2000:2500, :]
#h, w, _ = fig.shape
#
#green_fig = np.zeros_like(fig[..., 0])
#for i in tqdm(range(h)):
#    for j in range(w):
#        pixel = fig[i, j, :]
#        green_fig[i, j] = (int(pixel[1]) - int(pixel[2])) > 20
#        # green 20, blue 50 , 90(liquid)
#green_fig *= 1
##imageio.imsave('./data/3.jpg', fig)
#imageio.imsave('./data/blue_cell_label.jpg', 255*green_fig)    
#
#fig_mask = np.zeros_like(fig)
#for i in range(3):
#    fig_mask[..., i] = green_fig
#imageio.imsave('./data/blue_cell_mask.jpg', fig*fig_mask)   
        
#=========convert fig into 灰度图 and elastic it
test_path = './data/anotate_2.jpg'
fig = imageio.imread(test_path)[3000:4200, 3000:4200, :]
green_mean = np.mean(fig[..., 1])
green_std = np.std(fig[..., 1])
fig_green = fig[..., 1]
green_max = np.max(fig_green)
green_min = np.min(fig_green)
fig_green = (fig_green - green_min)/(green_max-green_min)
test_mean = np.mean(fig_green)
test_std = np.std(fig_green)

fig_label = np.zeros_like(fig[..., 1])
for i in range(fig.shape[0]):
    for j in range(fig.shape[1]):
        if fig_green[i, j] - test_mean > 10*test_std:
            fig_label[i, j] = 1
#fig_rotate = transform.rotate(fig, 45)
imageio.imsave('./data/test_crop.jpg', fig)            
imageio.imsave('./data/test_rlabel.jpg', 255*fig_label)