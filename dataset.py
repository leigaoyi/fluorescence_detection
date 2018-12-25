# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:40:41 2018

@author: kasy
"""

import numpy as np
import imageio
from skimage import transform, color, exposure, io
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
test_path = './data/anotate_green_1.jpg'
anotate_path = './data/anotate_model.jpg'
fig = imageio.imread(test_path)[2500:3500, 2000:4000, :]
#anotate = imageio.imread(anotate_path)[4000:2700, 4500:6500, :]

fig_label = np.zeros_like(fig[..., 1])
green_fig = fig[..., 1]
mean_green = np.mean(green_fig)
std_green = np.std(green_fig)
gam = 4/(1+np.exp(-0.35*(std_green-20))) + 1

if mean_green > 75 or std_green>20 :
    new_fig = exposure.adjust_gamma(fig, gamma=gam)#图像变暗
else :
    new_fig = np.asarray(fig)
    
light_green = np.asarray(new_fig)
light_mean = np.mean(light_green[..., 1])
light_max = np.max(light_green[..., 1])
large_pixel_list = []
for i in range(fig.shape[0]):
    for j in range(fig.shape[1]):
        if light_green[i, j, 1] > light_mean:
            large_pixel_list.append(light_green[i, j, 1])

distance = np.mean([np.mean(large_pixel_list), np.median(large_pixel_list)])

if std_green<20 and mean_green>75: #去底色
    for i in range(fig.shape[0]):
        for j in range(fig.shape[1]):
            pixel_green = light_green[i, j, 1]
            light_green[i, j, 1] = pixel_green*(1/(1+np.exp(-(pixel_green-distance)/10)))

    
mean_light = np.mean(light_green[..., 1])   
std_light = np.std(light_green[..., 1])

green_choose = mean_light + 3*std_light

imageio.imsave('./data/test_light.jpg', light_green)
print('distance ', distance)
print(gam)
print('ori mean ',mean_green)
print('ori std ', std_green)
print('light mean ', light_mean)
print(std_green)
for i in range(fig.shape[0]):
    for j in range(fig.shape[1]):
        #if int(light_green[i, j, 1]) - int(light_green[i, j, 2]) > green_sheld:
        if int(light_green[i, j, 1]) > green_choose:
            fig_label[i, j] = 1
#fig_rotate = transform.rotate(fig, 45)
io.imsave('./data/test_crop.jpg', fig)        
io.imsave('./data/test_new_fig.jpg', new_fig)    
io.imsave('./data/test_rlabel.jpg', 255*fig_label)
#imageio.imsave('./data/test_anotate.jpg', anotate)