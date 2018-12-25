# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:28:07 2018

@author: kasy
"""

'''
record shows 1800 epochs have nice performance
'''

import imageio
import numpy as np
from tqdm import tqdm
from skimage import transform, measure, color, exposure, io
import glob
import tensorflow as tf
from model import classifier, class_ori
import os


#fig_path = './data/anotate_sample.jpg'
dir_path = './data/trainset_2/'
tmp_fig_name = 'b'
tmp_path = './data/tmp/'
check_path = './data/check/'
ckpt_path = './checkpoints/cell_200.ckpt'
anotate_path = './result/anotate_figs/'
process_dir = './data/process/'
#label_count = 'aq_22'

if not os.path.exists(anotate_path):
    os.makedirs(anotate_path)
if not os.path.exists(check_path):
    os.makedirs(check_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(process_dir):
    os.makedirs(process_dir)    
    
 

def process_single_fig(fig_path, label_count):
    fig = imageio.imread(fig_path)#[1000:(-1000), 1000:(-1000), :]
    fig_name = os.path.basename(fig_path)
    new_fig = np.zeros_like(fig)#contrast the green
    #imageio.imsave('./data/test_crop_ori.jpg', fig)
    fig_w = fig.shape[0]
    fig_h = fig.shape[1]
    windows_size = 400
    w_count = fig_w//windows_size
    h_count = fig_h//windows_size

    label_fig = np.zeros_like(fig[..., 0])
    def seg_crop_green(input_crop):
        '''
        first reduce the green area according to its distribution
        second enhance the green part by using non-linear function
        '''
        green_label = np.zeros_like(input_crop[..., 0])
        h, w, _ = input_crop.shape
        std_crop = np.std(input_crop[..., 1])
        mean_crop = np.mean(input_crop[..., 1])
        crop_gamma = 4/(1+np.exp(-0.35*(std_crop-20))) + 1 #=================self
        if mean_crop>75 or std_crop>20 :#调节亮暗
            enhance_fig = exposure.adjust_gamma(input_crop, gamma=crop_gamma)
        else :
            enhance_fig = np.asanyarray(input_crop)
        #print('alpha', alpha)    
        large_pixel_list = []
        mean_enhance = np.mean(enhance_fig[..., 1])
        for i in range(h):#取底色
            for j in range(w):
                if enhance_fig[i, j, 1] > mean_enhance:
                    large_pixel_list.append(enhance_fig[i, j, 1])
        base_color = (np.mean(large_pixel_list)+np.median(large_pixel_list))/2            
        if mean_crop>75 and std_crop<20:#去底色
            for i in range(h):
                for j in range(w):
                    pixel_green = enhance_fig[i, j, 1]#===================self
                    enhance_fig[i, j, 1] = (1/(1+np.exp(-(pixel_green-base_color)/10)))*pixel_green            
        
        mean_enhance = np.mean(enhance_fig[..., 1])
        std_enhance = np.std(enhance_fig[..., 1])
        green_value = mean_enhance + 3*std_enhance
        
        for i in range(h):
            for j in range(w):
                if  int(enhance_fig[i, j, 1])> green_value:
                    green_label[i, j] = 1
                # green 40, 90(liquid); blue 50 , 90(liquid)
        return green_label, enhance_fig
    
    label_fig = np.zeros_like(fig[..., 0])
    new_fig = np.zeros_like(fig)
    
    for i in tqdm(range(w_count)):
        for j in range(h_count):
            label_fig[i*windows_size:(i+1)*windows_size, j*windows_size:(j+1)*windows_size], gam = seg_crop_green(fig[i*windows_size:(i+1)*windows_size, j*windows_size:(j+1)*windows_size, :])
            new_fig[i*windows_size:(i+1)*windows_size, j*windows_size:(j+1)*windows_size, :] = gam   
    if h_count*windows_size<fig_h:
        for i in range(w_count):
            label_fig[i*windows_size:(i+1)*windows_size, h_count*windows_size:], gam = seg_crop_green(fig[i*windows_size:(i+1)*windows_size, h_count*windows_size:, :])        
            new_fig[i*windows_size:(i+1)*windows_size, h_count*windows_size:, :]= gam
    
    if w_count*windows_size<fig_w:
        for j in range(h_count):
            label_fig[w_count*windows_size:, j*windows_size:(j+1)*windows_size], gam = seg_crop_green(fig[w_count*windows_size:, j*windows_size:(j+1)*windows_size, :])
            new_fig[w_count*windows_size:, j*windows_size:(j+1)*windows_size, :] = gam
    
    if  h_count*windows_size<fig_h and w_count*windows_size<fig_w:       
            label_fig[w_count*windows_size:,h_count*windows_size:], gam = seg_crop_green(fig[w_count*windows_size:,h_count*windows_size:, :])    
            new_fig[w_count*windows_size:,h_count*windows_size:, :] = gam

    point_list = []
    io.imsave(process_dir+fig_name, new_fig)
#    contours = measure.find_contours(label_fig, 0.5)
#    
#    count = 0
#    container_point_list = []
#    task = 'label'
##    cell_area = 175
#    predict_area = 81
#    for contour in contours:
#        row_list = contour[:, 0]
#        h_min = np.min(row_list)
#        h_max = np.max(row_list)
#        
#        col_list = contour[:, 1]
#        w_min = np.min(col_list)
#        w_max = np.max(col_list)
#        # cell 180, predict 512
#        if task == 'label' :
#            if ((h_max-h_min)*(w_max-w_min)) > predict_area  :  
#                count += 1    
#                container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])   
#        count_a = 0
#        fig_data = new_fig
#        for i in container_point_list:
#            #print('1')
#            h, w = int(i[0]), int(i[1])
#            if (h-16)>0 and (h+16)<fig_w and (w-16)>0 and (w+16)<fig_h: 
#                crop_fig = fig_data[(h-16):(h+16), (w-16):(w+16), :]
#                crop_fig = transform.resize(crop_fig, [64, 64, 3])
#                try :
#                    io.imsave(tmp_path+tmp_fig_name+'{0}_{1}.jpg'.format(label_count,count_a), crop_fig)
#                    count_a += 1   
#                    point_list.append(i)
#                except :
#                    print('error occuar')
#                    continue        
    return 0
#====================begin dir processing===========
dir_file_names = glob.glob(dir_path+'*')
file_names = []
for i in dir_file_names:
    fig_files_in_dir = glob.glob(i+'/*.jpg')
    for j in fig_files_in_dir:
        file_names.append(j)
print('amount of figures ', len(file_names))
for i in range(len(file_names)):
    print('running file ', i)
    a = process_single_fig(file_names[i], str(i))        
    
