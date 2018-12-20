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
from skimage import transform, measure, color
import glob
import tensorflow as tf
from model import classifier, class_ori


fig_path = './data/anotate_3.jpg'
tmp_path = './data/tmp/'
ckpt_path = './checkpoints/cell_1800.ckpt'
label_count = 'day_19'
fig = imageio.imread(fig_path)#[3000:4200, 3000:4200, :]
imageio.imsave('./data/test_crop_ori.jpg', fig)
fig_w = fig.shape[0]
fig_h = fig.shape[1]
windows_size = 150
w_count = fig_w//windows_size
h_count = fig_h//windows_size

#fig[100:125, 100:102, 0] = 255
#fig[100:102, 100:125, 0] = 255
#fig[125:127, 100:125, 0] = 255
#fig[100:125, 125:127, 0] = 255
#imageio.imsave('./data/anotate_test.jpg', fig)

#for fig_name in fig_path_list[2:]:



def seg_crop_green(input_crop):
    green_fig = np.zeros_like(input_crop[..., 0])
    h, w, _ = input_crop.shape
    green_slice = input_crop[..., 1]
    slice_max = np.max(green_slice)
    slice_min = np.min(green_slice)
    standard_crop = (green_slice - slice_min)/(slice_max - slice_min)
    mean = np.mean(standard_crop)
    std = np.std(standard_crop)
    if np.mean(green_slice) > 70 and np.mean(green_slice)<100 :        
        alpha = 3.0
    elif np.mean(green_slice) > 100:
        alpha = 3.5
    elif np.mean(green_slice)<70 and std < 0.18 :
        alpha = 5.0
    else :
        alpha = 2.5
    #print('alpha', alpha)    
    for i in range(h):
        for j in range(w):
            if standard_crop[i, j] - mean > (alpha * std) and (int(input_crop[i, j, 1]) - int(input_crop[i, j, 2])) > 40:
                green_fig[i, j] = 1
            # green 40, 90(liquid); blue 50 , 90(liquid)
    return green_fig    




label_fig = np.zeros_like(fig[..., 0])

for i in tqdm(range(w_count)):
    for j in range(h_count):
        label_fig[i*windows_size:(i+1)*windows_size, j*windows_size:(j+1)*windows_size] = seg_crop_green(fig[i*windows_size:(i+1)*windows_size, j*windows_size:(j+1)*windows_size, :])
if h_count*windows_size<fig_h:
    for i in range(w_count):
        label_fig[i*windows_size:(i+1)*windows_size, h_count*windows_size:] = seg_crop_green(fig[i*windows_size:(i+1)*windows_size, h_count*windows_size:, :])        
if w_count*windows_size<fig_w:
    for j in range(h_count):
        label_fig[w_count*windows_size:, j*windows_size:(j+1)*windows_size] = seg_crop_green(fig[w_count*windows_size:, j*windows_size:(j+1)*windows_size, :])
if  h_count*windows_size<fig_h and w_count*windows_size<fig_w:       
        label_fig[w_count*windows_size:,h_count*windows_size:] = seg_crop_green(fig[w_count*windows_size:,h_count*windows_size:, :])    


imageio.imsave('./data/anotate_label.jpg', 255*label_fig)   

##===============point list============
label_path_list = ['./data/anotate_label.jpg']

point_list = []
##=============create tmp figs===========
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
    predict_area = 81
    for contour in contours:
        row_list = contour[:, 0]
        h_min = np.min(row_list)
        h_max = np.max(row_list)
        
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
        elif task == 'label' :
            if ((h_max-h_min)*(w_max-w_min)) > predict_area  :  
                count += 1    
                container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])   
    count_a = 0
    fig_data = fig
    for i in container_point_list:
        #print('1')
        h, w = int(i[0]), int(i[1])
        if (h-16)>0 and (h+16)<fig_w and (w-16)>0 and (w+16)<fig_h: 
            crop_fig = fig_data[(h-16):(h+16), (w-16):(w+16), :]
            crop_fig = transform.resize(crop_fig, [64, 64, 3])
            imageio.imsave(tmp_path+'{0}_{1}.jpg'.format(label_count,count_a), crop_fig)
            count_a += 1   
            point_list.append(i)
    
#===================restore convert network ============    
        
check_point_list = point_list
check_num = len(check_point_list)
check_right_list = [] #check which is the positive

x_node = tf.placeholder(tf.float32, [1, 64, 64, 3])
y_predict = class_ori(x_node)    

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#=============sess saver restor==========
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)

for i in tqdm(range(check_num)):
    fig_train = imageio.imread(tmp_path+label_count+'_{}.jpg'.format(i))
    fig_L = color.rgb2gray(fig_train)
#    max_L = np.max(fig_L)
#    min_L = np.min(fig_L)
#    fig_L = (fig_L - min_L)/(max_L-min_L)# convert rgb into gray fig
    mean_fig = np.mean(fig_train)
    std_fig = np.std(fig_train)
    fig_train = (fig_train - mean_fig)/std_fig
    if fig_L.shape[0] != 64 or fig_L.shape[1] != 64:
        continue
    fig_train = np.reshape(fig_train, [1, 64, 64, 3])
    feed_dict = {x_node : fig_train}
    model_predict = sess.run(y_predict, feed_dict=feed_dict)
    
    if model_predict[0][0] < np.max(model_predict) :
        check_right_list.append(check_point_list[i])
        
np.savetxt('./data/anotate.txt', check_right_list)
#===========anatate on the fig==========
for i in check_right_list :
    h, w = i
    width = 14
    fig[(h-14):(h-12), (w-14):(w+14), 0] = 255
    fig[(h+12):(h+14), (w-14):(w+14), 0] = 255   
    fig[(h-14):(h+14), (w-14):(w-12), 0] = 255
    fig[(h-14):(h+14), (w+12):(w+14), 0] = 255
    
imageio.imsave('./data/anatate_model.jpg', fig)    