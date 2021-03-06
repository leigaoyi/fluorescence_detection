# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:28:07 2018

@author: kasy
"""

'''
record shows 1800 epochs have nice performance
'''

import numpy as np
from tqdm import tqdm
from skimage import transform, measure, color, exposure, io, morphology
import glob
import tensorflow as tf
from model import  class_ori, classifier_v2
import os


fig_path = './data/trainset_5/*'
fig_dir_list = glob.glob(fig_path)
fig_list = []
for i in fig_dir_list:
    fig_name_list = glob.glob(i+'/*.jpg')
    for j in fig_name_list :
        fig_list.append(j)
        
fig_path = fig_list[4]     
fig_path = './data/sample_ori_3.jpg'        
label_count = 'ax'
erosion_times = 3

tmp_path = './data/tmp/'
label_save_path = './data/labels/'
label_name = os.path.basename(fig_path)
#store_anotate_path = './data/sample_anotate_2.jpg'
check_path = './data/check/'
ckpt_path = './checkpoints/cell_100.ckpt'

fig = io.imread(fig_path)[2000:6000, 3000:7000, :]
io.imsave('./data/test_crop_ori.jpg', fig)
fig_w = fig.shape[0]
fig_h = fig.shape[1]
windows_size = 300
w_count = fig_w//windows_size
h_count = fig_h//windows_size

anotate_path = './data/anotate_figs/'
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
if not os.path.exists(label_save_path):
    os.makedirs(label_save_path)      

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
    green_value = mean_enhance + 1.5*std_enhance
    
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

fig_name = os.path.basename(fig_path)
edge_1 = morphology.erosion(label_fig)
for i in range(erosion_times-1):
    edge_1 = morphology.erosion(edge_1)

io.imsave(label_save_path+label_name, 255*edge_1)   
io.imsave(process_dir+fig_name, new_fig)
##===============point list============

point_list = []
##=============create tmp figs===========

fig_label = io.imread(label_save_path+label_name)

contours = measure.find_contours(fig_label, 0.5)

count = 0
container_point_list = []
cell_area = 175
predict_area = 50
for contour in contours:
    row_list = contour[:, 0]
    h_min = np.min(row_list)
    h_max = np.max(row_list)
    
    col_list = contour[:, 1]
    w_min = np.min(col_list)
    w_max = np.max(col_list)
    # cell 180, predict 512
    if ((h_max-h_min)*(w_max-w_min)) > predict_area  :  
        count += 1    
        container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])   
count_a = 0
fig_data = new_fig
select_data = []
for i in container_point_list:
    #print('1')
    h, w = int(i[0]), int(i[1])
    if (h-16)>0 and (h+16)<fig_w and (w-16)>0 and (w+16)<fig_h: 
        crop_fig = fig_data[(h-16):(h+16), (w-16):(w+16), :]
        crop_fig = transform.resize(crop_fig, [64, 64, 3], mode='reflect')  
        select_data.append(crop_fig)
        io.imsave(tmp_path+'{0}_{1}.jpg'.format(label_count,count_a), crop_fig)
        count_a += 1   
        point_list.append(i)
    
#===================restore convert network ============    
        
check_num = len(point_list)
check_right_list = [] #check which is the positive

x_node = tf.placeholder(tf.float32, [1, 64, 64, 3])
y_predict = classifier_v2(x_node, name='v2')    

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#=============sess saver restor==========
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)
index = 0
for i in select_data:
    mean = np.mean(i)
    std = np.std(i)
    fig_crop = (i-mean)/std
    fig_crop = np.reshape(fig_crop, [1, 64, 64, 3])
    feed_dict = {x_node : fig_crop}
    fig_predict = sess.run(y_predict, feed_dict=feed_dict)
    if np.max(fig_predict) == fig_predict[0][2]:
        check_right_list.append(point_list[index])
        io.imsave(check_path+label_count+'_'+str(index)+'.jpg', i)
    index += 1        
        
np.savetxt('./data/anotate.txt', check_right_list)
np.savetxt('./data/point.txt', point_list)
#===========store the right pathes chosen by network==========
        
point_list = []     
for i in check_right_list:
    exist_flag = False
    for j in range(len(point_list)):
        j_val = point_list[j]
        if np.abs(i[0]-j_val[0])<25 and np.abs(i[1]-j_val[1])<25 :
            exist_flag = True
            new_j = [(i[0]+j_val[0])/2, (i[1]+j_val[1])/2]
            point_list[j] = new_j
            #print(new_j)
            break
    if exist_flag:
        continue
    point_list.append(i)    
#print(point_list)        
predict_num = len(point_list)    
if len(fig_name)>20:
    pre_fix = fig_name[:3]
    middle = fig_name[4:7]
    post_fix = fig_name[20:23]
else :
    pre_fix = fig_name[:-4]
    middle = ''
    post_fix = ''    
    
#===========anatate on the fig==========
io.imsave(process_dir+fig_name, new_fig)    
for i in point_list :
    h, w = i
    width = 14
    h = int(h)
    w = int(w)
    
    fig[(h-14):(h-12), (w-14):(w+14), 0] = 255
    fig[(h+12):(h+14), (w-14):(w+14), 0] = 255   
    fig[(h-14):(h+14), (w-14):(w-12), 0] = 255
    fig[(h-14):(h+14), (w+12):(w+14), 0] = 255
    
    new_fig[(h-14):(h-12), (w-14):(w+14), 0] = 255
    new_fig[(h+12):(h+14), (w-14):(w+14), 0] = 255   
    new_fig[(h-14):(h+14), (w-14):(w-12), 0] = 255
    new_fig[(h-14):(h+14), (w+12):(w+14), 0] = 255
        
count_a = 0    

io.imsave(anotate_path+pre_fix+'_'+middle+'_'+post_fix+'_{}.jpg'.format(predict_num), fig)    
#io.imsave(store_anotate_path, fig)    