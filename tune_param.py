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
from skimage import transform, measure, color, exposure, io, feature, morphology
import skimage
import glob
import tensorflow as tf
from model import  class_ori, classifier_v2
import os

fig_dir = './data/pre_process_4/*.jpg'
fig_list = glob.glob(fig_dir)
fig_path = fig_list[9]
label_count = 'ax1'
fig_path = './data/process/sample_ori_3.jpg'
label_name = os.path.basename(fig_path)
tmp_path = './data/tmp/'
store_anotate_path = './data/anotate/test/'
label_save_path = './data/labels/'
check_path = './data/check/'
ckpt_path = './checkpoints/cell_100.ckpt'

erosion_times = 6

fig = io.imread(fig_path)#[2000:3500, 2500:4000, :]

io.imsave('./data/test_crop_ori.jpg', fig)

fig_w = fig.shape[0]
fig_h = fig.shape[1]
windows_size = 300
w_count = fig_w//windows_size
h_count = fig_h//windows_size

anotate_path = './data/anotate_figs/test/'
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
    enhance_fig = input_crop

    mean_enhance = np.mean(enhance_fig[..., 1])
    std_enhance = np.std(enhance_fig[..., 1])
    green_value = mean_enhance + 1.5*std_enhance#chosen value
    
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
#io.imsave(process_dir+fig_name, new_fig)
##===============point list============

point_list = []
##=============create tmp figs===========

fig_label = io.imread(label_save_path+label_name)

contours = measure.find_contours(fig_label, 0.5)

count = 0
container_point_list = []
task = 'label'
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
fig_data = new_fig
for i in container_point_list:
    #print('1')
    h, w = int(i[0]), int(i[1])
    if (h-16)>0 and (h+16)<fig_w and (w-16)>0 and (w+16)<fig_h: 
        crop_fig = fig_data[(h-16):(h+16), (w-16):(w+16), :]
        crop_fig = transform.resize(crop_fig, [64, 64, 3], mode='reflect')
        
        io.imsave(tmp_path+'{0}_{1}.jpg'.format(label_count,count_a), crop_fig)
        count_a += 1   
        point_list.append(i)
    
#===================restore convert network ============    
        
check_point_list = point_list
check_num = len(check_point_list)
check_right_list = [] #check which is the positive

x_node = tf.placeholder(tf.float32, [1, 64, 64, 3])
y_ori_predict = classifier_v2(x_node, name='v2')   
#y_down1_predict = classifier_down1(x_node, name='v2_down1')
y_predict = y_ori_predict #+ y_down1_predict

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#=============sess saver restor==========
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)
fig_right_name_list = []
for i in tqdm(range(check_num)): # first time
    fig_train = io.imread(tmp_path+label_count+'_{}.jpg'.format(i))
    mean_fig = np.mean(fig_train)
    std_fig = np.std(fig_train)
    fig_train = (fig_train - mean_fig)/std_fig
#    if fig_train.shape[0] != 64 or fig_train.shape[1] != 64:
#        continue
    fig_train = np.reshape(fig_train, [1, 64, 64, 3])
    feed_dict = {x_node : fig_train}
    model_predict = sess.run(y_predict, feed_dict=feed_dict)
    
    if model_predict[0][2] == np.max(model_predict) :
        check_right_list.append(check_point_list[i])
        
        
np.savetxt('./data/anotate.txt', check_right_list)
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
for i in point_list :
    h, w = i
    h = int(h)
    w = int(w)
    width = 14
    fig[(h-14):(h-12), (w-14):(w+14), 0] = 255
    fig[(h+12):(h+14), (w-14):(w+14), 0] = 255   
    fig[(h-14):(h+14), (w-14):(w-12), 0] = 255
    fig[(h-14):(h+14), (w+12):(w+14), 0] = 255
    
    new_fig[(h-14):(h-12), (w-14):(w+14), 0] = 255
    new_fig[(h+12):(h+14), (w-14):(w+14), 0] = 255   
    new_fig[(h-14):(h+14), (w-14):(w-12), 0] = 255
    new_fig[(h-14):(h+14), (w+12):(w+14), 0] = 255
count_a = 0    
 
io.imsave(anotate_path+pre_fix+'_'+post_fix+'_{}.jpg'.format(predict_num), fig)    