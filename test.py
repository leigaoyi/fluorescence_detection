# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:25:40 2018

@author: kasy
"""

from skimage import measure,draw, transform, color 
import imageio
import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf
from model import classifier, class_ori

batch_size = 5

#label_path = './data/1_label.jpg'
#
#fig_label = imageio.imread(label_path)
#
#contours = measure.find_contours(fig_label, 0.5)
#
#fig, (ax0,ax1) = plt.subplots(1,2,figsize=(20,20))
#ax0.imshow(fig_label,plt.cm.gray)
#ax1.imshow(fig_label,plt.cm.gray)
#count = 0
#for n, contour in enumerate(contours):
#    h_min = np.min(contour, axis=0)[0]
#    h_max = np.max(contour, axis=0)[0]
#    
#    col_list = contour[:, 1]
#    w_min = np.min(col_list)
#    w_max = np.max(col_list)
#    
#    if ((h_max-h_min)*(w_max-w_min)) > 512  :  
#        count += 1
#        ax1.plot(contour[:, 1], contour[:, 0], linewidth=1)
#ax1.axis('image')
#ax1.set_xticks([])
#ax1.set_yticks([])
#plt.show()

test_path = './data/train/'
fig_data = []
label_data = []
fig_ori_data = []
for i in range(3):
    fig_path_list = glob.glob(test_path+str(i)+'/*.jpg')
    for j in fig_path_list:
        fig = imageio.imread(j)
        fig_ori_data.append(fig)
        fig_L = color.rgb2grey(fig)
        fig_data.append(fig_L)
        label_data.append(i)

x_node = tf.placeholder(tf.float32, [batch_size, 64, 64, 1])
x_ori_node = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
y_predict = classifier(x_node)
y_ori_predict = class_ori(x_ori_node)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#=================restore============
ckpt_path = './checkpoints/cell_1800.ckpt'
saver.restore(sess, ckpt_path)

#==================testing===========
output_list = np.zeros_like(label_data)

zero_amount = 0
convert_zero = 0
ori_zero = 0

acc_convert_count = 0
acc_ori_count = 0

for i in range(len(fig_data)//batch_size):
    batch_data = fig_data[i*batch_size:(i+1)*batch_size]
    ori_batch_data = fig_ori_data[i*batch_size:(i+1)*batch_size]
    batch_label = label_data[i*batch_size:(i+1)*batch_size]
    batch_data = np.reshape(batch_data, [batch_size, 64, 64, 1])
    ori_batch_data = np.reshape(ori_batch_data, [batch_size, 64, 64, 3])
    feed_dict = {x_node:batch_data, x_ori_node:ori_batch_data}
    output_model = sess.run(y_predict, feed_dict=feed_dict)
    output_label = tf.arg_max(output_model, dimension=1)
    
    ori_output_model = sess.run(y_ori_predict, feed_dict=feed_dict)
    ori_output_label = tf.arg_max(ori_output_model, dimension=1)
    output_label = sess.run(output_label)
    ori_output_label = sess.run(ori_output_label)
    count = 0
    count_ori = 0
    for j in range(batch_size):
        if batch_label[j] == output_label[j]:
            count += 1
            acc_convert_count += 1
        if batch_label[j] == ori_output_label[j]:
            count_ori += 1
            acc_ori_count += 1
        if batch_label[j] == 0:
            zero_amount += 1
            if output_label[j] == 0:
                convert_zero += 1
            if ori_output_label[j] == 0:
                ori_zero += 1
    accuracy = count/batch_size
    ori_acc = count_ori/batch_size

    #print('convert accracy', accuracy, 'ori accuracy ', ori_acc)
print(ckpt_path)
#print('label ', label_data)
print('convert averagy acc ', acc_convert_count/len(fig_data), 'convert not stain ', convert_zero/zero_amount)   
print('ori average acc ', acc_ori_count/len(fig_data), 'ori not stain ', ori_zero/zero_amount) 

