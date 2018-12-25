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

test_path = './data/test/'

label_data = []
fig_ori_data = []
num_figs = []
for i in range(3):
    fig_path_list = glob.glob(test_path+str(i)+'/*.jpg')
    num_figs.append(len(fig_path_list))
    for j in fig_path_list:
        fig = imageio.imread(j)
        fig_ori_data.append(fig)

        label_data.append(i)

#x_node = tf.placeholder(tf.float32, [batch_size, 64, 64, 1])
x_ori_node = tf.placeholder(tf.float32, [1, 64, 64, 3])
#y_predict = classifier(x_node)
y_ori_predict = class_ori(x_ori_node)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#=================restore============
ckpt_path = './checkpoints/cell_300.ckpt'
saver.restore(sess, ckpt_path)

#==================testing===========
predict_list = []
predict_zero = 0
acc_count = 0
for i in fig_ori_data:
    mean_i = np.mean(i)
    std_i = np.std(i)
    test_i = (i-mean_i)/std_i
    test_i = np.reshape(test_i, [1, 64, 64, 3])
    feed_dict = {x_ori_node:test_i}
    
    y_predict_value = sess.run(y_ori_predict, feed_dict=feed_dict)
    predict_list.append(int(np.argmax(y_predict_value, axis=1)))
#print(predict_list)    
for j in range(len(predict_list)):
    if predict_list[j] == 0:
        if predict_list[j] == label_data[j] :
            predict_zero += 1
    if predict_list[j] == label_data[j] :
        acc_count += 1
print('Accuracy {0:.3f} , zeros identify {1:.3f}'.format(acc_count/np.sum(num_figs), predict_zero/num_figs[0]))            