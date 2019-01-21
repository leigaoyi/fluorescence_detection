# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 09:44:08 2018

@author: kasy
"""

import tensorflow as tf
from model import  class_ori, classifier_v2
import numpy as np

from skimage import color, transform
from tqdm import tqdm
import imageio
import os
import glob

from opt import noise_add, zoom

fig_path = './data/train_gen/'
check_dir = './checkpoints/'
result_dir = './result/'
test_path = './data/test/'
batch_size = 25
epoch = 300

fig_ori_data = []
label_data = []
test_fig_data = []
test_label_data = []

if not os.path.exists(check_dir):
    os.makedirs(check_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)    

for i in range(3):
    fig_path_list = glob.glob(fig_path+str(i)+'/*.jpg')
    for j in fig_path_list:
        fig = imageio.imread(j)
        fig_ori_data.append(fig)
        label_data.append(i)
        
for i in range(3):
    fig_path_list = glob.glob(test_path+str(i)+'/*.jpg')        
    for j in fig_path_list:
        test_fig = imageio.imread(j)
        test_fig_data.append(test_fig)
        test_label_data.append(i)
        
data_num = len(fig_data)        
shuffle = [i for i in range(data_num)] 
np.random.shuffle(shuffle)

batch_idx = data_num // batch_size
#===============build model==============
x_ori_node = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])

y_node = tf.placeholder(tf.int32, [batch_size])

x_check_node = tf.placeholder(tf.float32, [1, 64, 64, 3])

#y_predict = classifier_v2(x_node) # ============new method====
y_ori_predict = classifier_v2(x_ori_node, reuse=False, name='v2')
#y_down1_predict = classifier_down1(x_ori_node, reuse=False, name='v2_down1')
y_predict = y_ori_predict

y_ori_check = classifier_v2(x_check_node, reuse=True, name='v2') # every epoch , check the accuracy
#y_down1_check = classifier_down1(x_check_node, reuse=True, name='v2_down1')
y_check = y_ori_check
y_check_list = []

loss_ori = tf.losses.sparse_softmax_cross_entropy(labels=y_node, logits=y_ori_predict)


var_list = tf.trainable_variables()
var_ori_list = [i for i in var_list if 'v2' in i.name]
opt_ori = tf.train.AdamOptimizer(0.0001)
opt_ori = opt_ori.minimize(loss_ori, var_list=var_ori_list)

#===========init ==
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(max_to_keep=10)
#===============strat train=============
for i in tqdm(range(epoch)):
    #np.random.shuffle(shuffle)
    
    for j in range(batch_idx):
        ori_data_batch = []
        train_label_batch = []
        for k in range(batch_size): # shuffle
            index = shuffle[batch_size*j+k]
            ori_fig = fig_ori_data[index]
            ori_fig = zoom(ori_fig)
            ori_fig = noise_add(ori_fig)
            #============pre-process===============
            mean_ori_fig = np.mean(ori_fig)
            std_ori_fig = np.std(ori_fig)
            ori_fig = (ori_fig - mean_ori_fig)/std_ori_fig
            ori_data_batch.append(ori_fig)
            train_label_batch.append(label_data[index])  
        ori_data_batch = np.reshape(ori_data_batch, [batch_size, 64, 64, 3])  
          
        
        #===============sess run===============
        feed_dict = {y_node:train_label_batch,
                     x_ori_node: ori_data_batch}
        #============run optimization==========
        sess.run(opt_ori, feed_dict=feed_dict)
        loss_ori_val = sess.run(loss_ori, feed_dict=feed_dict)
    
    if (i+1)%50 == 0 :
        saver.save(sess, check_dir+'cell_{0}.ckpt'.format(i+1))
        
#=========================check train every epoch=========================
    network_result = []
    test_network_result = []
    for v in fig_ori_data:
        mean_v = np.mean(v)
        std_v = np.std(v)
        v_single = (v-mean_v)/std_v
        v_single = np.reshape(v_single, [1, 64, 64, 3])
        feed_dict = {x_check_node : v_single}
        single_test = sess.run(y_check, feed_dict=feed_dict)
        network_result.append(int(np.argmax(single_test, axis=1)))
    for v in test_fig_data:
        mean_v = np.mean(v)
        std_v = np.std(v)
        v_single = (v-mean_v)/std_v
        v_single = np.reshape(v_single, [1, 64, 64, 3])
        feed_dict = {x_check_node : v_single}
        single_test = sess.run(y_check, feed_dict=feed_dict)
        test_network_result.append(int(np.argmax(single_test, axis=1)))
    acc = 0
    test_acc = 0
    for v in range(len(network_result)):
        if network_result[v] == label_data[v]:
            acc += 1
    for v in range(len(test_network_result)):
        if test_network_result[v] == test_label_data[v] :
            test_acc += 1
    epoch_acc = acc/len(network_result)  
    epoch_test_acc = test_acc/len(test_network_result)
    y_check_list.append(epoch_acc)
    print('epoch {0} ori loss {1:.4f}, acc {2:.3f}'.format(i, loss_ori_val, epoch_acc))
    test_right_count = 0
    test_right_num = 0
    for v in range(len(test_network_result)):
        if test_label_data[v]>0:
            test_right_num += 1
        if test_network_result[v]>0 and test_label_data[v]>0:
            test_right_count += 1
    print('epoch {0} test acc {1:.3f} test right acc {2:.3f} '.format(i, epoch_test_acc, test_right_count/test_right_num))        